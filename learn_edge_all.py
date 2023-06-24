"""Unified interface to all dynamic graph model experiments"""
import torch
import numpy as np
import time

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from module import THAN
import utils
from utils import RandHetEdgeSampler, HetMiniBatchSampler, MiniBatchSampler


args = utils.get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROPOUT = args.dropout
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = args.use_memory
# can`t shuffle if use memory
SHUFFLE = args.shuffle
if USE_MEMORY:
    SHUFFLE = False

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

utils.check_dirs()

logger = utils.get_logger(args.prefix+"_"+args.data+"_bs"+str(BATCH_SIZE))
logger.info(args)

utils.set_random_seed(2022)


def evaluate_score(size, pos_prob, neg_prob):
    pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    auc = roc_auc_score(true_label, pred_score)
    ap = average_precision_score(true_label, pred_score)
    return ap, auc


def eval_one_epoch(hint, model: THAN, sampler: RandHetEdgeSampler, batch_sampler, data):
    logger.info(hint)
    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        batch_sampler.reset()
        while True:
            batch_idx = batch_sampler.get_batch_index()
            if batch_idx is None or len(batch_idx)==0:
                break

            src_l_cut = data.src_l[batch_idx]
            dst_l_cut = data.dst_l[batch_idx]
            ts_l_cut = data.ts_l[batch_idx]
            src_utype_l = data.u_type_l[batch_idx]
            tgt_utype_l = data.v_type_l[batch_idx]
            etype_l = data.e_type_l[batch_idx]

            size = len(src_l_cut)
            dst_l_fake = sampler.sample_dst_by_ntype_list(tgt_utype_l)
            fake_utype_l = tgt_utype_l

            pos_prob, neg_prob = model.link_contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, 
                                        src_utype_l, tgt_utype_l, fake_utype_l, etype_l, NUM_NEIGHBORS)

            ap, auc = evaluate_score(size, pos_prob, neg_prob)
            val_ap.append(ap)
            val_auc.append(auc)

    return np.mean(val_auc), np.mean(val_ap)


# load data and split into train val test
g, train, test = utils.load_and_split_data_train_test(DATA, args.n_dim, args.e_dim)

### Initialize the data structure for graph and edge sampling
train_ngh_finder = utils.get_neighbor_finder(train, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
full_ngh_finder = utils.get_neighbor_finder(g, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
# negative edge sampler
train_rand_sampler = RandHetEdgeSampler(train.src_l, train.dst_l, train.u_type_l, train.v_type_l)
test_rand_sampler = RandHetEdgeSampler(g.src_l, g.dst_l, g.u_type_l, g.v_type_l)
# mini-batch idx sampler
# train_batch_sampler = HetMiniBatchSampler(g.num_e_type, train.e_type_l, BATCH_SIZE, SHUFFLE)
# test_batch_sampler = HetMiniBatchSampler(g.num_e_type, test.e_type_l, BATCH_SIZE, hint='test')
train_batch_sampler = MiniBatchSampler(len(train.e_idx_l), BATCH_SIZE)
test_batch_sampler = MiniBatchSampler(len(test.e_idx_l), BATCH_SIZE, hint="test")


device = torch.device('cuda:{}'.format(GPU))

auc_l, ap_l = [], []
for i in range(args.n_runs):
    logger.info(f"【START】run num: {i}")

    ### Model initialize
    model = THAN(train_ngh_finder, g.n_feat, g.e_feat, g.e_type_feat, g.num_n_type, g.num_e_type, args.t_dim, num_layers=NUM_LAYER, use_time=USE_TIME, seq_len=SEQ_LEN, n_head=NUM_HEADS, dropout=DROPOUT, device=device, use_memory=USE_MEMORY, msg_agg=args.msg_agg)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    model = model.to(device)
    
    epoch_times = []
    best_auc, best_ap = 0., 0.
    for epoch in range(NUM_EPOCH):
        # training use only training graph
        start_time = time.time()
        
        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            model.memory.__init_memory__()
        
        model.ngh_finder = train_ngh_finder
        ap, auc, m_loss = [], [], []
        logger.info('start {} epoch'.format(epoch))
        train_batch_sampler.reset()
        while True:
            batch_idx = train_batch_sampler.get_batch_index()
            if batch_idx is None or len(batch_idx)==0:
                break

            src_l_cut, dst_l_cut = train.src_l[batch_idx], train.dst_l[batch_idx]
            ts_l_cut = train.ts_l[batch_idx]
            src_utype_l = train.u_type_l[batch_idx]
            tgt_utype_l = train.v_type_l[batch_idx]
            etype_l = train.e_type_l[batch_idx]

            dst_l_fake = train_rand_sampler.sample_dst_by_ntype_list(tgt_utype_l)
            bgd_utype_l = tgt_utype_l

            size = len(src_l_cut)

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)
                lbl = torch.cat((pos_label, neg_label), dim=0)

            optimizer.zero_grad()
            model = model.train()
            pos_prob, neg_prob = model.link_contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, 
                                            src_utype_l, tgt_utype_l, bgd_utype_l, etype_l, NUM_NEIGHBORS)

            loss = criterion(torch.cat((pos_prob, neg_prob), dim=0), lbl)
            loss += args.beta * model.affinity_score.reg_loss()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model = model.eval()
                _ap, _auc = evaluate_score(size, pos_prob, neg_prob)
                ap.append(_ap)
                auc.append(_auc)
                m_loss.append(loss.item())
            
            # Detach memory after each of batch so we don't backpropagate to
            if USE_MEMORY:
                model.memory.detach_memory()
            

        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = model.memory.backup_memory()
        
        # validation phase use all information
        model.ngh_finder = full_ngh_finder
        test_auc, test_ap = eval_one_epoch('test', model, test_rand_sampler, test_batch_sampler, test)
        
        if USE_MEMORY:
            model.memory.restore_memory(train_memory_backup)
        
        end_time = time.time()
        epoch_times.append(end_time - start_time)
        
        logger.info('epoch: {}, time: {:.1f}'.format(epoch, end_time - start_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train: auc: {:.4f}, ap: {:.4f}'.format(np.mean(auc), np.mean(ap)))
        logger.info('test: auc: {:.4f}, ap: {:.4f}'.format(test_auc, test_ap))
        
        if test_auc > best_auc:
            best_auc, best_ap = test_auc, test_ap
        
        torch.save(model.state_dict(), get_checkpoint_path(epoch))

    logger.info('Test Best: auc: {:.4f}, ap: {:.4f}\n\n'.format(best_auc, best_ap))

    auc_l.append(best_auc)
    ap_l.append(best_ap)

    # save epoch times
    with open(f"epoch_time/{args.prefix}_{args.data}_layer{args.n_layer}.txt", 'a', encoding='utf-8') as f:
        f.write(",".join(map(lambda x: format(x, ".1f"), epoch_times)))
        f.write("\n")

logger.info("Final result: \nauc: {:.2f}({:.2f}), ap: {:.2f}({:.2f})".format(
    np.mean(auc_l)*100, np.std(auc_l)*100, np.mean(ap_l)*100, np.std(ap_l)*100))
