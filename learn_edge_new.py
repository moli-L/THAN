"""Unified interface to all dynamic graph model experiments"""
import math
import torch
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from module import THAN
import utils
from utils import EarlyStopMonitor, RandHetEdgeSampler, HetMiniBatchSampler


args = utils.get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROPOUT = args.dropout
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
SHUFFLE = True # args.shuffle


MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.data}-{epoch}.pth'

logger = utils.get_logger()
logger.info(args)

utils.set_random_seed(2021)



def evaluate_score(size, pos_prob, neg_prob):
    pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
    pred_label = pred_score > 0.5
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    acc = (pred_label == true_label).mean()
    ap = average_precision_score(true_label, pred_score)
    auc = roc_auc_score(true_label, pred_score)
    return acc, ap, auc


def eval_one_epoch(hint, than: THAN, sampler: RandHetEdgeSampler, batch_sampler: HetMiniBatchSampler, data):
    logger.info(hint)
    val_acc, val_ap, val_auc = [], [], []
    with torch.no_grad():
        than = than.eval()
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

            pos_prob, neg_prob = than.link_contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, 
                                        src_utype_l, tgt_utype_l, fake_utype_l, etype_l, NUM_NEIGHBORS)

            acc, ap, auc = evaluate_score(size, pos_prob, neg_prob)
            val_acc.append(acc)
            val_ap.append(ap)
            val_auc.append(auc)

    return np.mean(val_acc), np.mean(val_ap), np.mean(val_auc)


# train val test split
g, _ = utils.load_data(DATA, args.n_dim, args.e_dim)
train, val, test, nn_val, nn_test = utils.split_data(g)

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
train_ngh_finder = utils.get_neighbor_finder(train, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
# full graph with all the data for the test and validation purpose
full_ngh_finder = utils.get_neighbor_finder(g, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)

# negative edge sampler
train_rand_sampler = RandHetEdgeSampler(train.src_l, train.dst_l, train.u_type_l, train.v_type_l)
val_rand_sampler = RandHetEdgeSampler((train.src_l, val.src_l), (train.dst_l, val.dst_l), (train.u_type_l, val.u_type_l), (train.v_type_l, val.v_type_l))
test_rand_sampler = RandHetEdgeSampler(g.src_l, g.dst_l, g.u_type_l, g.v_type_l)
nn_val_rand_sampler = RandHetEdgeSampler(nn_val.src_l, nn_val.dst_l, nn_val.u_type_l, nn_val.v_type_l)
nn_test_rand_sampler = RandHetEdgeSampler(nn_test.src_l, nn_test.dst_l, nn_test.u_type_l, nn_test.v_type_l)

# mini-batch idx sampler
train_batch_sampler = HetMiniBatchSampler(g.num_e_type, train.e_type_l, BATCH_SIZE, SHUFFLE, args.padd)
nn_val_batch_sampler = HetMiniBatchSampler(g.num_e_type, nn_val.e_type_l, BATCH_SIZE, hint='val')
nn_test_batch_sampler = HetMiniBatchSampler(g.num_e_type, nn_test.e_type_l, BATCH_SIZE, hint='val')

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
than = THAN(train_ngh_finder, g.n_feat, g.e_feat, g.e_type_feat, g.num_n_type, g.num_e_type, args.t_dim,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, seq_len=SEQ_LEN, 
            n_head=NUM_HEADS, dropout=DROPOUT)
optimizer = torch.optim.Adam(than.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
b_xent = torch.nn.BCEWithLogitsLoss()
than = than.to(device)


num_instance = len(train.src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)


best_epoch = 0
best_auc = 0.
for epoch in range(NUM_EPOCH):
    # training use only training graph
    than.ngh_finder = train_ngh_finder
    acc, ap, auc, m_loss = [], [], [], []
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
        than = than.train()
        pos_prob, neg_prob = than.link_contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, 
                                        src_utype_l, tgt_utype_l, bgd_utype_l, etype_l, NUM_NEIGHBORS)

        loss = criterion(torch.cat((pos_prob, neg_prob), dim=0), lbl)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            than = than.eval()
            _acc, _ap, _auc = evaluate_score(size, pos_prob, neg_prob)
            acc.append(_acc)
            ap.append(_ap)
            auc.append(_auc)
            m_loss.append(loss.item())


    # validation phase use all information
    than.ngh_finder = full_ngh_finder

    # val_acc, val_ap, val_auc = eval_one_epoch('val for nodes', than, val_rand_sampler, val_batch_sampler, val)
    val_acc, val_ap, val_auc = eval_one_epoch('val for new nodes', than, nn_val_rand_sampler, nn_val_batch_sampler, nn_val)

    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train statistics: auc: {:.4f}, ap: {:.4f}, acc: {:.4f}'.format(np.mean(auc), np.mean(ap), np.mean(acc)))
    logger.info('val statistics: auc: {:.4f}, ap: {:.4f}, acc: {:.4f}'.format(val_auc, val_ap, val_acc))
    
    torch.save(than.state_dict(), get_checkpoint_path(epoch))
    if val_auc > best_auc:
        best_epoch = epoch
        best_auc = val_auc


logger.info(f'Loading the best model at epoch {best_epoch} for test')
best_model_path = get_checkpoint_path(best_epoch)
than.load_state_dict(torch.load(best_model_path))
than.eval()

# testing phase use all information
than.ngh_finder = full_ngh_finder
test_acc, test_ap, test_auc = eval_one_epoch('test for nodes', than, nn_test_rand_sampler, nn_test_batch_sampler, test)

logger.info('Test statistics: auc: {:.4f}, ap: {:.4f}, acc: {:.4f}'.format(test_auc, test_ap, test_acc))

logger.info('Saving THAG model')
torch.save(than.state_dict(), MODEL_SAVE_PATH)
logger.info('THAN models saved')


