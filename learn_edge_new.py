"""Unified interface to all dynamic graph model experiments"""
import math
import torch
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from module import THAN
import utils
from utils import RandHetEdgeSampler, HetMiniBatchSampler


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
SHUFFLE = True # args.shuffle


MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

utils.check_dirs()

logger = utils.get_logger(args.prefix+"_new_"+args.data)
logger.info(args)

utils.set_random_seed(2022)



def evaluate_score(size, pos_prob, neg_prob):
    pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    auc = roc_auc_score(true_label, pred_score)
    ap = average_precision_score(true_label, pred_score)
    return auc, ap


def eval_one_epoch(hint, than: THAN, sampler: RandHetEdgeSampler, batch_sampler: HetMiniBatchSampler, data):
    logger.info(hint)
    val_auc, val_ap = [], []
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

            auc, ap = evaluate_score(size, pos_prob, neg_prob)
            val_auc.append(auc)
            val_ap.append(ap)

    return np.mean(val_auc), np.mean(val_ap)


# train test split
if DATA == 'movielens':
    train, test = utils.load_and_split_data_train_test(DATA, args.n_dim, args.e_dim)
    g = train
else:
    g, train, test = utils.load_and_split_data_train_test(DATA, args.n_dim, args.e_dim)
train, nn_test = utils.split_valid_train_nn_test(g, train, test)

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
train_ngh_finder = utils.get_neighbor_finder(train, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
# full graph with all the data for the test and validation purpose
full_ngh_finder = utils.get_neighbor_finder(g, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)

# negative edge sampler
train_rand_sampler = RandHetEdgeSampler(train.src_l, train.dst_l, train.u_type_l, train.v_type_l)
nn_test_rand_sampler = RandHetEdgeSampler(nn_test.src_l, nn_test.dst_l, nn_test.u_type_l, nn_test.v_type_l)

# mini-batch idx sampler
train_batch_sampler = HetMiniBatchSampler(g.num_e_type, train.e_type_l, BATCH_SIZE, SHUFFLE, args.padd)
nn_test_batch_sampler = HetMiniBatchSampler(g.num_e_type, nn_test.e_type_l, BATCH_SIZE, hint='nn_test')

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
than = THAN(train_ngh_finder, g.n_feat, g.e_feat, g.e_type_feat, g.num_n_type, g.num_e_type, args.t_dim,
            num_layers=NUM_LAYER, use_time=USE_TIME, seq_len=SEQ_LEN, n_head=NUM_HEADS, dropout=DROPOUT)
optimizer = torch.optim.Adam(than.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
than = than.to(device)


num_instance = len(train.src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)

best_auc, best_ap = 0., 0.
for epoch in range(NUM_EPOCH):
    # training use only training graph
    than.ngh_finder = train_ngh_finder
    auc, ap, m_loss = [], [], []
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
        loss += 0.001 * than.affinity_score.reg_loss()

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            than = than.eval()
            _auc, _ap = evaluate_score(size, pos_prob, neg_prob)
            auc.append(_auc)
            ap.append(_ap)
            m_loss.append(loss.item())


    # validation phase use all information
    than.ngh_finder = full_ngh_finder

    test_auc, test_ap = eval_one_epoch('test for new nodes', than, nn_test_rand_sampler, nn_test_batch_sampler, nn_test)
    logger.info('Epoch: {}, mean loss: {:.4f}:'.format(epoch, np.mean(m_loss)))
    logger.info('train statistics: auc: {:.4f}, ap: {:.4f}'.format(np.mean(auc), np.mean(ap)))
    logger.info('test statistics: auc: {:.4f}, ap: {:.4f}'.format(test_auc, test_ap))

    if test_auc > best_auc:
        best_auc = test_auc
        best_ap = test_ap


# testing phase use all information
logger.info('Final Best Test: auc: {:.2f}, ap: {:.2f}'.format(best_auc*100, best_ap*100))


