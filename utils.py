import os
import numpy as np
import logging
import torch
import random
import sys
import argparse
import pandas as pd
import json
import math
import logging
from graph import *


### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        
        self.epoch_count += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        if isinstance(src_list, (list, tuple)):
            src_list = np.concatenate(src_list)
        if isinstance(dst_list, (list, tuple)):
            dst_list = np.concatenate(dst_list)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]
    
    def sample_dst(self, size):
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.dst_list[dst_index]


class RandHetEdgeSampler(object):
    def __init__(self, src_list, dst_list, utype_list, vtype_list):
        if isinstance(src_list, (list, tuple)):
            src_list = np.concatenate(src_list)
        if isinstance(dst_list, (list, tuple)):
            dst_list = np.concatenate(dst_list)
        if isinstance(utype_list, (list, tuple)):
            utype_list = np.concatenate(utype_list)
        if isinstance(vtype_list, (list, tuple)):
            vtype_list = np.concatenate(vtype_list)
        # src node
        src_data = {}
        utypes = np.unique(utype_list)
        for utype in utypes:
            idx_mask = utype_list==utype
            src_l = np.unique(src_list[idx_mask])
            src_data[utype] = src_l
        self.src_data = src_data
        # dst node
        dst_data = {}
        vtypes = np.unique(vtype_list)
        for vtype in vtypes:
            idx_mask = vtype_list==vtype
            dst_l = np.unique(dst_list[idx_mask])
            dst_data[vtype] = dst_l
        self.dst_data = dst_data

    def sample(self, size, utype, vtype):
        src_l = self.src_data[utype]
        dst_l = self.dst_data[vtype]
        src_index = np.random.randint(0, len(src_l), size)
        dst_index = np.random.randint(0, len(dst_l), size)
        return src_l[src_index], dst_l[dst_index]
    
    def sample_dst(self, size, vtype):
        dst_l = self.dst_data[vtype]
        dst_index = np.random.randint(0, len(dst_l), size)
        return dst_l[dst_index]
    
    def sample_dst_by_ntype_list(self, vtype_list):
        """ sample equal num dst neg node to vtype_list """
        vtype_cnt = {}
        for i, vtype in enumerate(vtype_list):
            val = vtype_cnt.get(vtype, {'cnt':0, 'idx':[]})
            val['cnt'] += 1
            val['idx'].append(i)
            vtype_cnt[vtype] = val

        dst_index = np.zeros(len(vtype_list), dtype='int64')
        for vtype, val in vtype_cnt.items():
            dst_idx = self.sample_dst(val['cnt'], vtype)
            dst_index[val['idx']] = dst_idx
        return dst_index


class HetMiniBatchSampler(object):
    def __init__(self, num_etype, etype_list, batch_size, shuffle=False, pad_percent=9, hint="train"):
        """ padding the first i/10 events on ts, just use the other (10-i)/10 events for training """
        assert 0<=pad_percent<10, "padd should be in [0, 10) "
        self.padding = 0
        if hint == 'train':
            self.padding = pad_percent * len(etype_list) // 10
            etype_list = etype_list[self.padding:]

        self.num_inst = len(etype_list)
        self.num_batch = math.ceil(self.num_inst / batch_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hint = hint
        logger = logging.getLogger(self.__class__.__name__)
        logger.info('num of {} instances: {}'.format(hint, self.num_inst))
        logger.info('num of batches per epoch: {}'.format(self.num_batch))

        idx_list = np.arange(self.num_inst) + self.padding
        data = {}
        size_e_l = {}
        for etype in range(1, num_etype+1):
            mask = etype_list==etype
            idx = idx_list[mask]
            if self.shuffle:
                np.random.shuffle(idx)
            data[etype] = idx
            size_e_l[etype] = math.floor((len(idx) / self.num_inst) * batch_size)
        self.data = data
        self.size_e_l = size_e_l
        self.cur_batch = 0
        
    def get_batch_index(self):
        if self.cur_batch > self.num_batch:
            return None

        res_idx = []
        for etype, idx in self.data.items():
            size_e = self.size_e_l[etype]
            s_idx = self.cur_batch * size_e
            e_idx = min(len(idx) - 1, s_idx + size_e)
            res_idx.append(idx[s_idx:e_idx])
        
        self.cur_batch += 1
        print(f"{self.hint} batch {self.cur_batch}/{self.num_batch+1}\t\r", end='')
        return np.concatenate(res_idx)

    def reset(self):
        self.cur_batch = 0
        if self.shuffle:
            for etype, idx in self.data.items():
                np.random.shuffle(idx)
                self.data[etype] = idx


def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for THAT experiments on link predictions')

    parser.add_argument('-d', '--data', type=str, default='movielens', help='data sources to use, try twitter, mathoverflow, movielens')
    parser.add_argument('--bs', type=int, default=500, help='batch_size')
    parser.add_argument('--prefix', type=str, default='THAN', help='prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=8, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=4, help='number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--n_dim', type=int, default=32, help='Dimentions of the default node embedding')
    parser.add_argument('--e_dim', type=int, default=16, help='Dimentions of the default edge embedding')
    parser.add_argument('--t_dim', type=int, default=32, help='Dimentions of the time embedding')
    parser.add_argument('--e_type_dim', type=int, default=32, help='Dimentions of the edge type embedding')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--shuffle', action='store_true', help='shuffle index for bacth')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    return args


def check_dirs():
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    if not os.path.exists("saved_checkpoints"):
        os.mkdir("saved_checkpoints")


def get_logger(name="THAN"):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}.log'.format(name))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("\n")
    return logger


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


""" load data """

def _load_base(dataset: str, n_dim=None, e_dim=None):
    with open(f'./processed/{dataset}/desc.json', 'r') as f:
        desc = json.load(f)
    
    g_df = pd.read_csv(f'./processed/{dataset}/events.csv')

    # node_feat
    if os.path.exists(f'./processed/{dataset}/node_ft.npy'):
        n_feat = np.load(f'./processed/{dataset}/node_ft.npy')
    elif dataset == 'wsdm_b':
        n_feat = None
    else:
        n_feat = np.random.randn(desc['num_node'] + 1, n_dim) * 0.05
        n_feat[0] = 0.

    # edge_feat
    if os.path.exists(f'./processed/{dataset}/edge_ft.npy'):
        e_feat = np.load(f'./processed/{dataset}/edge_ft.npy')[:,:32]
    elif os.path.exists(f'./processed/{dataset}/edge_ft.csv'):
        e_feat = pd.read_csv(f'./processed/{dataset}/edge_ft.csv', header=None, index_col=[0])
    elif dataset.startswith("wsdm"):
        e_feat = None
    else:
        e_feat = np.zeros((desc['num_edge'] + 1, e_dim))

    # edge_type_feat
    if os.path.exists(f'./processed/{dataset}/etype_ft.npy'):
        etype_ft = np.load(f'./processed/{dataset}/etype_ft.npy')
    else:
        etype_ft = None
    return g_df, n_feat, e_feat, etype_ft, desc


def load_data(dataset:str, n_dim=None, e_dim=None):
    if dataset == 'movielens':
        return load_data_with_test_events(dataset, n_dim, e_dim)

    g_df, n_feat, e_feat, etype_ft, desc = _load_base(dataset, n_dim, e_dim)

    return TemHetGraphData(g_df, n_feat, e_feat, desc['num_node_type'], desc['num_edge_type'], etype_ft)


def load_data_with_test_events(dataset, n_dim, e_dim):
    g_train, n_feat, e_feat, etype_ft, desc = _load_base(dataset, n_dim, e_dim)

    g_test = pd.read_csv(f'./processed/{dataset}/events_test.csv')

    if dataset == 'movielens':
        return TemHetGraphData(g_train, n_feat, e_feat, desc['num_node_type'], desc['num_edge_type'], etype_ft), \
            Events(g_test.u.values, g_test.v.values, g_test.ts.values, g_test.e_idx.values, g_test.e_type.values, g_test.u_type.values, g_test.v_type.values)
    else:
        return TemHetGraphData(g_train, n_feat, e_feat, desc['num_node_type'], desc['num_edge_type'], etype_ft), \
            Events(g_test.u.values, g_test.v.values, g_test.ts.values, g_test.e_idx.values, g_test.e_type.values, g_test.u_type.values, g_test.v_type.values, g_test.label.values)


""" split data """

def split_data_train_test(g: TemHetGraphData, test_ratio=0.2):
    test_time = np.quantile(g.ts_l, 1. - test_ratio)

    ''' train '''
    valid_train_flag = g.ts_l <= test_time
    train = g.sample_by_mask(valid_train_flag)

    ''' test '''
    valid_test_flag = g.ts_l > test_time  # total test edges
    test = g.sample_by_mask(valid_test_flag)

    return train, test

# mask 10% node
def split_valid_train_nn_test(g: TemHetGraphData, train: TemHetGraphData, test: TemHetGraphData, mask_ratio=0.1):
    # mask_ratio: Make mask_ratio (eg. 10%) of the nodes not appear in the training set to achieve inductive
    nodes_in_test = set(test.src_l).union(set(test.dst_l))
    mask_node_set = set(random.sample(nodes_in_test, int(mask_ratio * len(g.node_set))))

    ''' valid train '''
    valid_src_flag_train = pd.Series(train.src_l).map(lambda x: x not in mask_node_set).values
    valid_dst_flag_train = pd.Series(train.dst_l).map(lambda x: x not in mask_node_set).values
    valid_train_flag = valid_src_flag_train * valid_dst_flag_train
    valid_train = train.sample_by_mask(valid_train_flag)

    ''' new node edges in test '''
    nn_test_flag = np.array([(a in mask_node_set or b in mask_node_set) for a, b in zip(test.src_l, test.dst_l)])
    nn_test = test.sample_by_mask(nn_test_flag)

    return valid_train, nn_test


def load_and_split_data_train_test(dataset:str, n_dim=None, e_dim=None, ratio=0.2):
    if dataset == 'movielens':
        train, test = load_data_with_test_events(dataset, n_dim, e_dim)
        return train, test
    
    g = load_data(dataset, n_dim, e_dim)
    train, test = split_data_train_test(g, ratio)
    return g, train, test


""" neighbor finder """

def get_neighbor_finder(data, max_idx, uniform=True, shuffle=False, num_edge_type=None):
    dst_l = data.dst_l
    e_idx_l = data.e_idx_l
    # if set shuffle true, the dst_l will be shuffled
    if shuffle:
        idx = np.random.permutation(len(dst_l))
        dst_l = dst_l[idx] 
        # e_idx_l = e_idx_l[idx] 

    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts, etype, utype, vtype in zip(data.src_l, dst_l, e_idx_l, data.ts_l, data.e_type_l, data.u_type_l, data.v_type_l):
        adj_list[src].append((dst, eidx, ts, etype, utype, vtype))
        adj_list[dst].append((src, eidx, ts, etype, utype, vtype))
    return NeighborFinder(adj_list, uniform=uniform, num_edge_type=num_edge_type)

