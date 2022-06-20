import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn.functional as F


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def preprocess(data_name, spitter=",", has_header=True):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(spitter)
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(d):  # one node type
    LEN_EV = len(d['u'])
    nodes, new_index = np.unique(d['u'].append(
        d['v']).values, return_inverse=True)
    new_u, new_v = new_index.reshape(2, -1)
    assert LEN_EV == len(new_u)

    NUM_NODES = len(nodes)
    d['u'] = new_u
    d['v'] = new_v
    print("num nodes:", NUM_NODES)
    # add padding node
    d['u'] += 1
    d['v'] += 1

    return d, NUM_NODES


def reindex2(d):  # two or more node types
    u = d['u'].sort_values().unique()
    idx_u = np.arange(len(u))
    u_dict = {u[k]: k for k in idx_u}
    d['u'] = d['u'].map(lambda x: u_dict[x])

    i = d['i'].sort_values().unique()
    idx_i = np.arange(len(i))
    i_dict = {i[k]: k for k in idx_i}
    d['i'] = d['i'].map(lambda x: i_dict[x])

    label = d['label'].sort_values().unique()
    label_ix = np.arange(len(label))
    label_dict = {label[k]: k for k in label_ix}
    d['label'] = d['label'].map(lambda x: label_dict[x])

    print(f'u:{len(u)}, i:{len(i)}, label:{len(label)}')
    return d


"""
process dataset
"""


def process_twitter():
    SRC_PATH = './data/twitter/higgs-activity_time.txt'
    OUT_DIR = './processed/twitter'
    OUT_EV = f'{OUT_DIR}/events.csv'
    OUT_NODE_FEAT = f'{OUT_DIR}/node_ft.npy'
    OUT_EDGE_FEAT = f'{OUT_DIR}/edge_ft.npy'
    OUT_DESC = f'{OUT_DIR}/desc.json'

    check_dir(OUT_DIR)

    df = pd.read_csv(SRC_PATH, header=None, names=[
                     'u', 'v', 'ts', 'e_type'], delimiter=' ')
    # sort by ts
    df = df.sort_values('ts')
    df = df.reset_index(drop=True)
    NUM_EV = df.shape[0]
    print("num events:", NUM_EV)

    # reindex nodes (twitter only has one type nodes)
    new_df, NUM_NODE = reindex(df)
    NUM_N_TYPE = 1
    print("num node types:", NUM_N_TYPE)

    # ts
    min_ts = new_df.ts.min()
    new_df.ts = new_df.ts - min_ts

    # node type
    new_df['u_type'] = 1
    new_df['v_type'] = 1

    # edge types
    e_types, new_index = np.unique(
        new_df['e_type'].values, return_inverse=True)
    new_df['e_type'] = pd.Series(new_index)
    new_df['e_type'] += 1    # encoding from 1
    NUM_E_TYPE = len(e_types)
    print("num edge types:", NUM_E_TYPE)

    # edge idx
    new_df['e_idx'] = pd.Series(np.arange(1, NUM_EV + 1))

    # feats, padding a default node & edge feat (no data)

    # save
    new_df.to_csv(OUT_EV, index=None)
    desc = {
        "num_node": NUM_NODE,
        "num_edge": NUM_EV,
        "num_node_type": NUM_N_TYPE,
        "num_edge_type": NUM_E_TYPE
    }
    with open(OUT_DESC, 'w') as f:
        json.dump(desc, f, indent=4)


def process_mathoverflow():
    SRC_a2q_PATH = './data/mathoverflow/sx-mathoverflow-a2q.txt'
    SRC_c2a_PATH = './data/mathoverflow/sx-mathoverflow-c2a.txt'
    SRC_c2q_PATH = './data/mathoverflow/sx-mathoverflow-c2q.txt'

    OUT_DIR = './processed/mathoverflow'
    OUT_EV = f'{OUT_DIR}/events.csv'
    OUT_NODE_FEAT = f'{OUT_DIR}/node_ft.npy'
    OUT_EDGE_FEAT = f'{OUT_DIR}/edge_ft.npy'
    OUT_DESC = f'{OUT_DIR}/desc.json'

    check_dir(OUT_DIR)

    df1 = pd.read_csv(SRC_a2q_PATH, header=None, names=[
                      'u', 'v', 'ts'], delimiter=' ')
    df2 = pd.read_csv(SRC_c2a_PATH, header=None, names=[
                      'u', 'v', 'ts'], delimiter=' ')
    df3 = pd.read_csv(SRC_c2q_PATH, header=None, names=[
                      'u', 'v', 'ts'], delimiter=' ')
    df1['e_type'] = 1
    df2['e_type'] = 2
    df3['e_type'] = 3

    NUM_N_TYPE = 1
    print("num node types:", NUM_N_TYPE)
    NUM_E_TYPE = 3
    print("num edge types:", NUM_E_TYPE)

    # merge events
    df = pd.concat([df1, df2, df3], ignore_index=True)
    NUM_EV = df.shape[0]
    print("num events:", NUM_EV)

    # sort by ts
    df = df.sort_values('ts')
    df = df.reset_index(drop=True)

    # reindex nodes, encoding from 1
    # a
    node_set, new_idx_n = np.unique(
        df.u.append(df.v).values, return_inverse=True)
    new_idx_n = new_idx_n + 1
    sp = len(df.u)
    df.u = new_idx_n[:sp]
    df.v = new_idx_n[sp:]
    NUM_NODE = len(node_set)
    print("num node:", NUM_NODE)

    df['u_type'] = 1
    df['v_type'] = 1

    # edge idx
    df['e_idx'] = pd.Series(np.arange(1, NUM_EV + 1))

    # feats, padding a default node & edge feat (no data)

    # save
    df.to_csv(OUT_EV, index=None)
    desc = {
        "num_node": NUM_NODE,
        "num_edge": NUM_EV,
        "num_node_type": NUM_N_TYPE,
        "num_edge_type": NUM_E_TYPE
    }
    with open(OUT_DESC, 'w') as f:
        json.dump(desc, f, indent=4)


def process_movielens():
    SRC_TRAIN_PATH = './data/movielens/u1.base'
    SRC_TEST_PATH = './data/movielens/u1.test'
    SRC_U_PATH = './data/movielens/u.user'
    SRC_V_PATH = './data/movielens/u.item'

    OUT_DIR = './processed/movielens'
    OUT_EV = f'{OUT_DIR}/events.csv'
    OUT_EV_TEST = f"{OUT_DIR}/events_test.csv"
    OUT_NODE_FEAT = f'{OUT_DIR}/node_ft.npy'
    OUT_EDGE_FEAT = f'{OUT_DIR}/edge_ft.npy'
    OUT_DESC = f'{OUT_DIR}/desc.json'

    check_dir(OUT_DIR)

    df_train = pd.read_csv(SRC_TRAIN_PATH, header=None, names=[
                           'u', 'v', 'e_type', 'ts'], delimiter='\t')
    df_test = pd.read_csv(SRC_TEST_PATH, header=None, names=[
                          'u', 'v', 'e_type', 'ts'], delimiter='\t')
    df_u = pd.read_csv(SRC_U_PATH, header=None, names=[
                       'id', 'age', 'gender', 'occupation'], usecols=[0, 1, 2, 3], delimiter='|')
    df_v = pd.read_csv(SRC_V_PATH, header=None,
                       delimiter='|', encoding='ISO-8859-1')

    # node feat
    u_ft = pd.get_dummies(df_u[['gender', 'occupation']]).iloc[:, 1:].values
    u_age = F.normalize(torch.from_numpy(
        df_u['age'].values).float(), dim=0).numpy()
    u_ft = np.c_[u_age, u_ft]
    v_ft = df_v.iloc[:, 5:].values
    max_dim = u_ft.shape[1]  # shape of u > v
    max_dim = max_dim + 4 - (max_dim % 4)  # 补至4的整数倍
    empty1 = np.zeros((v_ft.shape[0], max_dim-v_ft.shape[1]))
    empty2 = np.zeros((u_ft.shape[0], max_dim-u_ft.shape[1]))
    v_ft = np.hstack([v_ft, empty1])
    u_ft = np.hstack([u_ft, empty2])
    n_feat = np.vstack([np.zeros(max_dim), u_ft, v_ft])  # padding a default

    # u=1, v=2
    df_train['u_type'] = 1
    df_test['u_type'] = 1
    df_train['v_type'] = 2
    df_test['v_type'] = 2

    NUM_N_TYPE = 2
    print("num node types:", NUM_N_TYPE)
    NUM_E_TYPE = 5   # five rating
    print("num edge types:", NUM_E_TYPE)

    # reindex nodes, encoding from 1
    # v
    NUM_N_U = 943
    NUM_N_V = 1682
    df_train.v += NUM_N_U
    df_test.v += NUM_N_U

    NUM_NODE = NUM_N_U + NUM_N_V
    print("num node:", NUM_NODE)

    # merge events
    NUM_EV = df_train.shape[0]
    print("num events:", NUM_EV)
    NUM_EV_TEST = df_test.shape[0]
    print("num events test:", NUM_EV_TEST)

    # sort by ts
    df = df_train.sort_values('ts')
    df = df.reset_index(drop=True)

    # edge idx
    df['e_idx'] = pd.Series(np.arange(1, NUM_EV + 1))
    df_test['e_idx'] = pd.Series(np.arange(NUM_EV_TEST))

    # save
    df.to_csv(OUT_EV, index=None)
    df_test.to_csv(OUT_EV_TEST, index=None)
    np.save(OUT_NODE_FEAT, n_feat)
    desc = {
        "num_node": NUM_NODE,
        "num_edge": NUM_EV,
        "num_edge_test": NUM_EV_TEST,
        "num_node_type": NUM_N_TYPE,
        "num_edge_type": NUM_E_TYPE,
        "num_node_u": NUM_N_U,
        "num_node_v": NUM_N_V,
    }
    with open(OUT_DESC, 'w') as f:
        json.dump(desc, f, indent=4)



def process(name):
    eval(f'process_{name}')()


process('twitter')
process('mathoverflow')
process('movielens')

