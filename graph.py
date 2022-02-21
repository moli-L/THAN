import numpy as np
import random


class Events(object):
    def __init__(self, src_l, dst_l, ts_l, e_idx_l, e_type_l=None, u_type_l=None, v_type_l=None, label_l=None):
        self.src_l = src_l
        self.dst_l = dst_l
        self.ts_l = ts_l
        self.e_idx_l = e_idx_l
        self.e_type_l = e_type_l
        self.u_type_l = u_type_l
        self.v_type_l = v_type_l
        self.label_l = label_l
        self.node_set = set(np.unique(np.hstack([self.src_l, self.dst_l])))
        self.num_nodes = len(self.node_set)

    def sample_by_mask(self, mask):
        sam_src_l = self.src_l[mask]
        sam_dst_l = self.dst_l[mask]
        sam_ts_l = self.ts_l[mask]
        sam_e_idx_l = self.e_idx_l[mask]
        sam_e_type_l = self.e_type_l[mask] if self.e_type_l is not None else None
        sam_u_type_l = self.u_type_l[mask] if self.u_type_l is not None else None
        sam_v_type_l = self.v_type_l[mask] if self.v_type_l is not None else None
        sam_label_l = self.label_l[mask] if self.label_l is not None else None

        return Events(sam_src_l, sam_dst_l, sam_ts_l, sam_e_idx_l, sam_e_type_l, sam_u_type_l, sam_v_type_l, sam_label_l)


class TemHetGraphData(Events):
    def __init__(self, g_df, n_feat, e_feat, num_n_type, num_e_type, e_type_feat=None):
        super(TemHetGraphData, self).__init__(g_df.u.values, g_df.v.values, g_df.ts.values, g_df.e_idx.values, g_df.e_type.values, g_df.u_type.values, g_df.v_type.values)
        self.g_df = g_df
        self.n_feat = n_feat
        self.e_feat = e_feat
        self.num_n_type = num_n_type
        self.num_e_type = num_e_type
        self.e_type_feat = e_type_feat
        self.max_idx = max(self.src_l.max(), self.dst_l.max())
        self.n_dim = n_feat.shape[1]
        self.e_dim = e_feat.shape[1]


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, num_edge_type=None):
        """
        Params
        ------
          node_idx_l: List[int]
          node_ts_l: List[int]
          off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """
        node_idx_l, node_ts_l, e_idx_l, e_type_l, u_type_l, v_type_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = e_idx_l
        self.edge_type_l = e_type_l
        self.u_type_l = u_type_l
        self.v_type_l = v_type_l
        self.off_set_l = off_set_l
        
        if num_edge_type is None:
            num_edge_type = len(np.unique(e_type_l))
        self.num_edge_type = num_edge_type + 1  # padding 0 type
        self.uniform = uniform
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
          adj_list: List[List[int]]
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        e_type_l = []
        u_type_l = []
        v_type_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]  #节点i的邻居
            curr = sorted(curr, key=lambda x: x[2])  #根据所属ts排序,以便快速查找之前的邻居
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
            e_type_l.extend([x[3] for x in curr])
            u_type_l.extend([x[4] for x in curr])
            v_type_l.extend([x[5] for x in curr])

            off_set_l.append(len(n_idx_l))  #节点i邻居在n_idx的终止下标
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        e_type_l = np.array(e_type_l)
        u_type_l = np.array(u_type_l)
        v_type_l = np.array(v_type_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, e_type_l, u_type_l, v_type_l, off_set_l
        
    def find_before(self, src_idx, cut_time):
        """
        Params
        ------
          src_idx: int
          cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l   #需要按时间升序排列
        edge_idx_l = self.edge_idx_l
        edge_type_l = self.edge_type_l
        # u_type_l = self.u_type_l
        v_type_l = self.v_type_l
        off_set_l = self.off_set_l
        
        ngh_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_e_type = edge_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        # ngh_u_type = u_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_v_type = v_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        
        if len(ngh_idx) == 0 or len(ngh_ts) == 0:
            return ngh_idx, ngh_e_idx, ngh_ts, ngh_e_type, ngh_v_type

        left = 0
        right = len(ngh_idx) - 1
        
        while left + 1 < right:   # binary search
            mid = (left + right) // 2
            curr_t = ngh_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if ngh_ts[left] >= cut_time:
            return ngh_idx[:left], ngh_e_idx[:left], ngh_ts[:left], ngh_e_type[:left], ngh_v_type[:left]
        elif ngh_ts[right] < cut_time:
            return ngh_idx[:right+1], ngh_e_idx[:right+1], ngh_ts[:right+1], ngh_e_type[:right+1], ngh_v_type[:right+1]
        else:
            return ngh_idx[:right], ngh_e_idx[:right], ngh_ts[:right], ngh_e_type[:right], ngh_v_type[:right]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
          src_idx_l: List[int]
          cut_time_l: List[float],
          num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_etype_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        # out_ngh_utype_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_vtype_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts, ngh_etype, ngh_vtype = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    # samd_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    real_num_neighbors = min(num_neighbors, len(ngh_idx))
                    nidx = np.arange(len(ngh_idx)).tolist()
                    samd_idx = random.sample(nidx, real_num_neighbors)

                    out_ngh_node_batch[i, :real_num_neighbors] = ngh_idx[samd_idx]
                    out_ngh_t_batch[i, :real_num_neighbors] = ngh_ts[samd_idx]
                    out_ngh_eidx_batch[i, :real_num_neighbors] = ngh_eidx[samd_idx]
                    out_ngh_etype_batch[i, :real_num_neighbors] = ngh_etype[samd_idx]
                    # out_ngh_utype_batch[i, :real_num_neighbors] = ngh_utype[samd_idx]
                    out_ngh_vtype_batch[i, :real_num_neighbors] = ngh_vtype[samd_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                    out_ngh_etype_batch[i, :] = out_ngh_etype_batch[i, :][pos]
                    # out_ngh_utype_batch[i, :] = out_ngh_utype_batch[i, :][pos]
                    out_ngh_vtype_batch[i, :] = out_ngh_vtype_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[-num_neighbors:]
                    ngh_idx = ngh_idx[-num_neighbors:]
                    ngh_eidx = ngh_eidx[-num_neighbors:]
                    ngh_etype = ngh_etype[-num_neighbors:]
                    # ngh_utype = ngh_utype[-num_neighbors:]
                    ngh_vtype = ngh_vtype[-num_neighbors:]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    assert(len(ngh_etype) <= num_neighbors)
                    # assert(len(ngh_utype) <= num_neighbors)
                    assert(len(ngh_vtype) <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx
                    out_ngh_etype_batch[i,  num_neighbors - len(ngh_etype):] = ngh_etype
                    # out_ngh_utype_batch[i,  num_neighbors - len(ngh_etype):] = ngh_utype
                    out_ngh_vtype_batch[i,  num_neighbors - len(ngh_etype):] = ngh_vtype
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_etype_batch, out_ngh_vtype_batch

    def get_temporal_hetneighbor(self, src_idx_l, cut_time_l, num_neighbors=10):
        """
        Params
        ------
          src_idx_l: List[int]
          cut_time_l: List[float],
          num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        total_num_nghs = num_neighbors * self.num_edge_type

        out_ngh_node_batch = np.zeros((len(src_idx_l), total_num_nghs)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), total_num_nghs)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), total_num_nghs)).astype(np.int32)
        out_ngh_etype_batch = np.zeros((len(src_idx_l), total_num_nghs)).astype(np.int32)
        out_ngh_vtype_batch = np.zeros((len(src_idx_l), total_num_nghs)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts, ngh_etype, ngh_vtype = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                etype_mask = {}
                for etype in np.unique(ngh_etype):
                    etype_mask[etype] = ngh_etype==etype

                ix_l = []
                ix = np.arange(len(ngh_idx))
                for etype, mask in etype_mask.items():
                    if self.uniform:
                        # 同一分布选择的邻居，每一类最多选num个
                        tmp_idx = ix[mask]
                        real_num_neighbors = min(num_neighbors, len(tmp_idx))
                        sam_idx = random.sample(tmp_idx.tolist(), real_num_neighbors)
                        ix_l.append(sam_idx)
                    else:
                        ix_l.append(ix[mask][-num_neighbors:])  # 选择时间最近的邻居，每一类最多选num个
                
                nidx = np.sort(np.concatenate(ix_l))
                real_num_nghs = len(nidx)
                
                out_ngh_node_batch[i, total_num_nghs-real_num_nghs:] = ngh_idx[nidx]
                out_ngh_t_batch[i, total_num_nghs-real_num_nghs:] = ngh_ts[nidx]
                out_ngh_eidx_batch[i, total_num_nghs-real_num_nghs:] = ngh_eidx[nidx]
                out_ngh_etype_batch[i, total_num_nghs-real_num_nghs:] = ngh_etype[nidx]
                out_ngh_vtype_batch[i, total_num_nghs-real_num_nghs:] = ngh_vtype[nidx]
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch, out_ngh_etype_batch, out_ngh_vtype_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """ Sampling the k-hop temporal sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k-1):
            ngn_node_set, ngh_t_set = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_set.shape
            ngn_node_set = ngn_node_set.flatten()
            ngn_t_set = ngh_t_set.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_set, ngn_t_set, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors) # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records

    def find_k_hop_subgraph(self, k, src_idx_l, src_time_l, num_neighbors=20):
        """ Sampling the k-hop temporal sub graph
        """
        batch_size = len(src_idx_l)
        edge = []
        eidx = []
        t_delta = []
        for _ in range(k):
            ngh_node_batch, ngh_eidx_batch, ngh_t_batch = self.get_temporal_neighbor(src_idx_l, src_time_l, num_neighbors)
            ngh_node_set = ngh_node_batch.flatten()
            e = np.stack([src_idx_l.repeat(num_neighbors), ngh_node_set])

            edge.append(e)
            eidx.append(ngh_eidx_batch.flatten())
            ngh_t_set = ngh_t_batch.flatten()
            ngh_t_delta = np.abs(src_time_l.repeat(num_neighbors) - ngh_t_set)
            t_delta.append(ngh_t_delta)

            src_idx_l = ngh_node_set
            src_time_l = ngh_t_set
        
        edge = np.concatenate(edge, axis=1)
        eidx = np.concatenate(eidx)
        t_delta = np.concatenate(t_delta)
        
        edge = edge.transpose().reshape(batch_size,-1,2).transpose(0,2,1)
        eidx = eidx.reshape(batch_size,-1)
        t_delta = t_delta.reshape(batch_size,-1)

        edge_batch = []
        eidx_batch = []
        t_batch = []
        node_set_batch = []
        for i in range(batch_size):
            e = edge[i]
            # mask = (e==e)[0]
            mask = e!=0
            mask = mask[0] * mask[1]
            e = e[:, mask]

            # get node set and reindex edge_index
            node_set, e = np.unique(e, return_inverse=True)
            e = e.reshape(2, -1)
            edge_batch.append(e)
            node_set_batch.append(node_set)
            eidx_batch.append(eidx[i][mask])
            t_batch.append(t_delta[i][mask])

        return edge_batch, eidx_batch, t_batch, node_set_batch


