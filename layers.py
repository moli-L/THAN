import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class HetMatchDecoder(torch.nn.Module):
    def __init__(self, num_etypes, dim, etype_feat=None):
        super().__init__()
        if etype_feat is None:
            etype_feat = torch.Tensor(num_etypes, dim)
            self.fc1 = torch.nn.Linear(dim*3, dim)
        else:
            etype_feat = torch.from_numpy(etype_feat)
            self.fc1 = torch.nn.Linear(dim*2 + etype_feat.shape[1], dim)
        self.rel_emb = nn.Parameter(etype_feat)

        self.fc2 = torch.nn.Linear(dim, 1)
        self.act = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb.data)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, y, edge_type_l):
        etype_l = torch.from_numpy(edge_type_l).to(x.device)
        etype_l -= 1
        e = self.rel_emb[etype_l]
        h = torch.cat([x, e, y], dim=1)
        h = self.act(self.fc1(h))
        return self.fc2(h).squeeze(dim=-1)
    
    def reg_loss(self):
        return self.rel_emb.pow(2).mean()


class MergeFFN(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x1, x2):
        h = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(h))
        return self.fc2(h)
      

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, r_pri=None, mask=None):
        # q, k:  (n*b) x N x dk
        # v:     (n*b) x N x dv
        # r_pri: b x N

        attn = torch.sum(q * k, dim=-1)  # [n*b, N]
        if r_pri is None:
            attn = attn / self.temperature
        else:
            attn = attn.reshape(-1, *r_pri.shape) * r_pri / self.temperature
            attn = attn.reshape(-1, r_pri.size(1))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n*b, N]
        attn = self.dropout(attn)

        output = (attn.unsqueeze(-1) * v).sum(dim=1)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, num_n_type=1, num_e_type=1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.num_n_type = num_n_type
        self.num_e_type = num_e_type
        
        self.d_r = self.d_model
        self.w_qs = nn.ModuleList()
        self.w_ks = nn.ModuleList()
        self.w_vs = nn.ModuleList()
        for _ in range(num_e_type+1):
            self.w_qs.append(nn.Linear(self.d_r, n_head * d_k, bias=False))
            self.w_ks.append(nn.Linear(self.d_r, n_head * d_k, bias=False))
            self.w_vs.append(nn.Linear(self.d_r, n_head * d_v, bias=False))

        self.relation_pri = nn.Parameter(torch.ones(num_n_type + 1, num_e_type + 1))

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout=dropout)
        
        self.fc_lins = nn.ModuleList()
        for _ in range(num_n_type+1):
            self.fc_lins.append(nn.Linear(self.n_head*self.d_v, self.d_model))
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameter()

    def reset_parameter(self):
        for i in range(len(self.w_qs)):
            nn.init.xavier_uniform_(self.w_qs[i].weight.data)
            nn.init.xavier_uniform_(self.w_ks[i].weight.data)
            nn.init.xavier_uniform_(self.w_vs[i].weight.data)
        
        for fc in self.fc_lins:
            nn.init.xavier_uniform_(fc.weight.data)

    def _get_relation_pri_of_q(self, seq_utype, seq_etype):
        N = seq_etype.size(1)
        seq = seq_utype.repeat(N, 1).t()
        idx = seq * (self.num_e_type+1) + seq_etype
        pri = self.relation_pri.flatten()
        pri = pri.index_select(0, idx.flatten())
        return pri.view(-1, N)

    def _compute_QKV_by_etype(self, q, k, v, seq_etype):
        for i, etype in enumerate(seq_etype.unique()):
            msk = (seq_etype==etype).unsqueeze(-1)
            if i==0:
                Q = self.w_qs[etype](q * msk)
                K = self.w_ks[etype](k * msk)
                V = self.w_vs[etype](v * msk)
            else:
                Q += self.w_qs[etype](q * msk)
                K += self.w_ks[etype](k * msk)
                V += self.w_vs[etype](v * msk)
        return Q, K, V

    def forward(self, q, k, v, seq_etype=None, seq_utype=None, seq_vtype=None, mask=None):
        """
        params:
            q:    [B, N, D]  of src
            k, v: [B, N, D]  of ngh
            seq_etype: [B, N]
            seq_utype: [B]
            seq_vtype: [B, N]
            mask: [B, N]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_e = q.size(0), k.size(1)

        # compute Q, K, V in relature feature space
        q, k, v = self._compute_QKV_by_etype(q, k, v, seq_etype)  # (b, N, n*D)

        q = q.view(sz_b, len_e, n_head, d_k)  # (b, N, n_head, D)  源节点到各个边的转移向量
        k = k.view(sz_b, len_e, n_head, d_k)
        v = v.view(sz_b, len_e, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_k) # (n*b) x N x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_k) # (n*b) x N x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_v) # (n*b) x N x dv

        # pri = None
        pri = self._get_relation_pri_of_q(seq_utype, seq_etype)  # [B, N]
        mask = mask.repeat(n_head, 1) # (n*b, N) x .. x ..
        output, attn = self.attention(q, k, v, pri, mask=mask)  # output (n*b) x dv

        output = output.view(n_head, sz_b, d_v)  # n x b x dv
        output = output.transpose(0, 1).contiguous().view(sz_b, -1) # b x (n*dv) = b x d_model

        """ map out to q's feature space """
        out = torch.zeros_like(output)
        for utype in seq_utype.unique():
            mask = seq_utype==utype
            out[mask] = self.fc_lins[utype](output[mask])
        output = out
        
        # [b, d_model]
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output, attn


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic  # [N, L, time_dim]


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


def resize(e, dim):
    # e:[B, N, D]
    if e.size(-1) >= dim:
        res = e[:, :, :dim]
    else:
        e_padd = torch.zeros(e.size(0), e.size(1), dim-e.size(2)).to(e.device)
        res = torch.cat([e, e_padd], dim=-1)
    return res


class Transfer(torch.nn.Module):
    """ transfer head and tail node to edge feature space """
    
    def __init__(self, num_n_type, num_e_type, n_dim, e_dim=32, e_type_feat=None):
        super().__init__()
        # node type and edge type vector, padding an idx
        self.node_trans = nn.Embedding(num_n_type+1, n_dim, padding_idx=0)
        if e_type_feat is not None:
            self.edge_trans = nn.Embedding.from_pretrained(e_type_feat, padding_idx=0)
        else:
            self.edge_trans = nn.Embedding(num_e_type+1, e_dim, padding_idx=0)
        self.e_dim = self.edge_trans.weight.size(1)

    def forward(self, src, nghs, seq_utype, seq_vtype, seq_etype):  
        """
        params:
            src:  [B, D]  of src
            nghs: [B, N, D]  of ngh
            seq_utype: [B]
            seq_vtype: [B, N]
            seq_etype: [B, N]
        """
        q = src.unsqueeze(1) # [B, 1, D]
        k = nghs
        u_trans = self.node_trans(seq_utype).unsqueeze(1)  # [B, 1, D]
        v_trans = self.node_trans(seq_vtype)  # [B, N, D]
        r_trans = self.edge_trans(seq_etype)  # [B, N, De]

        # [B, N, De]
        q = torch.sum(q * u_trans, dim=-1, keepdim=True) * r_trans + resize(q, self.e_dim)
        k = torch.sum(k * v_trans, dim=-1, keepdim=True) * r_trans + resize(k, self.e_dim)
        return q, k


class AttnModel(nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, transfer, n_head=2, dropout=0.1,
                 num_n_type=1, num_e_type=1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          dropout: probability of dropping a neural
          num_n_type: number of node types
          num_e_type: number of edge types
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        
        self.model_dim = (feat_dim + time_dim)
        assert(self.model_dim % n_head == 0)

        self.transfer = transfer
        
        self.norm1, self.norm2 = nn.LayerNorm(feat_dim), nn.LayerNorm(feat_dim)
        
        self.multi_head_target = MultiHeadAttention(n_head, 
                                            d_model=self.model_dim, 
                                            d_k=self.model_dim // n_head, 
                                            d_v=self.model_dim // n_head, 
                                            dropout=dropout,
                                            num_n_type=num_n_type,
                                            num_e_type=num_e_type)
        
        
        self.merger = MergeFFN(self.model_dim, feat_dim, feat_dim, feat_dim)
        
        
    def forward(self, src, src_t, seq, seq_t, seq_e, seq_etype, seq_utype, seq_vtype, mask):
        """"Attention based temporal attention forward pass
        args:
          src:   float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, 1, Dt], Dt == D
          seq:   float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          seq_etype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          seq_utype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          seq_vtype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          mask:  boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.
        returns:
          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """
        # type mapping
        q, k = self.transfer(src, seq, seq_utype, seq_vtype, seq_etype) # [B, N, D]
        # q, k = src.unsqueeze(1).repeat(1, seq.size(1), 1), seq  # no transfer

        # add & norm
        q = self.norm1(q)  # source node is not associated with edge feature
        k = self.norm2(k + resize(seq_e, self.feat_dim))

        # [B, D+Dt]
        q = torch.cat([q, src_t.repeat(1, q.size(1), 1)], dim=2)
        k = torch.cat([k, seq_t], dim=2)

        # target-attention
        # output: [B, D], attn: [B, N, N]
        output, attn = self.multi_head_target(q=q, k=k, v=k, seq_etype=seq_etype, seq_utype=seq_utype,
                                                 seq_vtype=seq_vtype, mask=mask)
        
        # residual
        # why not use q for residual: src_e is zeros vector and src_t is one vector, so they will not contribute useful information for the source node v0
        output = self.merger(output, src)
        
        return output, attn




class GraphMeanEmbedding(nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, transfer, n_head=2, dropout=0.1,
                 num_n_type=1, num_e_type=1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          dropout: probability of dropping a neural
          num_n_type: number of node types
          num_e_type: number of edge types
        """
        super(GraphMeanEmbedding, self).__init__()
        
        self.linear_1 = torch.nn.Linear(feat_dim + time_dim + edge_dim, feat_dim)
        self.linear_2 = torch.nn.Linear(feat_dim + feat_dim + time_dim, feat_dim)
               
    def forward(self, src, src_t, seq, seq_t, seq_e, seq_etype, seq_utype, seq_vtype, mask):
        """"Attention based temporal attention forward pass
        args:
          src:   float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, 1, Dt], Dt == D
          seq:   float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          seq_etype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          seq_utype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          seq_vtype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          mask:  boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.
        returns:
          output: float Tensor of shape [B, D]
        """
        neighbors_features = torch.cat([seq, seq_t, seq_e], dim=2)
        neighbor_embs = self.linear_1(neighbors_features)
        neighbors_mean = F.relu(torch.mean(neighbor_embs, dim=1))

        source_features = torch.cat([src, src_t.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_mean, source_features], dim=1)
        source_embedding = self.linear_2(source_embedding)
        
        return source_embedding, None



