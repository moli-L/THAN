import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HetMatchDecoder(torch.nn.Module):
    def __init__(self, num_etypes, dim, etype_feat=None):
        super().__init__()
        if etype_feat is None:
            etype_feat = torch.Tensor(num_etypes+1, dim)
            self.fc1 = torch.nn.Linear(dim * 3, dim)
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
        e = self.rel_emb[etype_l]
        h = torch.cat([x, e, y], dim=1)
        h = self.act(self.fc1(h))
        return self.fc2(h).squeeze(dim=-1)
    

class FFN(torch.nn.Module):
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
        # r_pri: (n*b) x N

        # attn = torch.bmm(q, k.transpose(1, 2))
        attn = torch.sum(q * k, dim=-1)  # [n*b, N]
        if r_pri is None:
            attn = attn / self.temperature
        else:
            attn = attn * r_pri / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n*b, N]
        attn = self.dropout(attn)

        # output = torch.bmm(attn, v)
        output = (attn.unsqueeze(-1) * v).sum(dim=1)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, num_n_type=1, num_e_type=1, e_type_feat=None):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.num_n_type = num_n_type
        self.num_e_type = num_e_type

        # node type and edge type vector, padding an idx
        # self.node_trans = nn.Embedding(num_n_type+1, d_model, padding_idx=0)
        # if e_type_feat is not None:
        #     self.edge_trans = nn.Embedding.from_pretrained(e_type_feat, padding_idx=0)
        # else:
        #     self.edge_trans = nn.Embedding(num_e_type+1, d_k, padding_idx=0)
        # self.d_r = self.edge_trans.weight.size(1)
        
        self.d_r = self.d_model
        self.w_qs = nn.ModuleList()
        self.w_ks = nn.ModuleList()
        self.w_vs = nn.ModuleList()
        for _ in range(num_e_type+1):
            self.w_qs.append(nn.Linear(self.d_r, n_head * d_k, bias=False))
            self.w_ks.append(nn.Linear(self.d_r, n_head * d_k, bias=False))
            self.w_vs.append(nn.Linear(self.d_r, n_head * d_v, bias=False))

        self.relation_pri = nn.Parameter(torch.ones(n_head, num_n_type + 1, num_e_type + 1))

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # self.fc = nn.Linear(n_head * d_v, d_model)
        self.fc_lins = nn.ModuleList()
        for _ in range(num_n_type+1):
            self.fc_lins.append(nn.Linear(n_head *d_v, d_model))

        self.dropout = nn.Dropout(dropout)

        self.reset_parameter()

    def reset_parameter(self):
        for i in range(len(self.w_qs)):
            nn.init.xavier_uniform_(self.w_qs[i].weight.data)
            nn.init.xavier_uniform_(self.w_ks[i].weight.data)
            nn.init.xavier_uniform_(self.w_vs[i].weight.data)
        
        # nn.init.xavier_uniform_(self.fc.weight.data)
        for fc in self.fc_lins:
            nn.init.xavier_uniform_(fc.weight.data)

    def get_transfer_Mx(self, seq_etype, seq_utype, seq_vtype):
        # seq_utype: [B]
        # seq_vtype: [B, N]
        # seq_etype: [B, N]
        device = seq_vtype.device
        zn = self.node_trans(torch.arange(self.num_n_type+1).to(device))
        re = self.edge_trans(torch.arange(self.num_e_type+1).to(device))
        M = torch.einsum('ab,cd->acbd', zn, re)  # [Nn, Ne, D, Dr]
        I = torch.eye(self.d_model, self.d_r).to(device)
        M[1:,1:] += I
        M = F.normalize(M, dim=-1)
        M = M.reshape(-1, self.d_model, self.d_r)  # [Nn*Ne, D, Dr]
        
        seq_utype = seq_utype.repeat(seq_etype.size(1), 1).t()  # [B, N]
        idx_u = seq_utype * (self.num_e_type+1) + seq_etype
        idx_v = seq_vtype * (self.num_e_type+1) + seq_etype

        return M[idx_u], M[idx_v]  # [B, N, D, Dr]

    def _resize(self, e, mask=None):
        # e:[B, N, D] mask:[B] or [B, N]
        if e.size(-1) >= self.d_r:
            res = e[:, :, :self.d_r]
        else:
            res = torch.cat([e, torch.zeros(e.size(0), e.size(1), self.d_r-e.size(2)).to(e.device)], dim=-1)

        if mask is not None:
            res = res.clone()
            res[mask] = torch.zeros_like(res[mask])
        return res

    def _transfer(self, q, k, v, seq_etype, seq_utype, seq_vtype):
        """
        params:
            q:    [B, N, D]  of src
            k, v: [B, N, D]  of ngh
            seq_etype: [B, N]
            seq_utype: [B]
            seq_vtype: [B, N]
        """
        u_trans = self.node_trans(seq_utype).unsqueeze(1)  # [B, 1, D]
        v_trans = self.node_trans(seq_vtype)  # [B, N, D]
        r_trans = self.edge_trans(seq_etype)  # [B, N, Dr]

        q = torch.sum(q * u_trans, dim=-1, keepdim=True) * r_trans + self._resize(q, seq_utype==0)
        k = torch.sum(k * v_trans, dim=-1, keepdim=True) * r_trans + self._resize(k, seq_vtype==0)
        v = torch.sum(v * v_trans, dim=-1, keepdim=True) * r_trans + self._resize(v, seq_etype==0)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)

        return q, k, v

    def _get_relation_pri_of_q(self, seq_utype, seq_etype):
        N = seq_etype.size(1)
        seq = seq_utype.repeat(N, 1).t()
        idx = seq * (self.num_e_type+1) + seq_etype
        pri = self.relation_pri.view(self.n_head, -1)
        pri = pri.index_select(1, idx.flatten())
        return pri.view(-1, N)

    def _compute_QKV_by_etype(self, q, k, v, seq_etype):
        etypes = seq_etype.unique()
        for i, etype in enumerate(etypes):
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
            q:    [B, D]  of src
            k, v: [B, N, D]  of ngh
            seq_etype: [B, N]
            seq_utype: [B]
            seq_vtype: [B, N]
            mask: [B, N]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_e = q.size(0), k.size(1)
        residual = q

        if len(q.size()) != len(k.size()):
            q = q.unsqueeze(1).repeat(1, k.size(1), 1)  # [b, N, D]

        # transfer to relation feature space input: 
        # q, k, v = self._transfer(q, k, v, seq_etype, seq_utype, seq_vtype)  # (b, N, Dr)

        # compute Q, K, V in relature feature space
        q, k, v = self._compute_QKV_by_etype(q, k, v, seq_etype)  # (b, N, n*D)

        q = q.view(sz_b, len_e, n_head, d_k)  # (b, N, n_head, D)  源节点到各个边的转移向量
        k = k.view(sz_b, len_e, n_head, d_k)
        v = v.view(sz_b, len_e, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_k) # (n*b) x N x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_k) # (n*b) x N x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_v) # (n*b) x N x dv

        pri = self._get_relation_pri_of_q(seq_utype, seq_etype)  # [n*b, N]
        # pri = None
        mask = mask.repeat(n_head, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, pri, mask=mask)  # output (n*b) x dv

        output = output.view(n_head, sz_b, d_v)  # n x b x dv
        output = output.transpose(0, 1).contiguous().view(sz_b, -1) # b x (n*dv)

        # map out to q's feature space
        out = torch.zeros_like(output)
        utypes = seq_utype.unique()
        for utype in utypes:
            mask = seq_utype==utype
            out[mask] = self.fc_lins[utype](output[mask])
        output = out

        # output = self.fc(output)
        
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output, attn


class SemanticAttention(nn.Module):
    """
    语义自注意力层
    """
    def __init__(self, in_feature, hid_feature):
        super(SemanticAttention, self).__init__()
        self.in_feature = in_feature
        self.hid_feature = hid_feature
        self.W = nn.Parameter(torch.empty(in_feature, hid_feature))
        self.q = nn.Parameter(torch.empty(in_feature, 1))
        self.b = nn.Parameter(torch.empty(1, hid_feature))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.q.data)
        nn.init.xavier_uniform_(self.b.data)

    def forward(self, x_list):
        """
        input: shape of x_list [x_list_size, (batch, n, d)]
        output: (batch, n, d)
        """
        size = len(x_list)
        batch = x_list[0].size(0)
        N = x_list[0].size(1)
        x = torch.cat(x_list, dim=1)

        h = torch.tanh(x.matmul(self.W) + self.b)
        h = h.matmul(self.q).view(batch, size, -1)
        att = F.softmax(h.mean(dim=2), dim=1)
        att = att.view(batch, size, 1, 1)
        emb = x.view(batch, size, N, self.in_feature)
        h_emb = emb.mul(att).sum(dim=1)
        return h_emb


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


class AttnModel(nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, n_head=2, dropout=0.1,
                 num_n_type=1, num_e_type=1, e_type_feat=None):
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
          e_type_feat: init edge type feature.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        
        self.model_dim = (feat_dim + edge_dim + time_dim)
        assert(self.model_dim % n_head == 0)

        self.merger = FFN(self.model_dim, feat_dim, feat_dim, feat_dim)
        
        self.multi_head_target = MultiHeadAttention(n_head, 
                                            d_model=self.model_dim, 
                                            d_k=self.model_dim // n_head, 
                                            d_v=self.model_dim // n_head, 
                                            dropout=dropout,
                                            num_n_type=num_n_type,
                                            num_e_type=num_e_type,
                                            e_type_feat=e_type_feat)
        
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
        src_e_ph = torch.zeros(src.size(0), self.edge_dim).to(src.device)
        q = torch.cat([src, src_e_ph, src_t.squeeze(1)], dim=1) # [B, D + De + Dt] -> [B, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, N, D + De + Dt] -> [B, N, D]

        # target-attention
        # output: [B, D + De + Dt], attn: [B, N, N]
        output, attn = self.multi_head_target(q=q, k=k, v=k, seq_etype=seq_etype, seq_utype=seq_utype,
                                                 seq_vtype=seq_vtype, mask=mask)
        output = self.merger(output, src)
        return output, attn


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = FFN(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, seq_etype, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = FFN(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)
        
    def forward(self, src, src_t, seq, seq_t, seq_e, seq_etype, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, x):
        # shape of x (batch, nodes, features) or (nodes, features)
        # out: shape (batch, features) or (features)
        dim = 1 if len(x.size())==3 else 0
        return torch.mean(x, dim)
