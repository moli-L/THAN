import logging
import numpy as np
import torch
import torch.nn as nn
from graph import NeighborFinder
from layers import *


class HetMatchDecoder(torch.nn.Module):
    def __init__(self, num_etypes, dim, etype_feat=None):
        super().__init__()
        if etype_feat is None:
            etype_feat = torch.Tensor(num_etypes+1, dim)
        else:
            etype_feat = torch.from_numpy(etype_feat)
        self.rel_emb = nn.Parameter(etype_feat)

        self.fc1 = torch.nn.Linear(dim * 3, dim)
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
        return self.fc2(h)


class THAN(nn.Module):
    def __init__(self, ngh_finder: NeighborFinder, n_feat, e_feat, e_type_feat=None, num_n_type=1, num_e_type=1, t_dim=128, 
            use_time='time', agg_method='attn', num_layers=2, n_head=4, dropout=0.1, seq_len=None):
        super(THAN, self).__init__()
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.logger = logging.getLogger(__name__)
        
        n_feat = torch.from_numpy(n_feat.astype(np.float32))
        e_feat = torch.from_numpy(e_feat.astype(np.float32))
        self.node_embed = nn.Embedding.from_pretrained(n_feat, padding_idx=0, freeze=True)
        self.edge_embed = nn.Embedding.from_pretrained(e_feat, padding_idx=0, freeze=True)

        self.n_feat_dim = n_feat.shape[1]
        self.e_feat_dim = e_feat.shape[1]
        self.t_feat_dim = t_dim
        self.out_dim = self.n_feat_dim

        self.num_n_type = num_n_type
        self.num_e_type = num_e_type

        if e_type_feat is not None:
            e_type_feat = torch.from_numpy(e_type_feat.astype(np.float32))

        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = nn.ModuleList([AttnModel(self.n_feat_dim, 
                                                            self.e_feat_dim, 
                                                            self.t_feat_dim,
                                                            n_head=n_head, 
                                                            dropout=dropout,
                                                            num_n_type=num_n_type,
                                                            num_e_type=num_e_type,
                                                            e_type_feat=e_type_feat) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = nn.ModuleList([LSTMPool(self.n_feat_dim,
                                                            self.e_feat_dim,
                                                            self.t_feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = nn.ModuleList([MeanPool(self.n_feat_dim,
                                                                 self.e_feat_dim) for _ in range(num_layers)])
        else:
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        # time encoder
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.t_feat_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.t_feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.t_feat_dim)
        else:
            raise ValueError('invalid time option!')

        self.affinity_score = HetMatchDecoder(num_e_type, self.out_dim, e_type_feat)


    def forward(self, src_idx_l, tgt_idx_l, cut_time_l, src_utype_l, tgt_utype_l, etype_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, src_utype_l, self.num_layers, num_neighbors)
        tgt_embed = self.tem_conv(tgt_idx_l, cut_time_l, tgt_utype_l, self.num_layers, num_neighbors)
        
        score = self.affinity_score(src_embed, tgt_embed, etype_l).squeeze(dim=-1)
        
        return score.sigmoid()


    def link_contrast(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, src_utype_l, tgt_utype_l, bgd_utype_l, etype_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, src_utype_l, self.num_layers, num_neighbors)
        tgt_embed = self.tem_conv(tgt_idx_l, cut_time_l, tgt_utype_l, self.num_layers, num_neighbors)
        # fake targets
        bgd_embed = self.tem_conv(bgd_idx_l, cut_time_l, bgd_utype_l, self.num_layers, num_neighbors)
        
        pos_score = self.affinity_score(src_embed, tgt_embed, etype_l).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, bgd_embed, etype_l).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()


    def tem_conv(self, src_idx_l, cut_time_l, src_utype_l, curr_layers, num_neighbors=20, uniform=None, neg=False):
        assert(curr_layers >= 0)

        device = self.node_embed.weight.device
        batch_size = len(src_idx_l)

        src_node_feat = self.node_embed(torch.from_numpy(src_idx_l).long().to(device))

        if curr_layers == 0:
            return src_node_feat

        src_node_conv_feat = self.tem_conv(src_idx_l, cut_time_l, src_utype_l,
                                                curr_layers=curr_layers-1, 
                                                num_neighbors=num_neighbors,
                                                uniform=uniform, neg=neg)

        cut_time_l_th = torch.from_numpy(cut_time_l).float().unsqueeze(1) # [B, 1]
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th).to(device))

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch, src_ngh_etype, src_ngh_vtype \
                = self.ngh_finder.get_temporal_hetneighbor(src_idx_l, cut_time_l, num_neighbors)
        
        # get previous layer's node features
        src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)
        src_ngh_vtype_flat = src_ngh_vtype.flatten() #reshape(batch_size, -1)
        src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                src_ngh_t_batch_flat,
                                                src_ngh_vtype_flat,
                                                curr_layers=curr_layers - 1, 
                                                num_neighbors=num_neighbors,
                                                uniform=uniform,
                                                neg=neg)

        src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors * (self.num_e_type+1) , -1)

        src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
        src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
        
        src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
        src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)

        # get edge time features and edge features
        src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)  #△t = t0-ti
        src_ngn_edge_feat = self.edge_embed(src_ngh_eidx_batch)  # edge features

        # src/ngh node & edge label
        src_utype = torch.from_numpy(src_utype_l).long().to(device)
        src_ngh_etype = torch.from_numpy(src_ngh_etype).long().to(device)
        src_ngh_vtype = torch.from_numpy(src_ngh_vtype).long().to(device)

        # attention aggregation
        mask = src_ngh_node_batch_th == 0
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _ = attn_m(src_node_conv_feat, 
                            src_node_t_embed,
                            src_ngh_feat,
                            src_ngh_t_embed, 
                            src_ngn_edge_feat,
                            src_ngh_etype,
                            src_utype,
                            src_ngh_vtype,
                            mask)

        return local


