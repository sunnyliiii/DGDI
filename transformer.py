import math
import os
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv,GATConv
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, adj,num_layers=2,
                 dropout=0.5, use_bn=True, heads=2, out_heads=1):
        super(GAT, self).__init__()
        self.adj = adj
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, dropout=dropout, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, dropout=dropout, heads=out_heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu 
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        device = data.device 
        edge_index=torch.tensor(np.array(self.adj.nonzero()), dtype=torch.long).to(device)

        batch_size, seq_len, in_channels = data.shape
        x = data.view(batch_size * seq_len, in_channels)  

        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = x.view(batch_size, seq_len, -1)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, adj, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()
        self.adj = adj
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        device = x.device 
        edge_index=torch.tensor(np.array(self.adj.nonzero()), dtype=torch.long).to(device)
        edge_weight=None

        batch_size, seq_len, in_channels = x.shape
        x = x.view(batch_size * seq_len, in_channels)  

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = x.view(batch_size, seq_len, -1)
        return x
    
def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention=torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) #[N, N]
        normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N,1]
        attention=attention/normalizer


    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, adj, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()
        self.adj = adj
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act=use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        device = x.device 
        edge_index=torch.tensor(np.array(self.adj.nonzero()), dtype=torch.long).to(device)
        edge_weight = None
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x, edge_index, edge_weight)

            if self.residual:
                x = x.view(layer_[i].shape)

                x = self.alpha * x + (1 - self.alpha) * layer_[i]

            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  
    
class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, adj, num_layers=2, num_heads=1, 
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, graph_weight=0.8):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, adj, num_layers, num_heads, alpha, dropout, use_bn, use_residual, use_weight)
        self.gnn = GCN(in_channels, hidden_channels, out_channels, adj, num_layers, dropout)
        self.graph_weight = graph_weight
        
        self.fc = nn.Linear(hidden_channels, out_channels)
        
        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, data):
        x1 = self.trans_conv(data)
        x2 = self.gnn(data)
        x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        x = self.fc(x)
        return x
    
    def get_attentions(self, x):
        return self.trans_conv.get_attentions(x)

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        self.gnn.reset_parameters()