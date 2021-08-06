import torch
import functools, math
import torch.nn as nn
from dgl import function as fn
from dgl.nn.pytorch import utils
from dgl.base import DGLError


class RelGCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=True, activation=None, self_loop=True, dropout=0.0, alpha=0.5, param=True):
        '''
        RelGCN layer for network alignment.

        @param in_feat: input feature dimension.
        @param out_feat: output feature dimension.
        @param num_rels: number of relations. For two input graphs, num_rels=2.
        @param bias: whether to apply bias. Default is True.
        @param activation: whether to apply activation function. Default is None.
        @param self_loop: whether to apply self loops. Default is True.
        @param dropout: dropout rate. Default is 0.
        @param alpha: hyper-parameter in Eq. (6).
        @param param: whether to apply weight matrices.

        '''
        super(RelGCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.dropout = dropout
        self.alpha = alpha
        self.param = param

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        self.message_func = self.base_message_func

        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)


    def base_message_func(self, edges, etypes):
        '''
        Message passing function.

        @param edges: edges in the input graph.
        @param etypes: edge types in the graph, indicating which graph the edges belong to.
        @return:
            msg: messages that will be passed along edges.
        '''
        weight = self.weight
        h = edges.src['h']

        if h.dtype == torch.int64 and h.ndim == 1:
            weight = weight.view(-1, weight.shape[2])
            flat_idx = etypes * weight.shape[1] + h
            msg = weight.index_select(0, flat_idx)
        else:
            if self.param:
                weight = weight.index_select(0, etypes)
                msg = torch.bmm(h.unsqueeze(1), weight).squeeze(1)
            else:
                msg = h

        return {'msg': msg}

    def forward(self, g, feat, etypes):
        '''
        Forward pass of RelGCN layer.

        @param g: input merged graph.
        @param feat: input features.
        @param etypes: edge types of merged graph.
        @return:
            node_repr: node embedding matrix.
        '''
        if isinstance(etypes, torch.Tensor):
            if len(etypes) != g.num_edges():
                raise DGLError('"etypes" tensor must have length equal to the number of edges'
                               ' in the graph. But got {} and {}.'.format(
                    len(etypes), g.num_edges()))

        with g.local_scope():
            g.srcdata['h'] = feat
            if self.self_loop:
                if self.param:
                    loop_message = utils.matmul_maybe_select(feat[:g.number_of_dst_nodes()],
                                                             self.loop_weight)
                else:
                    loop_message = feat[:g.number_of_dst_nodes()]
            # message passing
            g.update_all(functools.partial(self.message_func, etypes=etypes),
                         fn.sum(msg='msg', out='h'))
            node_repr = g.dstdata['h'] * math.sqrt(self.alpha)

            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message * math.sqrt(1 - self.alpha)
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)

            return node_repr