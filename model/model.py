import torch.nn as nn
from layers.RelGCN import RelGCN
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self, num_nodes, out_features, anchor_nodes, distance='inner', num_anchors=None, num_attrs=0):
        '''
        Model architecture.
        
        @param num_nodes: number of nodes in the merged graph.
        @param out_features: output feature dimension.
        @param anchor_nodes: anchor nodes in the merged graph.
        @param distance: distance function for scoring.
        @param num_anchors: number of anchor nodes.
        @param num_attrs: number of input node attributes.
        '''
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.distance = distance
        self.num_attrs = num_attrs
        if num_attrs > 0:
            self.conv_attr = RelGCN(num_attrs, out_features, 2)
            self.combine = nn.Linear(num_anchors + out_features, out_features)
        else:
            self.combine = nn.Linear(num_anchors, out_features)

        self.conv_anchor = RelGCN(num_anchors, num_anchors, 2, bias=True, param=False)
        self.conv_one_hot = RelGCN(num_nodes, out_features, 2, bias=True)

        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1)
        self.score_lin = nn.Linear(4, 1)

        self.anchor_nodes = anchor_nodes

        self.loss_func1 = nn.BCEWithLogitsLoss()
        self.loss_func2 = nn.BCEWithLogitsLoss()

    def forward(self, g, x, etype):
        '''
        Forward pass of the whole model.

        @param g: input merged graph.
        @param x: input node attributes of merged graph. Either a tuple (one-hot encoding, pre-positioning)
                  for plain graph or (one-hot encoding, pre-positioning, node attributes) for attributed graph.
        @param etype: edge types of input merged graph.
        @return:
            out_x: output node embeddings.
        '''
        x1, x2 = x[0], x[1]

        out_x1 = self.conv_one_hot(g, x1, etype)
        out_x1 = nn.functional.normalize(out_x1, p=1, dim=-1)

        anchor_emb = torch.zeros_like(x2)
        anchor_emb[self.anchor_nodes, torch.arange(len(self.anchor_nodes))] += 1
        out_x2 = self.conv_anchor(g, anchor_emb, etype)
        out_x2 = nn.functional.normalize(out_x2, p=1, dim=-1)

        anchor_emb = out_x1[self.anchor_nodes]
        att1_score = self.att1(out_x1)
        att2_score = self.att2(anchor_emb)
        att1_score = att1_score.repeat(1, self.anchor_nodes.shape[0])
        att2_score = att2_score.reshape(1, -1).repeat(att1_score.shape[0], 1)
        att_score = att1_score + att2_score
        att_score = torch.softmax(att_score, dim=1)

        out_x = torch.multiply(out_x2, att_score)
        out_x = out_x + x2  # skip connections to encode pre-positioning.

        if self.num_attrs > 0:
            out_x3 = self.conv_attr(g, x[2], etype)
            out_x3 = nn.functional.normalize(out_x3, p=1, dim=-1)
            out_x = torch.cat([out_x, out_x3], dim=1)

        out_x = self.combine(out_x)
        out_x = nn.functional.normalize(out_x, p=1, dim=-1)
        self.out_x = out_x

        return self.out_x

    def score(self, emb1, emb2, graph_name):
        '''
        Scoring function.

        @param emb1: node embeddings of graph G1.
        @param emb2: node embeddings of graph G2.
        @param graph_name: indicate whether G1 or G2.
        @return:
            predict_scores: alignment scores.
        '''
        dim = emb1.shape[1]
        emb1_1, emb1_2 = emb1[:, 0: dim//2], emb1[:, dim//2: dim]
        emb2_1, emb2_2 = emb2[:, 0: dim//2], emb2[:, dim//2: dim]
        if self.distance == 'inner':
            score1 = torch.sum(torch.multiply(emb1_1, emb2_1), dim=1).reshape((-1, 1))
            score2 = torch.sum(torch.multiply(emb1_1, emb2_2), dim=1).reshape((-1, 1))
            score3 = torch.sum(torch.multiply(emb1_2, emb2_1), dim=1).reshape((-1, 1))
            score4 = torch.sum(torch.multiply(emb1_2, emb2_2), dim=1).reshape((-1, 1))
        else:
            score1 = -torch.sum(torch.abs(emb1_1 - emb2_1), dim=1).reshape((-1, 1))
            score2 = -torch.sum(torch.abs(emb1_1 - emb2_2), dim=1).reshape((-1, 1))
            score3 = -torch.sum(torch.abs(emb1_2 - emb2_1), dim=1).reshape((-1, 1))
            score4 = -torch.sum(torch.abs(emb1_2 - emb2_2), dim=1).reshape((-1, 1))
        if graph_name == 'g1':
            scores = torch.cat([score1, score2, score3, score4], dim=1)
        else:
            scores = torch.cat([score4, score3, score2, score1], dim=1)
        predict_scores = self.score_lin(scores)

        return predict_scores


    def loss(self, input_embs):
        '''
        Loss functions for alignment.

        @param input_embs: a tuple of node embedding matrices.
            anchor1_emb, anchor2_emb: embeddings of anchors nodes in G1 and G2.
            context_pos1_emb, context_pos2_emb: embeddings of sampled context positive nodes in G1 and G2.
            context_neg1_emb, context_neg2_emb: embeddings of sampled context negative nodes in G1 and G2.
            anchor_neg1_emb, anchor_neg2_emb: embeddings of sampled negative alignment nodes in G1 and G2.
        @return:
            loss1: within-network link prediction loss.
            loss2: anchor link prediction loss.
        '''
        (anchor1_emb, anchor2_emb, context_pos1_emb, context_pos2_emb, context_neg1_emb, context_neg2_emb,
             anchor_neg1_emb, anchor_neg2_emb) = input_embs

        device = anchor1_emb.device
        num_instance1 = anchor1_emb.shape[0]
        num_instance2 = context_neg1_emb.shape[0]
        N_negs = num_instance2 // num_instance1
        dim = anchor1_emb.shape[1]

        # loss for within-network
        term1 = self.score(anchor1_emb, context_pos1_emb, 'g1')
        term2 = self.score(anchor1_emb.repeat(1, N_negs).reshape(-1, dim), context_neg1_emb, 'g1')
        term3 = self.score(anchor2_emb, context_pos2_emb, 'g2')
        term4 = self.score(anchor2_emb.repeat(1, N_negs).reshape(-1, dim), context_neg2_emb, 'g2')

        terms1 = torch.cat([term1, term2], dim=0).reshape((-1,))
        labels1 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        terms2 = torch.cat([term3, term4], dim=0).reshape((-1,))
        labels2 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        loss1 = self.loss_func1(terms1, labels1) + self.loss_func1(terms2, labels2)

        # loss for cross-network
        term5 = self.score(anchor1_emb, anchor1_emb, 'g1')
        term7 = self.score(anchor2_emb, anchor2_emb, 'g2')
        term6 = self.score(anchor1_emb.repeat(1, N_negs).reshape(-1, dim), anchor_neg1_emb, 'g1')
        term8 = self.score(anchor2_emb.repeat(1, N_negs).reshape(-1, dim), anchor_neg2_emb, 'g2')

        terms3 = torch.cat([term5, term6], dim=0).reshape((-1,))
        labels3 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        terms4 = torch.cat([term7, term8], dim=0).reshape((-1,))
        labels4 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])

        loss2 = self.loss_func2(terms3, labels3) + self.loss_func2(terms4, labels4)

        return loss1, loss2

