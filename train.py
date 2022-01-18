import torch
import dgl
from utils.utils import *
from utils.rwr_scoring import rwr_scores
from utils.test import test
from utils.node2vec import load_walks
from model.model import Model
from model.negative_sampling import negative_sampling_exact
import networkx as nx
import argparse
import time, os
from dataset.data import Train_Data
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--dim', type=int, default=128, help='dimension of output embeddings.')
parser.add_argument('--num_layer', type=int, default=1, help='number of layers.')
parser.add_argument('--ratio', type=float, default=0.2, help='training ratio.')
parser.add_argument('--coeff1', type=float, default=1.0, help='coefficient for within-network link prediction loss.')
parser.add_argument('--coeff2', type=float, default=1.0, help='coefficient for anchor link prediction loss.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs.')
parser.add_argument('--batch_size', type=int, default=300, help='batch_size.')
parser.add_argument('--walks_num', type=int, default=100,
                        help='length of walk per user node.')
parser.add_argument('--N_steps', type=int, default=10,
                        help='burn-in iteration.')
parser.add_argument('--N_negs', type=int, default=20,
                        help='number of negative samples per anchor node.')
parser.add_argument('--p', type=int, default=1,
                        help='return hyperparameter. Default is 1.')
parser.add_argument('--q', type=int, default=1,
                        help='inout hyperparameter. Default is 1.')
parser.add_argument('--walk_length', type=int, default=80,
                    help='Length of walk per source. Default is 80.')
parser.add_argument('--num_walks', type=int, default=10,
                    help='Number of walks per source. Default is 10.')
parser.add_argument('--dataset', type=str, default='new_ACM-DBLP', help='dataset name.')
parser.add_argument('--use_attr', action='store_true')
parser.add_argument('--gpu', type=int, default=0, help='cuda number.')
parser.add_argument('--dist', type=str, default='L1', help='distance for scoring.')

args = parser.parse_args()
edge_index1, edge_index2, x1, x2, anchor_links, test_pairs = load_data('dataset/' + args.dataset, args.ratio, args.use_attr)

anchor_nodes1, anchor_nodes2 = anchor_links[:, 0], anchor_links[:, 1]
anchor_links2 = anchor_nodes2

G1, G2 = nx.Graph(), nx.Graph()
G1.add_edges_from(edge_index1)
G2.add_edges_from(edge_index2)
n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
for edge in G1.edges():
    G1[edge[0]][edge[1]]['weight'] = 1
for edge in G2.edges():
    G2[edge[0]][edge[1]]['weight'] = 1

################################################################################################
# run node2vec or load from existing file for positive context pairs
t0 = time.time()
if not os.path.isfile('dataset/node2vec_context_pairs_%s_%.1f.npz' % (args.dataset, args.ratio)):
    # run node2vec from scratch
    walks1 = load_walks(G1, args.p, args.q, args.num_walks, args.walk_length)
    walks2 = load_walks(G2, args.p, args.q, args.num_walks, args.walk_length)
    context_pairs1 = extract_pairs(walks1, anchor_nodes1)
    context_pairs2 = extract_pairs(walks2, anchor_nodes2)
    context_pairs1, context_pairs2 = balance_inputs(context_pairs1, context_pairs2)
    np.savez('dataset/node2vec_context_pairs_%s_%.1f.npz' % (args.dataset, args.ratio), context_pairs1=context_pairs1, context_pairs2=context_pairs2)
else:
    contexts = np.load('dataset/node2vec_context_pairs_%s_%.1f.npz' % (args.dataset, args.ratio))
    context_pairs1 = contexts['context_pairs1']
    context_pairs2 = contexts['context_pairs2']
print('Finished positive context pair sampling in %.2f seconds' % (time.time() - t0))

################################################################################################
# run random walk with restart or load from existing file for pre-positioning
t0 = time.time()
if not os.path.isfile('dataset/rwr_emb_%s_%.1f.npz' % (args.dataset, args.ratio)):
    rwr_score1, rwr_score2 = rwr_scores(G1, G2, anchor_links)
    np.savez('dataset/rwr_emb_%s_%.1f.npz' % (args.dataset, args.ratio), rwr_score1=rwr_score1, rwr_score2=rwr_score2)
else:
    scores = np.load('dataset/rwr_emb_%s_%.1f.npz' % (args.dataset, args.ratio))
    rwr_score1, rwr_score2 = scores['rwr_score1'], scores['rwr_score2']

################################################################################################
# Set initial relative positions
position_score1, position_score2 = anchor_emb(G1, G2, anchor_links)
for node in G1.nodes:
    if node not in anchor_nodes1:
        position_score1[node] += rwr_score1[node]
for node in G2.nodes:
    if node not in anchor_nodes2:
        position_score2[node] += rwr_score2[node]
x1 = (position_score1, x1) if args.use_attr else position_score1
x2 = (position_score2, x2) if args.use_attr else position_score2
print('Finished initial relative positioning in %.2f seconds' % (time.time() - t0))

################################################################################################
# merge input networks into a world-view network
t0 = time.time()
node_mapping1 = np.arange(G1.number_of_nodes()).astype(np.int64)
edge_index, edge_types, x, node_mapping2 = merge_graphs(edge_index1, edge_index2, x1, x2, anchor_links)
print('Finished merging networks in %.2f seconds' % (time.time() - t0))

# input node features: (one-hot encoding, position, optional - node attributes)
x1 = np.arange(len(x[0]), dtype=np.int64) if args.use_attr else np.arange(len(x), dtype=np.int64)
x2 = x[0].astype(np.float32) if args.use_attr else x.astype(np.float32)
x = (x1, x2, x[1]) if args.use_attr else (x1, x2)

args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:%d' % args.gpu

landmark = torch.from_numpy(anchor_nodes1).to(args.device)
num_nodes = x[0].shape[0]
num_attrs = x[2].shape[1] if args.use_attr else 0
num_anchors = x[1].shape[1]

model = Model(num_nodes, args.dim, landmark, args.dist, num_anchors=num_anchors, num_attrs=num_attrs)

################################################################################################
# to device
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
g = dgl.graph((edge_index.T[0], edge_index.T[1]), device=args.device)
x1 = torch.from_numpy(x[0]).to(args.device)
x2 = torch.from_numpy(x[1]).to(args.device)
x = (x1, x2, torch.from_numpy(x[2]).to(args.device)) if args.use_attr else (x1, x2)
edge_types = torch.from_numpy(edge_types).to(args.device)
node_mapping1 = torch.from_numpy(node_mapping1).to(args.device)
node_mapping2 = torch.from_numpy(node_mapping2).to(args.device)

################################################################################################
# prepare training data
dataset = Train_Data(context_pairs1, context_pairs2)
data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
data_loader_size = len(data_loader)

################################################################################################
# start training
pn_examples1, pn_examples2, pnc_examples1, pnc_examples2 = [], [], [], []
t_neg_sampling, t_get_emb, t_loss, t_model = 0, 0, 0, 0
total_loss = 0

topk = [1, 10, 30, 50, 100]
max_hits = np.zeros(len(topk), dtype=np.float32)
max_hit_10, max_hit_30, max_epoch = 0, 0, 0


for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(data_loader):
        nodes1, nodes2 = data
        nodes1 = nodes1.to(args.device)
        nodes2 = nodes2.to(args.device)
        anchor_nodes1 = nodes1[:, 0].reshape((-1,))
        pos_context_nodes1 = nodes1[:, 1].reshape((-1,))
        anchor_nodes2 = nodes2[:, 0].reshape((-1,))
        pos_context_nodes2 = nodes2[:, 1].reshape((-1,))
        # forward pass
        t0 = time.time()
        out_x = model(g, x, edge_types)
        t_model += (time.time() - t0)

        t0 = time.time()
        context_pos1_emb = out_x[node_mapping1[pos_context_nodes1]]
        context_pos2_emb = out_x[node_mapping2[pos_context_nodes2]]

        pn_examples1, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes1, node_mapping1,
                                                          'p_n', 'g1')
        pn_examples2, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes2, node_mapping2,
                                                          'p_n', 'g2')
        pnc_examples1, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes1, node_mapping1,
                                                            'p_nc', 'g1', node_mapping2=node_mapping2)
        pnc_examples2, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes2, node_mapping2,
                                                            'p_nc', 'g2', node_mapping2=node_mapping1)

        t_neg_sampling += (time.time() - t0)

        # get node embeddings
        t0 = time.time()

        pn_examples1 = torch.from_numpy(pn_examples1).reshape((-1,)).to(args.device)
        pn_examples2 = torch.from_numpy(pn_examples2).reshape((-1,)).to(args.device)
        pnc_examples1 = torch.from_numpy(pnc_examples1).reshape((-1,)).to(args.device)
        pnc_examples2 = torch.from_numpy(pnc_examples2).reshape((-1,)).to(args.device)

        anchor1_emb = out_x[node_mapping1[anchor_nodes1]]
        anchor2_emb = out_x[node_mapping2[anchor_nodes2]]
        context_neg1_emb = out_x[node_mapping1[pn_examples1]]
        context_neg2_emb = out_x[node_mapping2[pn_examples2]]
        anchor_neg1_emb = out_x[node_mapping2[pnc_examples1]]
        anchor_neg2_emb = out_x[node_mapping1[pnc_examples2]]

        input_embs = (anchor1_emb, anchor2_emb, context_pos1_emb, context_pos2_emb, context_neg1_emb,
                      context_neg2_emb, anchor_neg1_emb, anchor_neg2_emb)

        t_get_emb += (time.time() - t0)

        # compute loss
        t0 = time.time()
        loss1, loss2 = model.loss(input_embs)
        total_loss = args.coeff1 * loss1 + args.coeff2 * loss2
        t_loss += (time.time() - t0)

        print("Epoch:{}, Iteration:{}, Training loss:{}, Loss1:{},"
              " Loss2:{}".format(epoch + 1, i + 1, round(total_loss.item(), 4), round(loss1.item(), 4),
                                 round(loss2.item(), 4)))

        # backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_t_model = round(t_model / ((epoch+1) * data_loader_size), 2)
    avg_t_neg_sampling = round(t_neg_sampling / ((epoch+1) * data_loader_size), 2)
    avg_t_get_emb = round(t_get_emb / ((epoch+1) * data_loader_size), 2)
    avg_t_loss = round(t_loss / ((epoch+1) * data_loader_size), 2)
    time_cost = [avg_t_model, avg_t_neg_sampling, avg_t_get_emb, avg_t_loss]

    train_hits = test(model, topk, g, x, edge_types, node_mapping1, node_mapping2, anchor_links, anchor_links2, args.dist)
    hits = test(model, topk, g, x, edge_types, node_mapping1, node_mapping2, test_pairs, anchor_links2, args.dist, 'testing')
    print("Epoch:{}, Training loss:{}, Train_Hits:{},  Test_Hits:{}, Time:{}".format(
        epoch+1, round(total_loss.item(), 4), train_hits, hits, time_cost))

    if hits[2] > max_hit_30 or (hits[2] == max_hit_30 and hits[1] > max_hits[1]):
        max_hit_30 = hits[2]
        max_hits = hits
        max_epoch = epoch + 1

    print("Max test hits:{} at epoch: {}".format(max_hits, max_epoch))

if args.use_attr:
    with open('results/results_%s_attr_%.1f.txt' % (args.dataset, args.ratio), 'a+') as f:
        f.write(', '.join([str(x) for x in max_hits]) + '\n')
else:
    with open('results/results_%s_%.1f.txt' % (args.dataset, args.ratio), 'a+') as f:
        f.write(', '.join([str(x) for x in max_hits]) + '\n')







