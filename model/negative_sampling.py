import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from scipy.special import expit


def negative_sampling_exact(x, N_negs, node1, node_mapping, target_dist_name, graph_name, removed=0, node_mapping2=None):
    '''
    Proposed sampling in NextAlign.
    @param x: node embedding matrix
    @param N_negs: number of sampled negative samples
    @param node1: mini-batch of anchor nodes
    @param node_mapping: original node indices to node indices in world-view graph
    @param target_dist_name: what distribution for sampling
    @param graph_name: indicates nodes in which graph
    @param removed: nodes that are not allowed to be sampled
    @return:
        generate_examples: nodes that are sampled
        probs: sampling probabilities
    '''
    node1 = node1.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    node_mapping = node_mapping.cpu().detach().numpy()
    if node_mapping2 is not None:
        node_mapping2 = node_mapping2.cpu().detach().numpy()
    out_features = x.shape[1]
    if target_dist_name == 'p_n':
        if graph_name == 'g1':
            emb1 = x[node_mapping[node1], 0: out_features//2]
            emb2 = x[node_mapping, 0: out_features//2]
        else:
            emb1 = x[node_mapping[node1], out_features // 2: out_features]
            emb2 = x[node_mapping, out_features // 2: out_features]
        n2 = node_mapping.shape[0]
        inner_products = np.dot(emb1, emb2.T)
        inner_products = expit(-inner_products)
        # inner_products = manhattan_distances(emb1, emb2)
        p_probs = np.clip(inner_products, 1e-7, 1e7)

    elif target_dist_name == 'p_dc':
        if graph_name == 'g1':
            emb1 = x[node_mapping[node1], out_features//2: out_features]
            emb2 = x[node_mapping2, out_features//2: out_features]
        else:
            emb1 = x[node_mapping[node1], 0: out_features//2]
            emb2 = x[node_mapping2, 0: out_features//2]
        n2 = node_mapping2.shape[0]
        inner_products = np.dot(emb1, emb2.T)
        inner_products = expit(inner_products)
        # inner_products = -manhattan_distances(emb1, emb2)

        p_probs = np.clip(inner_products, 1e-7, 1e7)

    else:
        emb1_1 = x[node_mapping[node1], 0:out_features//2]
        emb1_2 = x[node_mapping[node1], out_features // 2: out_features]
        emb2_1 = x[node_mapping2, 0:out_features//2]
        emb2_2 = x[node_mapping2, out_features // 2: out_features]
        n2 = node_mapping2.shape[0]

        inner_products = np.dot(emb1_1, emb2_2.T) * 1 \
                         + np.dot(emb1_2, emb2_1.T) * 1
        # inner_products = manhattan_distances(emb1_1, emb2_2) + manhattan_distances(emb1_2, emb2_1)
        inner_products = expit(inner_products)
        p_probs = np.clip(inner_products, 1e-7, 1e7)

    generate_examples = []
    probs = []


    for i in range(len(node1)):
        p_probs[i][node_mapping[node1[i]]] = 0
        if not isinstance(removed, int):
            p_probs[i][removed] = 0
        p = p_probs[i] / np.sum(p_probs[i])
        samples = np.random.choice(n2, N_negs, p=p)
        probs.append(list(p[samples]))
        generate_examples.append(list(samples))

    generate_examples = np.array(generate_examples, dtype=np.int64).reshape((-1,))
    probs = np.array(probs, dtype=np.float32)

    return generate_examples, probs

