import numpy as np

def load_data(file_name, p, use_attr):
    '''
    Load dataset.
    @param file_name: file name of the dataset
    @param p: raining ratio
    @param use_attr: whether to use input node attributes
    @return:
        edge_index1, edge_index2: edge list of graph G1, G2
        x1, x2: node attributes of graph G1, G2
        anchor_links: training node alignment, i.e., anchor links
        test_pairs: test node alignment
    '''

    data = np.load('%s_%.1f.npz' % (file_name, p))
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)
    if use_attr:
        x1, x2 = data['x1'].astype(np.float32), data['x2'].astype(np.float32)
    else:
        x1, x2 = None, None

    return edge_index1, edge_index2, x1, x2, anchor_links, test_pairs

def merge_graphs(edge_index1, edge_index2, x1, x2, anchor_links):
    '''
    Merge input two graphs into a world-view graph.
    @param edge_index1: edge indices of graph G1
    @param edge_index2: edge indices of graph G2
    @param x1: input node features of graph G1, either a tuple (positions, attributes) or an array of positions
    @param x2: input node features of graph G2, either a tuple (positions, attributes) or an array of positions
    @param anchor_links: anchor links for training
    @return:
        edge_index: edges in the merged graph
        x: node features in the merged graph
        node_map: node mappings for nodes in G2
    '''

    visit = 0
    n1 = x1.shape[0] if not isinstance(x1, tuple) else x1[0].shape[0]
    n2 = x2.shape[0] if not isinstance(x2, tuple) else x2[0].shape[0]
    node_mapping = {}

    for i, (node1, node2) in enumerate(anchor_links):
        node_mapping[node2] = node1

    ########################################################
    # merge node features
    x, x_pos, x_attr = [], [], []
    for i in range(n1):
        if not isinstance(x1, tuple):
            x.append(x1[i])
        else:
            x_pos.append(x1[0][i])
            x_attr.append(x1[1][i])

    for i in range(n2):
        if i not in node_mapping:
            node_mapping[i] = i + n1 - visit
            if not isinstance(x2, tuple):
                x.append(x2[i])
            else:
                x_pos.append(x2[0][i])
                x_attr.append(x2[1][i])
        else:
            visit += 1
            if not isinstance(x2, tuple):
                x[node_mapping[i]] = np.maximum(x1[node_mapping[i]], x2[i])
            else:
                x_pos[node_mapping[i]] = np.maximum(x1[0][node_mapping[i]], x2[0][i])
                x_attr[node_mapping[i]] = np.maximum(x1[1][node_mapping[i]], x2[1][i])
    if not isinstance(x1, tuple) and not isinstance(x2, tuple):
        x = np.array(x)
    else:
        x = (np.array(x_pos), np.array(x_attr))

    ########################################################
    # merge edges
    for i in range(len(edge_index2)):
        edge_index2[i][0] = node_mapping[edge_index2[i][0]]
        edge_index2[i][1] = node_mapping[edge_index2[i][1]]
    edge_index = np.vstack([edge_index1, edge_index2])
    edge_types = np.concatenate([np.zeros(len(edge_index1)), np.ones(len(edge_index2))]).astype(np.int64)

    node_map = np.zeros(n2, dtype=np.int64)
    for k, v in node_mapping.items():
        node_map[k] = v

    return edge_index, edge_types, x, node_map


def anchor_emb(g1, g2, anchor_links):
    '''
    Set relative positions for anchor nodes.
    @param g1: Network G1
    @param g2: Network G2
    @param anchor_links: array of anchor links
    @return: relative positions of anchor nodes
    '''
    n1 = g1.number_of_nodes()
    n2 = g2.number_of_nodes()
    m = len(anchor_links)
    x1 = np.zeros((n1, m), dtype=np.float32)
    x2 = np.zeros((n2, m), dtype=np.float32)
    for i, (node1, node2) in enumerate(anchor_links):
        x1[node1, i] = 1
        x2[node2, i] = 1

    return x1, x2


def extract_pairs(walks, center_nodes, window_size=2):
    context_pairs = []
    for walk in walks:
        for i, node in enumerate(walk):
            center_node = node
            if center_node in center_nodes:
                left = max(0, i - window_size)
                right = min(i+window_size+1, len(walk))
                for j in range(left, right, 1):
                    context_pairs.append((center_node, walk[j]))

    context_pairs = np.array(list(set(context_pairs)))
    np.random.shuffle(context_pairs)

    return context_pairs

def balance_inputs(context_pairs1, context_pairs2):
    if len(context_pairs1) < len(context_pairs2):
        len_diff = len(context_pairs2) - len(context_pairs1)
        idx = np.random.choice(len(context_pairs1), len_diff)
        imputes = context_pairs1[idx]
        context_pairs1 = np.vstack([context_pairs1, imputes])
    else:
        len_diff = len(context_pairs1) - len(context_pairs2)
        idx = np.random.choice(len(context_pairs2), len_diff)
        imputes = context_pairs2[idx]
        context_pairs2 = np.vstack([context_pairs2, imputes])
    return context_pairs1, context_pairs2

