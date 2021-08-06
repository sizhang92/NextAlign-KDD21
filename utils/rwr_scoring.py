import networkx as nx
import numpy as np

def rwr_scores(G1, G2, anchors):
    '''
    Compute initial node embedding vectors by random walk with restart
    :param edge_list1: network G1
    :param edge_list2: network G2
    :param anchors: anchor nodes, e.g., [1,1; 2,2]
    :return: rwr vectors of two networks
    '''
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    score1, score2 = [], []

    for i, (node1, node2) in enumerate(anchors):
        s1 = nx.pagerank_scipy(G1, personalization={node1: 1})
        s2 = nx.pagerank_scipy(G2, personalization={node2: 1})
        s1_list = [0] * n1
        s2_list = [0] * n2
        for k, v in s1.items():
            s1_list[k] = v
        for k, v in s2.items():
            s2_list[k] = v
        score1.append(s1_list)
        score2.append(s2_list)
        print(i)

    rwr_score1 = np.array(score1).T
    rwr_score2 = np.array(score2).T

    return rwr_score1, rwr_score2


if __name__ == '__main__':
    data = np.load('../dataset/foursquare-twitter.npz')
    edge_index1, edge_index2 = data['edge_index1'], data['edge_index2']
    anchor_links, test_pairs = data['pos_pairs'], data['test_pairs']
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1.add_edges_from(edge_index1.T)
    G2.add_edges_from(edge_index2.T)
    import time
    t0 = time.time()
    rwr_score1, rwr_score2 = rwr_scores(G1, G2, anchor_links)
    print('running time: %.2f' % (time.time() - t0))
    np.savez('../dataset/rwr_emb_foursquare-twitter.npz', rwr_score1=rwr_score1, rwr_score2=rwr_score2)
    print('test finished.')