import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from vector2graph.vector_movement_analyzer.ei_isi import get_ei_isi_score, do_min_max_scale_for_ei_isi
from vector2graph.vector_movement_analyzer.eec import get_eec_score_per_one_example_with_threshold

def make_graph(vector, vector_relation):
    vector_relation = np.abs(vector_relation)

    for vector_idx in range(0, len(vector_relation)):
        vector_relation[vector_idx][vector_idx] = 0

    G = nx.from_numpy_matrix(vector_relation)

    for idx in range(0, len(vector)):
        G.nodes[idx]['node_value'] = vector[idx]

    A = nx.to_numpy_matrix(G)

    return G, A

def draw_and_save_graph(graph, image_path, with_dims=False, do_save=True, top_n=10, need_edges=True):
    # For Preserving Original Graph
    drawing_graph = graph.copy()

    # Node position --> shell_layout for fixing Node position
    pos = nx.shell_layout(drawing_graph)

    # pre-processing top_n Before drawing graph --> Invisible nodes not to be expressed.
    if (top_n != len(drawing_graph)) and (top_n != -1):
        ei_isi = []
        for k, v in list(drawing_graph.nodes(data=True)):
            ei_isi.append(v['node_value'])
        ei_isi_np = np.array(ei_isi, dtype=np.float64)
        ei_isi_sort_idx = np.argsort(-ei_isi_np)  # descending ei-isi score --> top_1, top_2, ...

        idx_for_using = []
        for ei_isi_idx in ei_isi_sort_idx:
            idx_for_using.append(ei_isi_idx)

            if len(idx_for_using) >= top_n:
                break;

        for k, v in list(drawing_graph.nodes(data=True)):
            if k not in idx_for_using:
                v['node_value'] = 0
        for k1, k2, v in list(drawing_graph.edges(data=True)):
            if (k1 not in idx_for_using) or (k2 not in idx_for_using):
                e = (k1, k2, v)
                drawing_graph.remove_edge(*e[:2])  # For selecting Edge Weight

    # Node Color --> Determined by Node's EI-ISI value
    node_color = list(drawing_graph.nodes(data=True))
    node_color = [v['node_value'] for k, v in node_color]
    # cmap = plt.cm.Blues
    cmap = plt.cm.hsv  # ---> fixed to hsv color map for painting node colors

    # Node Size --> 0 if not included in the top k nodes
    node_size = list(drawing_graph.nodes(data=True))
    node_size = [0 if v['node_value'] == 0 else 300 for k, v in node_size]  # node default size in networkx : 300

    # Edge width --> Determined by EEC value between nodes
    edge_width = list(drawing_graph.edges(data=True))
    edge_width = [v['weight'] for k1, k2, v in edge_width]

    # For Graph Representation Excluded Weighted Edge
    if not need_edges:
        edge_width = [0 for temp in edge_width]

    # Drawing
    nx.draw(drawing_graph, pos, node_color=node_color, node_size=node_size, width=edge_width, with_labels=with_dims,
            cmap=cmap)

    # Saving
    if do_save:
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()



def build_graph(labels, vector_movement, image_dir, need_edges=True, top_n=10):
    ei_isi_scores = get_ei_isi_score(vector_movement)
    ei_isi_scores = do_min_max_scale_for_ei_isi(ei_isi_scores)

    eec_scores = get_eec_score_per_one_example_with_threshold(vector_movement)

    from tqdm.auto import tqdm
    label_iterator = tqdm(labels, desc="Iteration")
    for idx, label in enumerate(label_iterator):
        ei_isi = ei_isi_scores[idx]

        eec_score = eec_scores[idx]
        eec_score = np.abs(eec_score)  # We do not consider the direction of the edge !

        label_dir = os.path.join(image_dir, label)
        file_name = str(idx) + ".jpg"
        image_path = os.path.join(label_dir, file_name)

        G, adjacency = make_graph(ei_isi, eec_score)

        draw_and_save_graph(G, image_path, with_dims=False, do_save=True, top_n=top_n, need_edges=need_edges)

    print("Representation files is dumped at ", image_dir)