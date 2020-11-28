import numpy as np

def get_ei_isi_score(vector):
    num_examples, batch_size, emb_dim = vector.shape

    # std(e(k))
    flatted_test_top_vt = vector.reshape((-1, emb_dim))
    std_e_k = np.std(flatted_test_top_vt, axis=0)
    print("std(e(k)) Shape: {}".format(std_e_k.shape))

    ei_isi_scores = []
    for ex_num in range(0, num_examples):
        # std(e(k, j))
        std_e_k_j = np.std(vector[ex_num], axis=0)

        ei_isi = std_e_k / std_e_k_j
        ei_isi_scores.append(ei_isi)

    ei_isi_scores = np.array(ei_isi_scores, dtype=np.float64)

    return ei_isi_scores

# Scaling --------------------------------------------------------------------------------------------------------------
# According to the experiment results, We adopt to min-max scale for scaling ei-isi score.
def do_min_max_scale_for_ei_isi(ei_isi_scores):
    max_ei_isi = ei_isi_scores.max()
    print("Max EI_ISI Score : {}".format(max_ei_isi))
    min_ei_isi = ei_isi_scores.min()
    print("Min EI_ISI Score : {}".format(min_ei_isi))

    ei_isi_scores = [[(x - min_ei_isi) / (max_ei_isi - min_ei_isi) for x in ei_isi] for ei_isi in ei_isi_scores]
    ei_isi_scores = np.array(ei_isi_scores, dtype=np.float64)
    return ei_isi_scores

def do_softmax_for_ei_isi(ei_isi_scores):
    import torch
    import torch.nn.functional as F
    scaled_ei_isi_scores = torch.tensor(ei_isi_scores)
    scaled_ei_isi_scores = F.softmax(scaled_ei_isi_scores, dim=1)
    scaled_ei_isi_scores = scaled_ei_isi_scores.tolist()

    return scaled_ei_isi_scores

def do_sigmoid_for_ei_isi(ei_isi_scores):
    import torch
    import torch.nn.functional as F
    scaled_ei_isi_scores = torch.tensor(ei_isi_scores)
    scaled_ei_isi_scores = F.sigmoid(scaled_ei_isi_scores)
    scaled_ei_isi_scores = scaled_ei_isi_scores.tolist()

    return scaled_ei_isi_scores