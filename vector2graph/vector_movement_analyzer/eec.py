import numpy as np

def get_eec_score_per_one_example_with_threshold(vector):
    cov_arr = []

    for idx in range(0, len(vector)):
        cov = np.corrcoef(vector[idx].T)
        cov_arr.append(cov)

    # Thresholding Edge
    threshold = 0.8
    for idx in range(0, len(cov_arr)):
        cov_ex = cov_arr[idx]
        cov_ex[cov_ex < threshold] = 0

    return cov_arr