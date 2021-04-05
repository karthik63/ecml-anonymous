import numpy as np
from os.path import join as oj
import json
import networkx as nx
import pandas as pd
import scipy
from scipy import sparse
import time
from tensorflow.python.ops.op_selector import get_generating_ops

path_embs = '/scratch/scratch1/vk/Experiments/eigen_embeddings/FB_long/node_r.npy'
path_embs_desc = '/scratch/scratch1/vk/Experiments/eigen_embeddings/FB_long/desc_r.npy'

# path_embs = '/scratch/scratch1/vk/Experiments/eigen_embeddings/FB_short/node_r.npy'
# path_embs_desc = '/scratch/scratch1/vk/Experiments/eigen_embeddings/FB_short/desc_r.npy'

# path_embs = '/scratch/scratch1/vk/Experiments/eigen_embeddings/OWE_FB_short_correct/node_r.npy'
# path_embs_desc = '/scratch/scratch1/vk/Experiments/eigen_embeddings/OWE_FB_short_correct/desc_r.npy'

# path_embs = '/scratch/scratch1/vk/Experiments/eigen_embeddings/OWE_FB_short/node_r.npy'
# path_embs_desc = '/scratch/scratch1/vk/Experiments/eigen_embeddings/OWE_FB_short/desc_r.npy'

# path_embs = '/scratch/scratch1/vk/Experiments/eigen_embeddings/OWE_FB_long/node_r.npy'
# path_embs_desc = '/scratch/scratch1/vk/Experiments/eigen_embeddings/OWE_FB_long/desc_r.npy'

embs = np.load(path_embs)
embs_desc = np.load(path_embs_desc)

dataset = 'FB15k-237-OWE'

print(path_embs)
print(path_embs_desc)

dataset_dir = oj('/scratch/scratch1/vk/Datasets', dataset)

n_closed = len(pd.read_csv(oj(dataset_dir, 'entity2id.txt')
                           , header=None,).values)
n_open = len(pd.read_csv(oj(dataset_dir, 'entity2id_zeroshot.txt')
                           , header=None,).values)

n_total = n_closed + n_open
def recentre(np_emb):
    average_value = np.average(np_emb, axis=0)
    np_emb -= average_value

    np_emb /= np.linalg.norm(np_emb, axis=1, keepdims=True)
    test_norm = np.linalg.norm(np_emb, axis = 1)

    return np_emb


def get_a_matrix(feat_Mat, avgdeg=100):
    '''
    Get adjacent matrix from file
    '''

    # feat_Mat = feat_Mat.astype(np.float16)

    print('getting_a_matrix')
    n_nodes = feat_Mat.shape[0]
    print(n_nodes)
    #FB
    # avgdeg=n_nodes
    #WN
    # avgdeg=4
    #yg
    # avgdeg=14
    # avgdeg = 5
    # avg_deg = n_nodes * 0.001

    n_edges = int(n_nodes * avgdeg / 2)
    # avgdeg = int(n_nodes * 0.01)

    a_square = np.sum(feat_Mat * feat_Mat, 1, keepdims=True)
    b_square = np.sum(feat_Mat * feat_Mat, 1, keepdims=True).T
    two_a_b = 2*feat_Mat @ (feat_Mat.transpose())

    print(1)

    mat = a_square + b_square - two_a_b

    print(2)
    vals_partitioned = np.partition(mat, avgdeg, axis=1)
    partition_values = np.expand_dims(vals_partitioned[:, avgdeg], 1)

    print(3)
    condition = mat < partition_values

    mat = np.where(condition, np.ones_like(mat, dtype=np.int8), np.zeros_like(mat, dtype=np.int8)).astype(np.int8)

    print(4)

    print(np.sum(mat))

    # G = nx.from_numpy_array(mat)
    # edges=sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
    # edges=edges[:avgdeg*n_nodes]
    # H=nx.Graph()
    # H.add_edges_from(edges)
    # adjacent = nx.to_scipy_sparse_matrix(H, format='dok')
    sparse_mat = sparse.dok_matrix(mat)
    print('done')

    return mat

def get_eigen_values(np_emb):
    top_k = 10
    n_rows = len(np_emb)
    a_matrix = (np_emb @ np_emb.T + 1) / 2
    print(a_matrix.shape)
    # sorted = np.argsort(-a_matrix, axis=1)
    # print('done_sorting')
    # top_k_indices = sorted[:, :top_k]
    # empty_matrix = np.zeros([n_rows, n_rows], np.float32)
    # indices = np.reshape(np.arange(n_rows), [-1,1])
    # indices_stacked = indices.copy()
    # for i in range(top_k-1):
    #     indices_stacked = np.hstack((indices_stacked, indices))
    #
    # print(indices_stacked.shape)
    #
    # indices_flattened = indices_stacked.flatten()
    # top_k_indices_flattened = top_k_indices.flatten()
    #
    # empty_matrix[indices_flattened, top_k_indices_flattened] = a_matrix[indices_flattened, top_k_indices_flattened]
    #
    # top_k_a_sparse = sparse.coo_matrix(empty_matrix)

    # print('converting_to_a_graph')
    # G = nx.convert_matrix.from_numpy_matrix(empty_matrix)
    ## G = nx.convert_matrix.from_scipy_sparse_matrix(top_k_a_sparse)
    # print('done converting to graph')

    # a_matrix = get_a_matrix(np_emb)

    print('gonna find values')
    time_now = time.time()
    # L = nx.linalg.laplacianmatrix.laplacian_matrix(G).A
    # print(np.sum(a_matrix, axis=1, keepdims=True))
    values = np.linalg.eigvalsh(np.diag(np.sum(a_matrix, axis=1)) - a_matrix)
    print('done finding values ', (time.time() - time_now) / 60)

    # values = nx.linalg.spectrum.laplacian_spectrum(G)
    # print('done finding eigen values')
    return values

def get_k_value(values):
    threshhold = 0.9 * np.sum(values)
    values_sorted = values[np.argsort(-values)]
    sum = 0
    for i in range(len(values)):
        sum += values_sorted[i]
        if sum > threshhold:
            break

    print('K: ', i+1)
    return i+1

def get_total(np_emb, np_emb_desc, min_k):
    np_emb = np_emb[:min_k]
    np_emb_desc = np_emb_desc[:min_k]

    ans_mean = np.mean((np_emb - np_emb_desc)**2)
    ans_sum = np.sum((np_emb - np_emb_desc)**2)

    return ans_mean, ans_sum

# def get_hubness(np_emb, source_emb):
#     a_matrix = np_emb @ np_emb.T
#     a_matrix = (a_matrix + 1) / 2
#     eye = np.eye(len(np_emb))
#     eye -= 1
#     eye *= -1
#     a_matrix *= eye
#     # print(eye[0])
#     # print(a_matrix[0])
#     sorted = np.argsort(-a_matrix, axis=1)
#     nnbrs = np.squeeze(sorted)[:,1]
#
#     counts = np.zeros(len(a_matrix), np.int32)
#
#     n_nodes = len(a_matrix)
#
#     for nbr in nnbrs:
#         counts[nbr] += 1
#
#     ten_perc_nodes = int(0.1 * n_nodes)
#     fifty_perc_nodes = int(0.5 * n_nodes)
#
#     counts = counts[np.argsort(-counts)]
#
#     ten_total = 0
#     fifty_total = 0
#     total_total = 0
#
#     for hub_10, c in enumerate(counts, start=1):
#         ten_total += c
#         if ten_total >= ten_perc_nodes:
#             break
#
#     for hub_50, c in enumerate(counts, start=1):
#         fifty_total += c
#         if fifty_total >= fifty_perc_nodes:
#             break
#
#     for hub_100, c in enumerate(counts, start=1):
#         total_total += c
#         if total_total >= n_nodes:
#             break
#
#     print('hub_10: ', hub_10)
#     print('hub_50: ', hub_50)
#     print('hub_100: ', hub_100)
#
#     return hub_10, hub_100

def get_hubness(np_emb, source_emb):
    a_matrix = np_emb @ source_emb.T
    a_matrix = (a_matrix + 1) / 2
    eye = np.eye(len(np_emb))
    eye -= 1
    eye *= -1
    a_matrix *= eye

    sorted = np.argsort(-a_matrix, axis=1)
    nnbrs = np.squeeze(sorted)[:,1]

    counts = np.zeros(len(a_matrix), np.int32)

    n_nodes = len(a_matrix)

    for nbr in nnbrs:
        counts[nbr] += 1

    ten_perc_nodes = int(0.1 * n_nodes)
    fifty_perc_nodes = int(0.5 * n_nodes)

    counts = counts[np.argsort(-counts)]

    ten_total = 0
    fifty_total = 0
    total_total = 0

    for hub_10, c in enumerate(counts, start=1):
        ten_total += c
        if ten_total >= ten_perc_nodes:
            break

    for hub_50, c in enumerate(counts, start=1):
        fifty_total += c
        if fifty_total >= fifty_perc_nodes:
            break

    for hub_100, c in enumerate(counts, start=1):
        total_total += c
        if total_total >= n_nodes:
            break

    print('hub_10: ', hub_10)
    print('hub_50: ', hub_50)
    print('hub_100: ', hub_100)

    return hub_10, hub_100

def get_eigen_difference():
    values, values_desc = get_eigen_values(embs), get_eigen_values(embs_desc)

    values = values[np.argsort(-values)]
    values_desc = values_desc[np.argsort(-values_desc)]

    print(values)
    print(values_desc)

    k1, k2 = get_k_value(values), get_k_value(values_desc)
    k = min(k1, k2)

    eigen_difference_mean, eigen_difference_sum = get_total(values, values_desc, k)

    print("eigen difference mean: ", eigen_difference_mean)
    print("eigen difference sum: ", eigen_difference_sum)

embs, embs_desc = recentre(embs), recentre(embs_desc)
embs, embs_desc = recentre(embs), recentre(embs_desc)

embs = embs[:12324]
embs_desc = embs_desc[:12324]

# get_hubness(embs_desc, embs)
get_eigen_difference()

