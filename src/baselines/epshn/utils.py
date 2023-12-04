import torch
import numpy as np


def calc_precision_at_1(embeddings, labels):
    num_correct = 0
    for i in range(len(embeddings)):
        distances = np.linalg.norm(embeddings - embeddings[i], axis=1)

        sorted_idxs = np.argsort(distances)
        closest = sorted_idxs[1]

        if labels[i] == labels[closest]:
            num_correct += 1

    precision_at_1 = num_correct / len(embeddings)
    return precision_at_1


def calc_recall_at_k(T, Y, k):
    """
    T: target labels
    Y: predicted labels
    k: k
    """

    s = 0
    for t, y in zip(T, Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def get_recall(x, y, ds_name):
    k_list = [1]
    y = torch.Tensor(y)
    x = torch.Tensor(x)

    if ds_name == "CUB":
        k_list = [1, 2, 4, 8]
    else:
        k_list = [1, 10, 100]

    dist_m = x @ x.T

    y_cur = y[dist_m.topk(1+max(k_list), largest=True)[1][:, 1:]]

    recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
    return recall, k_list
