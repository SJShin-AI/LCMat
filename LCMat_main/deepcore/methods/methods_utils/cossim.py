import numpy as np
import torch
import os
import csv

def save_important_statistics(args, dict, name):

    os.makedirs(os.path.join(args.save_path,'csv'),exist_ok=True)
    with open(os.path.join(args.save_path, 'csv','Config_'+name +'_'+args.checkpoint_name +'.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for key in dict:
            w.writerow([key,dict[key]])
    return
def cossim_np(v1, v2):
    num = np.dot(v1, v2.T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0.
    return 0.5 + 0.5 * res

def max_diff_np(v1, v2):
    max_curv = np.max(np.abs(v1-v2))
    return max_curv

def sum_diff_np(v1, v2):
    sum_curv = np.sum(np.abs(v1-v2))
    return sum_curv

def l2_norm_np(v1, v2):
    m = v1.shape[0]  # x has shape (m, d)
    n = v2.shape[0]  # y has shape (n, d)
    x2 = np.sum(v1 ** 2, axis=1).reshape((m, 1))
    y2 = np.sum(v2 ** 2, axis=1).reshape((1, n))
    xy = v1.dot(v2.T)  # shape is (m, n)
    dists = np.sqrt(x2 + y2 - 2 * xy)  # shape is (m, n)
    dists[np.isnan(dists)] = 0

    return dists

def l1_norm_np(v1, v2):
    l1_dist = np.abs(v1[:, None, :] - v2[None, :, :]).sum(axis=-1)
    return l1_dist

def hessian_pick(hessians,K=100):
    dominant_theta = np.mean(np.abs(hessians),axis=0)
    argsort = np.argsort(dominant_theta)[::-1]
    pick = argsort[:K]
    hessians_reduced = hessians[:,pick]
    return hessians_reduced, pick

def hessian_pick_var(hessians,K=100):
    dominant_theta = np.var(hessians,axis=0)
    sorted_var = np.sort(dominant_theta)[::-1]

    argsort = np.argsort(dominant_theta)[::-1]
    pick = argsort[:K]
    hessians_reduced = hessians[:,pick]
    return hessians_reduced, pick, sorted_var


def cossim_pair_np(v1):
    num = np.dot(v1, v1.T)
    norm = np.linalg.norm(v1, axis=1)
    denom = norm.reshape(-1, 1) * norm
    res = num / denom
    res[np.isneginf(res)] = 0.
    return 0.5 + 0.5 * res

def cossim(v1, v2):
    num = torch.matmul(v1, v2.T)
    denom = torch.norm(v1, dim=1).view(-1, 1) * torch.norm(v2, dim=1)
    res = num / denom
    res[torch.isneginf(res)] = 0.
    return 0.5 + 0.5 * res

def cossim_pair(v1):
    num = torch.matmul(v1, v1.T)
    norm = torch.norm(v1, dim=1)
    denom = norm.view(-1, 1) * norm
    res = num / denom
    res[torch.isneginf(res)] = 0.
    return 0.5 + 0.5 * res

