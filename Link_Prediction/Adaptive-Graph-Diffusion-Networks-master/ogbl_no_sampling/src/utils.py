import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy_indexed as npi
import dgl
from tqdm import tqdm


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def count_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def adjust_lr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_

def to_undirected(graph):
    print(f'Previous edge number: {graph.number_of_edges()}')
    graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
    keys = list(graph.edata.keys())
    for k in keys:
        if k != 'weight':
            graph.edata.pop(k)
        else:
            graph.edata[k] = graph.edata[k].float()
    graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True, aggregator='sum')
    print(f'After adding reversed edges: {graph.number_of_edges()}')
    return graph

def filter_edge(split, nodes):
    mask = npi.in_(split['edge'][:,0], nodes) & npi.in_(split['edge'][:,1], nodes)
    print(len(mask), mask.sum())
    split['edge'] = split['edge'][mask]
    split['year'] = split['year'][mask]
    split['weight'] = split['weight'][mask]
    if 'edge_neg' in split.keys():
        mask = npi.in_(split['edge_neg'][:,0], nodes) & npi.in_(split['edge_neg'][:,1], nodes)
        split['edge_neg'] = split['edge_neg'][mask]
    return split

def evaluate_hits(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results
        

def evaluate_mrr(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_valid_pred,
    })['mrr_list'].mean().item()

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr, valid_mrr, test_mrr)
    
    return results

def evaluate_rocauc(evaluator, pos_train_pred, neg_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    results = {}
    train_rocauc = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_train_pred,
    })[f'rocauc']
    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'rocauc']
    test_rocauc = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'rocauc']
    results['ROC-AUC'] = (train_rocauc, valid_rocauc, test_rocauc)
    return results
    
# def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
#     train_auc = roc_auc_score(train_true, train_pred)
#     valid_auc = roc_auc_score(val_true, val_pred)
#     test_auc = roc_auc_score(test_true, test_pred)
#     results = {}
#     results['AUC'] = (train_auc, valid_auc, test_auc)

#     return results

def precompute_adjs(A):
    '''
    0:cn neighbor
    1:aa
    2:ra
    '''
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    w1 = A.sum(axis=0) / A.sum(axis=0)
    temp = np.log(A.sum(axis=0))
    temp = 1 / temp
    temp[np.isinf(temp)] = 0
    D_log = A.multiply(temp).tocsr()
    D = A.multiply(w).tocsr()
    D_common = A.multiply(w1).tocsr()
    return (A, D, D_log, D_common)


def RA_AA_CN(adjs, edge):
    A, D, D_log, D_common = adjs
    ra = []
    cn = []
    aa = []

    src, dst = edge
    # if len(src) < 200000:
    #     ra = np.array(np.sum(A[src].multiply(D[dst]), 1))
    #     aa = np.array(np.sum(A[src].multiply(D_log[dst]), 1))
    #     cn = np.array(np.sum(A[src].multiply(D_common[dst]), 1))
    # else:
    batch_size = 1000000
    ra, aa, cn = [], [], []
    for idx in tqdm(DataLoader(np.arange(src.size(0)), batch_size=batch_size, shuffle=False, drop_last=False)):
        ra.append(np.array(np.sum(A[src[idx]].multiply(D[dst[idx]]), 1)))
        aa.append(np.array(np.sum(A[src[idx]].multiply(D_log[dst[idx]]), 1)))
        cn.append(np.array(np.sum(A[src[idx]].multiply(D_common[dst[idx]]), 1)))
    ra = np.concatenate(ra, axis=0)
    aa = np.concatenate(aa, axis=0)
    cn = np.concatenate(cn, axis=0)

        # break
    scores = np.concatenate([ra, aa, cn], axis=1)
    return torch.FloatTensor(scores)
