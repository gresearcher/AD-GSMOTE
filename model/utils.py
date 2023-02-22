#!/usr/bin/env python
# encoding: utf-8


from sklearn.metrics import roc_auc_score, f1_score,confusion_matrix
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
# import ran_result
import ipdb
import os
import random
from scipy.spatial.distance import pdist,squareform
from imblearn.metrics import geometric_mean_score
from scipy.io import loadmat


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5

def get_performance(output, labels, pre='valid'):
    pre_num = 0
    # print class-wise performance
    '''
    for i in range(labels.max()+1):

        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)

        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_lis[i]
    '''

    # ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output, dim=-1).cpu().detach(), average='macro',
                                  multi_class='ovr')
        gmean = geometric_mean_score(labels.cpu().detach(), torch.argmax(output, dim=-1).cpu().detach(),
                                     average='multiclass',correction=0.001)
    else:
        auc_score = roc_auc_score(labels.cpu().detach(), F.softmax(output+1e-6, dim=-1)[:, 1].cpu().detach())
        gmean = geometric_mean_score(labels.cpu().detach(), torch.argmax(output, dim=-1).cpu().detach(),correction=0.001)
        # print(gmean)
        # conf = confusion_matrix(labels.cpu().detach(), torch.argmax(output, dim=-1).cpu().detach())
        # tn, fp, fn, tp = conf_gnn.ravel()
        # gmean = conf_gmean(conf)
        # print(gmean)

    macro_F = f1_score(labels.cpu().detach(), torch.argmax(output, dim=-1).cpu().detach(), average='macro')

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    # correct_index = torch.where(correct==1.0)[0]
    corrects = correct.sum()
    # print(str(pre) + ' current auc-roc score: {:f}, current macro_F score: {:f}, current gmean score: {:f}'.format(auc_score, macro_F, gmean))
    return corrects / len(labels),auc_score,macro_F, gmean,correct
    # return corrects / len(labels),auc_score,macro_F


def get_wrong_index(output, labels, pre='valid'):

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    wrong_index = torch.where(correct == 0.0)
    return wrong_index[0].tolist(),labels[wrong_index].tolist()

def get_correct_index(correct):

    return  torch.where(correct==1.0)[0]

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # np.savetxt("cat.txt",labels_onehot,fmt="%s",delimiter=",")
    return labels_onehot


def load_twitter_data(dataset="twitter"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # embed_dir = r'./data/id_embed_all.txt'
    # embed_dir = r'./data/id_embed_all_homo.txt'
    embed_dir = r'./data/id_embed_all_homo_single.txt'
    # embed_dir = './data/id_embed_all_homo_single_1224.txt'

    # embed_dir = r'./data/id_embed_all_0.2.txt'
    # relation_dir = r'./data/relation_all.txt'
    # relation_dir = r'data/relation_all_update.txt'
    # relation_dir = r'data/relation_all_update_revision.txt'
    # relation_dir = r'data/relation_homo.txt'
    # relation_dir = r'data/relation_homo_single.txt'

    relation_dir = r'data/relation_homo_single_plus_keyword_0.1.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.03_1224.txt'

    # relation_dir = r'data/relation_homo_single_plus_keyword_0.1_1224.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.05_1224.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.09_1224.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.08_1224.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.2_1224.txt'


    # relation_dir = r'data/relation_homo_single_plus_keyword_0.07_1224.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.04_1224.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.05.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.04.txt'
    # relation_dir = r'data/relation_homo_single_plus_keyword_0.03.txt'
    # relation_dir = r'data/relation_all_update_0.2.txt'

    # relation_dir1 = r'data/relation_homo_single_plus_keyword_0.1_1224_k.txt'
    # relation_dir2 = r'data/relation_homo_single_plus_keyword_0.1_1224_u.txt'
    # relation_dir1 = r'data/relation_homo_single_plus_keyword_0.1_k.txt'
    # relation_dir2 = r'data/relation_homo_single_plus_keyword_0.1_u.txt'

    relation_dir1 = r'data/relation_homo_single_plus_keyword_0.1_3_k.txt'
    relation_dir2 = r'data/relation_homo_single_plus_keyword_0.1_3_u.txt'
    relation_dir3 = r'data/relation_homo_single_plus_keyword_0.1_3_t.txt'

    # relation_dir1 = r'data/relation_homo_single_plus_keyword_0.1_1224_3_k.txt'
    # relation_dir2 = r'data/relation_homo_single_plus_keyword_0.1_1224_3_u.txt'
    # relation_dir3 = r'data/relation_homo_single_plus_keyword_0.1_1224_3_t.txt'

    idx_features_labels = np.genfromtxt(embed_dir,
                                        dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
    # number = 27945
    number = idx_features_labels.shape[0]
    features = sp.csr_matrix(idx_features_labels[:number, 2:], dtype=np.float32)

    label = idx_features_labels[:number, 1].astype(float).astype(int)

    label[label == 3] = 2
    label[label == 4] = 3

    # build graph
    # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.float)
    idx_map = {j: i for i, j in enumerate(idx)}

    # edges_unordered = np.genfromtxt(relation_dir,
    #                                 dtype=np.int32)[:,:]




    # edges_unordered = np.unique(np.genfromtxt(relation_dir, dtype=np.int32)[:, :], axis=0)


    # filter = np.asarray([8,9,11,12])
    # filter = np.asarray([1,2,3,4,5,6,7,10])
    # filter = np.asarray([1,2,3,4,11,12])
    # filter = np.asarray([1, 2, 3, 4,5,6,7])
    filter = np.asarray([1, 2,3, 4, 5, 6, 7])

    # edges_unordered = edges_unordered[np.in1d(edges_unordered[:,2],filter)]
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(features.shape[0], features.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # # adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = adj + sp.eye(adj.shape[0])

    ####
    edges_unordered1 = np.unique(np.genfromtxt(relation_dir1, dtype=np.int32)[:, :], axis=0)

    edges1 = np.array(list(map(idx_map.get, edges_unordered1.flatten())),
                     dtype=np.int32).reshape(edges_unordered1.shape)
    adj1 = sp.coo_matrix((np.ones(edges1.shape[0]), (edges1[:, 0], edges1[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj1 = adj1 + sp.eye(adj1.shape[0])
    ####

    ####
    edges_unordered2 = np.unique(np.genfromtxt(relation_dir2, dtype=np.int32)[:, :], axis=0)

    edges2 = np.array(list(map(idx_map.get, edges_unordered2.flatten())),
                      dtype=np.int32).reshape(edges_unordered2.shape)
    adj2 = sp.coo_matrix((np.ones(edges2.shape[0]), (edges2[:, 0], edges2[:, 1])),
                         shape=(features.shape[0], features.shape[0]),
                         dtype=np.float32)

    # build symmetric adjacency matrix
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj2 = adj2 + sp.eye(adj2.shape[0])
    ####

    ####
    edges_unordered3 = np.unique(np.genfromtxt(relation_dir3, dtype=np.int32)[:, :], axis=0)

    edges3 = np.array(list(map(idx_map.get, edges_unordered3.flatten())),
                      dtype=np.int32).reshape(edges_unordered3.shape)
    adj3 = sp.coo_matrix((np.ones(edges3.shape[0]), (edges3[:, 0], edges3[:, 1])),
                         shape=(features.shape[0], features.shape[0]),
                         dtype=np.float32)

    # build symmetric adjacency matrix
    adj3 = adj3 + adj3.T.multiply(adj3.T > adj3) - adj3.multiply(adj3.T > adj3)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj3 = adj3 + sp.eye(adj3.shape[0])
    ####

    # print(Counter(np.sum(adj, axis=0).tolist()[0]))  plt.hist(x=Counter(np.sum(adj, axis=0).tolist()[0]).keys(),bins=len(Counter(np.sum(adj, axis=0).tolist()[0])),         color="steelblue",         edgecolor="black")
    index = [i for i in range(label.shape[0])]
    length = len(index)
    # ran_result.shuffle(index)
    random.shuffle(index)

    num_classes = len(set(label.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (label == i).nonzero()[0].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        # ran_result.shuffle(c_idx)
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num < 4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
        else:
            # c_num_mat[i, 0] = int(c_num / 4)
            # c_num_mat[i, 1] = int(c_num / 4)
            # c_num_mat[i, 2] = int(c_num / 2)

            # c_num_mat[i, 0] = int(c_num *0.4)
            # c_num_mat[i, 1] = int(c_num * 0.2)
            # c_num_mat[i, 2] = int(c_num * 0.4)

            # c_num_mat[i, 0] = int(c_num * 0.2)
            # c_num_mat[i, 1] = int(c_num * 0.2)
            # c_num_mat[i, 2] = int(c_num * 0.6)

            # c_num_mat[i, 0] = int(c_num * 0.1)
            # c_num_mat[i, 1] = int(c_num * 0.2)
            # c_num_mat[i, 2] = int(c_num * 0.7)

            # c_num_mat[i, 0] = int(c_num * 0.05)
            # c_num_mat[i, 1] = int(c_num * 0.2)
            # c_num_mat[i, 2] = int(c_num * 0.75)

            c_num_mat[i, 0] = int(c_num * 0.6)
            c_num_mat[i, 1] = int(c_num * 0.2)
            c_num_mat[i, 2] = int(c_num * 0.2)

        train_idx = train_idx + c_idx[:c_num_mat[i, 0]]

        val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_idx = test_idx + c_idx[
                              c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]

    random.shuffle(train_idx)

    # ipdb.set_trace()

    idx_train = torch.LongTensor(train_idx)
    idx_val = torch.LongTensor(val_idx)
    idx_test = torch.LongTensor(test_idx)
    # c_num_mat = torch.LongTensor(c_num_mat)

    y = torch.tensor(label).to(torch.long)


    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(label)[1])
    # labels_a = torch.LongTensor(labels).to(CFG.finetune_device)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
    adj3 = sparse_mx_to_torch_sparse_tensor(adj3)

    label = torch.LongTensor(label)


    # return adj, features, labels, idx_train, idx_val, idx_test,idx_pair
    # return [adj,adj1,adj2,adj3], features,label, idx_train, idx_val, idx_test
    return [adj1, adj2, adj3], features, label, idx_train, idx_val, idx_test

def load_twitter_data_imbalance(dataset="twitter"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))


    embed_dir = r'./data/id_embed_all_homo_single.txt'

    relation_dir = r'data/relation_homo_single_plus_keyword_0.1.txt'

    relation_dir1 = r'data/relation_homo_single_plus_keyword_0.1_3_k.txt'
    relation_dir2 = r'data/relation_homo_single_plus_keyword_0.1_3_u.txt'
    relation_dir3 = r'data/relation_homo_single_plus_keyword_0.1_3_t.txt'

    idx_features_labels = np.genfromtxt(embed_dir,
                                        dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
    # number = 27945
    number = idx_features_labels.shape[0]
    features = sp.csr_matrix(idx_features_labels[:number, 2:], dtype=np.float32)

    label = idx_features_labels[:number, 1].astype(float).astype(int)

    label[label == 3] = 2
    label[label == 4] = 3

    # build graph
    # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.float)
    idx_map = {j: i for i, j in enumerate(idx)}

    filter = np.asarray([1, 2,3, 4, 5, 6, 7])

    ####
    edges_unordered1 = np.unique(np.genfromtxt(relation_dir1, dtype=np.int32)[:, :], axis=0)

    edges1 = np.array(list(map(idx_map.get, edges_unordered1.flatten())),
                     dtype=np.int32).reshape(edges_unordered1.shape)
    adj1 = sp.coo_matrix((np.ones(edges1.shape[0]), (edges1[:, 0], edges1[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj1 = adj1 + sp.eye(adj1.shape[0])
    ####

    ####
    edges_unordered2 = np.unique(np.genfromtxt(relation_dir2, dtype=np.int32)[:, :], axis=0)

    edges2 = np.array(list(map(idx_map.get, edges_unordered2.flatten())),
                      dtype=np.int32).reshape(edges_unordered2.shape)
    adj2 = sp.coo_matrix((np.ones(edges2.shape[0]), (edges2[:, 0], edges2[:, 1])),
                         shape=(features.shape[0], features.shape[0]),
                         dtype=np.float32)

    # build symmetric adjacency matrix
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj2 = adj2 + sp.eye(adj2.shape[0])
    ####

    ####
    edges_unordered3 = np.unique(np.genfromtxt(relation_dir3, dtype=np.int32)[:, :], axis=0)

    edges3 = np.array(list(map(idx_map.get, edges_unordered3.flatten())),
                      dtype=np.int32).reshape(edges_unordered3.shape)
    adj3 = sp.coo_matrix((np.ones(edges3.shape[0]), (edges3[:, 0], edges3[:, 1])),
                         shape=(features.shape[0], features.shape[0]),
                         dtype=np.float32)

    # build symmetric adjacency matrix
    adj3 = adj3 + adj3.T.multiply(adj3.T > adj3) - adj3.multiply(adj3.T > adj3)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj3 = adj3 + sp.eye(adj3.shape[0])
    ####

    # print(Counter(np.sum(adj, axis=0).tolist()[0]))  plt.hist(x=Counter(np.sum(adj, axis=0).tolist()[0]).keys(),bins=len(Counter(np.sum(adj, axis=0).tolist()[0])),         color="steelblue",         edgecolor="black")
    index = [i for i in range(label.shape[0])]
    length = len(index)
    # ran_result.shuffle(index)
    random.shuffle(index)

    num_classes = len(set(label.tolist()))
    c_idxs = []  # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    # for i in range(num_classes):
    #     c_idx = (label == i).nonzero()[0].tolist()
    #     c_num = len(c_idx)
    #     print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
    #     # ran_result.shuffle(c_idx)
    #     random.shuffle(c_idx)
    #     c_idxs.append(c_idx)
    #
    #     if c_num < 4:
    #         if c_num < 3:
    #             print("too small class type")
    #             ipdb.set_trace()
    #         c_num_mat[i, 0] = 1
    #         c_num_mat[i, 1] = 1
    #         c_num_mat[i, 2] = 1
    #     else:
    #         # c_num_mat[i, 0] = int(c_num / 4)
    #         # c_num_mat[i, 1] = int(c_num / 4)
    #         # c_num_mat[i, 2] = int(c_num / 2)
    #
    #         c_num_mat[i, 0] = int(c_num *0.4)
    #         c_num_mat[i, 1] = int(c_num * 0.2)
    #         c_num_mat[i, 2] = int(c_num * 0.4)
    #
    #         # c_num_mat[i, 0] = int(c_num * 0.2)
    #         # c_num_mat[i, 1] = int(c_num * 0.2)
    #         # c_num_mat[i, 2] = int(c_num * 0.6)
    #
    #         # c_num_mat[i, 0] = int(c_num * 0.1)
    #         # c_num_mat[i, 1] = int(c_num * 0.2)
    #         # c_num_mat[i, 2] = int(c_num * 0.7)
    #
    #         # c_num_mat[i, 0] = int(c_num * 0.05)
    #         # c_num_mat[i, 1] = int(c_num * 0.2)
    #         # c_num_mat[i, 2] = int(c_num * 0.75)
    #
    #     train_idx = train_idx + c_idx[:c_num_mat[i, 0]]
    #
    #     val_idx = val_idx + c_idx[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
    #     test_idx = test_idx + c_idx[
    #                           c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]

    train_idx, val_idx, test_idx, c_num_mat = split_imbalance(torch.tensor(label), train_ratio=4, val_ratio=2,
                                                                  test_ratio=4, imbalance_ratio=0.01)

    random.shuffle(train_idx)

    # ipdb.set_trace()

    idx_train = torch.LongTensor(train_idx)
    idx_val = torch.LongTensor(val_idx)
    idx_test = torch.LongTensor(test_idx)
    # c_num_mat = torch.LongTensor(c_num_mat)

    y = torch.tensor(label).to(torch.long)


    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(label)[1])
    # labels_a = torch.LongTensor(labels).to(CFG.finetune_device)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
    adj3 = sparse_mx_to_torch_sparse_tensor(adj3)

    label = torch.LongTensor(label)


    # return adj, features, labels, idx_train, idx_val, idx_test,idx_pair
    # return [adj,adj1,adj2,adj3], features,label, idx_train, idx_val, idx_test
    return [adj1, adj2, adj3], features, label, idx_train, idx_val, idx_test


def split_imbalance(labels,train_ratio,val_ratio,test_ratio,imbalance_ratio):

    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    label_max = int(max(labels.tolist())+1)
    minority_index = [item for item in range(label_max) if labels.tolist().count(item) <  len(labels.tolist())/num_classes]


    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        # if i in minority_index: c_num =  int(len(labels.tolist())/num_classes * imbalance_ratio)
        if num_classes > 2:
            if i in minority_index: c_num = int(len(labels.tolist()) / num_classes * imbalance_ratio)
        else:
            if i in minority_index: c_num = int((len(labels.tolist())- labels.tolist().count(i)) * imbalance_ratio)

        print('{:d}-th class sample number: {:d}'.format(i,c_num))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *train_ratio)
            c_num_mat[i,1] = int(c_num/10 * val_ratio)
            c_num_mat[i,2] = int(c_num/10 * test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    #c_num_mat = torch.LongTensor(c_num_mat)

    return train_idx, val_idx, test_idx, c_num_mat

def recon_upsample_degree(embed, labels, idx_train, k,smote_device, adj=None, portion=0.0, im_class_num=3,dynamic=1,dynamic_model=0):
    embed = embed.to(smote_device)
    labels = labels.to(smote_device)
    idx_train = idx_train.to(smote_device)
    adj = adj.to(smote_device).to_dense()
    # adj = adj.detach()

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None
    chosen_tail_list,tail_new_list = [],[]
    tail_idx_neighbor_list, tail_idx_neighbor_second_list = [], []
    first_neighbor_list, second_neighbor_list = [], []
    num_egdes = np.sum(adj.cpu().numpy(),axis=1)
    tail_nodes = np.where(num_egdes<=k)[0]
    head_nodes = np.where(num_egdes>k)[0]
    tail_up_count = 0
    labels_other = torch.tensor([-1 for i in range(embed.shape[0]-labels.shape[0])])
    labels = torch.cat((labels,labels_other),dim=0)
    adj_center = torch.zeros(im_class_num,adj.shape[0])
    class_newnode_count_dict = {}
    feature_centernode = torch.empty((im_class_num,embed.shape[1]))
    # labels_centernode = torch.tensor((im_class_num,1))
    # idx_centernode = torch.tensor((im_class_num, 1))
    class_newnode_dict = {}

    for i in range(im_class_num):
        tail_new_class_list = []
        class_newnode_count_dict[i] = 0
        tail_list,tail_idx_list = [],[]
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        feature_centernode[i] = torch.mean(embed[chosen, :], dim=0)  # generate the embedding of center nodes

        chosen_copy = chosen.clone()
        chosen_list = chosen.tolist()
        for k in range(len(chosen_list)):
            if chosen_list[k] in tail_nodes:
                tail_list.append(chosen_list[k])
                tail_idx_list.append(k)
        chosen_tail = torch.tensor(tail_list)
        chosen_tail_idx = torch.tensor(tail_idx_list)
        # chosen_tail = torch.tensor([i for i in chosen.tolist() if i in tail_nodes])
        if chosen_tail.shape[0] != 0:
            chosen = chosen_tail
            print("the label is {}".format(c_largest - i))
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(500 / chosen.shape[0])
            # c_portion = int(avg_number / chosen.shape[0])
            # c_portion = int(5000 / chosen.shape[0])
            num = c_portion
        else:
            c_portion = 1
        print("the run number for fake node ge neration is {}".format(num))
        for j in range(c_portion):
            # chosen = chosen[:num]  # chosen tail nodes to generate fake nodes
            chosen_embed_class = embed[chosen_copy,:]     # embed of chosen tail nodes
            distance = squareform(pdist(chosen_embed_class.cpu().detach()))   # calculate the distance among the same class
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)        # 130 get the index of min distance among the same class
            tail_idx_neighbor = idx_neighbor[chosen_tail_idx]     # 32 get the index of the tail nodes
            chosen_embed = embed[chosen_copy, :]  # 130 the embed of all nodes
            try:
                idx_neighbor_second = np.argsort(distance, axis=0)[1, :]   # 130 get the index of the second minimum distance
            except:
                idx_neighbor_second = idx_neighbor
            tail_idx_neighbor_second = idx_neighbor_second[chosen_tail_idx]  # 32 get the index of the tail nodes

            chosen_tail_list += chosen.tolist()
            first_neighbor_index = chosen_copy[tail_idx_neighbor].tolist()
            second_neighbor_index = chosen_copy[tail_idx_neighbor_second].tolist()
            first_neighbor_list += chosen_copy[tail_idx_neighbor].tolist()
            second_neighbor_list += chosen_copy[tail_idx_neighbor_second].tolist()
            # int(adj_new[i, chosen].shape[0]) * int(adj_new[i, chosen][0].tolist()) + int(adj_new[0].count()) * int(
            #     adj_new[i, chosen][0].tolist()) + int(adj_new[i, chosen].shape[0]) * int(adj_new[i, chosen][0].tolist())
           # chosen_embed = embed[chosen, :]
            # distance = squareform(pdist(chosen_embed.cpu().detach()))  # get the ecludiean distance matrix
            # np.fill_diagonal(distance, distance.max() + 100)  # set a large value for the diagnoal value

            # idx_neighbor = distance.argmin(axis=-1)  # sort distance in descending
            if dynamic == 0:
                interp_place_center = random.random()
                interp_place_first = random.random()
                interp_place_second = random.random()
                # new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
                # new_embed = embed[chosen, :] + feature_centernode[i] * interp_place_center + (chosen_embed[tail_idx_neighbor, :] - embed[chosen, :]) * interp_place_first + (chosen_embed[tail_idx_neighbor_second, :] - embed[chosen, :]) * interp_place_second
                new_embed = embed[chosen, :] + feature_centernode[i] * interp_place_center


            # elif dynamic_model!=0:
            else:
                # new_embed = dynamic_model(embed,chosen_embed,chosen,idx_neighbor,tail_up_count)
                new_embed = dynamic_model(embed, chosen_embed, chosen, tail_idx_neighbor, tail_idx_neighbor_second, tail_up_count)
                tail_idx_neighbor_list += tail_idx_neighbor.tolist()
                tail_idx_neighbor_second_list += tail_idx_neighbor_second.tolist()


            tail_up_count += len(chosen)

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            tail_new_list+= list(idx_new)
            tail_new_class_list += list(idx_new)
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[tail_idx_neighbor, :], min=0.0, max=1.0))

                    adj_new = adj.new(torch.clamp_( adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :], min=0.0, max=1.0))

                else:
                    # temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    # temp = adj.new(torch.clamp_(adj[chosen, :] + adj[tail_idx_neighbor, :], min=0.0, max=1.0))

                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                    # temp = adj.new(torch.clamp_(adj[chosen, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

                    # class_newnode_count_dict[i] += chosen.shape[0]
        class_newnode_dict[i] = tail_new_class_list   # 每一个标签新增的node的index
        adj_center[i, chosen] = 1


    labels_centernode = torch.tensor([c_largest - i for i in range(im_class_num)])
    idx_centernode = np.arange(embed.shape[0], embed.shape[0] + 3)

    embed = torch.cat((embed,feature_centernode),dim=0)
    labels = torch.cat((labels,labels_centernode),dim=0)
    idx_train = torch.cat((idx_train,idx_train.new(idx_centernode)))
    # 把那些新加点的边的 adjancent matrix 放在后面
    if adj is not None:

        adj_new = torch.cat((adj_new,adj_center))

        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        for k,v in class_newnode_dict.items():
            new_adj[-(3-k), v] = 1

            new_adj[v, -(3-k)] = 1

        # column_index = adj.shape[0]
        # for i in range(im_class_num):
        #     new_adj[-(c_largest - i), column_index:column_index + class_newnode_count_dict[i]] = 1
        #     column_index = column_index + class_newnode_count_dict[i]
        #
        # # new_adj = torch.tensor(normalize(new_adj + torch.eye(new_adj.shape[0]))).to_sparse()
        # new_adj = new_adj + torch.eye(new_adj.shape[0])


        # add_num = adj_new.shape[0]
        # new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        # new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        # new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        # new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]
        # column_index = adj.shape[0]+3
        # for i in range(im_class_num):
        #     new_adj[adj.shape[0]+i, column_index:column_index+class_newnode_count_dict[i]] = 1
        #     column_index = column_index+class_newnode_count_dict[i]
        #
        # new_adj = normalize(new_adj+torch.eye(new_adj.shape[0]))
        # new_adj = sparse_mx_to_torch_sparse_tensor(new_adj)
        print("the number of new nodes is {}".format(add_num))
        new_adj = new_adj.to_sparse()
        # new_adj = torch.tensor(normalize(new_adj + torch.eye(new_adj.shape[0]))).to_sparse()
        # new_adj = (new_adj + torch.eye(new_adj.shape[0])).to_sparse()
        return embed, labels.long(), idx_train, new_adj.detach(),tail_up_count,torch.tensor(chosen_tail_list).long(),torch.tensor(tail_new_list).long(),torch.tensor(first_neighbor_list),torch.tensor(second_neighbor_list)

    else:
        return embed, labels, idx_train


def recon_upsample_degrees(embed, labels, idx_train, k,smote_device,adj, im_class_num,n_node_degree=None, portion=0.0,dynamic=1,dynamic_model=0):
    embed = embed.to(smote_device)
    labels = labels.to(smote_device)
    idx_train = idx_train.to(smote_device)
    adj = adj.to(smote_device).to_dense()
    # adj = adj.detach()

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None
    chosen_tail_list,tail_new_list = [],[]
    tail_idx_neighbor_list, tail_idx_neighbor_second_list = [], []
    first_neighbor_list, second_neighbor_list = [], []
    num_egdes = np.sum(adj.cpu().numpy(),axis=1)
    tail_nodes = np.where(num_egdes<=k)[0]
    head_nodes = np.where(num_egdes>k)[0]
    tail_up_count = 0
    labels_other = torch.tensor([-1 for i in range(embed.shape[0]-labels.shape[0])])
    labels = torch.cat((labels,labels_other),dim=0)
    adj_center = torch.zeros(im_class_num,adj.shape[0])
    class_newnode_count_dict = {}
    feature_centernode = torch.empty((im_class_num,embed.shape[1]))
    # labels_centernode = torch.tensor((im_class_num,1))
    # idx_centernode = torch.tensor((im_class_num, 1))
    class_newnode_dict = {}
    n_tail_class = []  # the number of tail nodes in each class and the number of runs in each class
    for i in range(im_class_num):
        tail_new_class_list = []
        class_newnode_count_dict[i] = 0
        tail_list,tail_idx_list = [],[]
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        feature_centernode[i] = torch.mean(embed[chosen, :], dim=0)  # generate the embedding of center nodes

        chosen_copy = chosen.clone()
        chosen_list = chosen.tolist()
        for k in range(len(chosen_list)):
            if chosen_list[k] in tail_nodes:
                tail_list.append(chosen_list[k])
                tail_idx_list.append(k)
        chosen_tail = torch.tensor(tail_list)
        chosen_tail_idx = torch.tensor(tail_idx_list)
        # chosen_tail = torch.tensor([i for i in chosen.tolist() if i in tail_nodes])
        if chosen_tail.shape[0] != 0:
            chosen = chosen_tail
            print("the label is {}".format(c_largest - i))
            print("the number of tail nodes is {}".format(chosen.shape[0]))
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            # c_portion = int(500 / chosen.shape[0])
            c_portion = int(avg_number / chosen.shape[0])
            # c_portion = int(n_node_degree / chosen.shape[0])
            # c_portion = int(2000 / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1
        # c_portion = 20
        n_tail_class.append([chosen.shape[0],c_portion])
        print("the run number for fake node generation is {}".format(c_portion))
        for j in range(c_portion):
            # chosen = chosen[:num]  # chosen tail nodes to generate fake nodes
            chosen_embed_class = embed[chosen_copy,:]     # embed of chosen tail nodes
            distance = squareform(pdist(chosen_embed_class.cpu().detach()))   # calculate the distance among the same class
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)        # 130 get the index of min distance among the same class
            tail_idx_neighbor = idx_neighbor[chosen_tail_idx]     # 32 get the index of the tail nodes
            chosen_embed = embed[chosen_copy, :]  # 130 the embed of all nodes
            try:
                idx_neighbor_second = np.argsort(distance, axis=0)[1, :]   # 130 get the index of the second minimum distance
            except:
                idx_neighbor_second = idx_neighbor
            tail_idx_neighbor_second = idx_neighbor_second[chosen_tail_idx]  # 32 get the index of the tail nodes

            chosen_tail_list += chosen.tolist()
            first_neighbor_index = chosen_copy[tail_idx_neighbor].tolist()
            second_neighbor_index = chosen_copy[tail_idx_neighbor_second].tolist()
            first_neighbor_list += chosen_copy[tail_idx_neighbor].tolist()
            second_neighbor_list += chosen_copy[tail_idx_neighbor_second].tolist()
            # int(adj_new[i, chosen].shape[0]) * int(adj_new[i, chosen][0].tolist()) + int(adj_new[0].count()) * int(
            #     adj_new[i, chosen][0].tolist()) + int(adj_new[i, chosen].shape[0]) * int(adj_new[i, chosen][0].tolist())
           # chosen_embed = embed[chosen, :]
            # distance = squareform(pdist(chosen_embed.cpu().detach()))  # get the ecludiean distance matrix
            # np.fill_diagonal(distance, distance.max() + 100)  # set a large value for the diagnoal value

            # idx_neighbor = distance.argmin(axis=-1)  # sort distance in descending
            if dynamic == 0:
                interp_place_center = random.random()
                interp_place_first = random.random()
                interp_place_second = random.random()
                # new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
                # new_embed = embed[chosen, :] + feature_centernode[i] * interp_place_center + (chosen_embed[tail_idx_neighbor, :] - embed[chosen, :]) * interp_place_first + (chosen_embed[tail_idx_neighbor_second, :] - embed[chosen, :]) * interp_place_second
                new_embed = embed[chosen, :] + feature_centernode[i] * interp_place_center


            # elif dynamic_model!=0:
            else:
                # new_embed = dynamic_model(embed,chosen_embed,chosen,idx_neighbor,tail_up_count)
                new_embed = dynamic_model(embed, chosen_embed, chosen, tail_idx_neighbor, tail_idx_neighbor_second, tail_up_count)
                tail_idx_neighbor_list += tail_idx_neighbor.tolist()
                tail_idx_neighbor_second_list += tail_idx_neighbor_second.tolist()


            tail_up_count += len(chosen)

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            tail_new_list+= list(idx_new)
            tail_new_class_list += list(idx_new)
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[tail_idx_neighbor, :], min=0.0, max=1.0))

                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :], min=0.0, max=1.0))

                else:
                    # temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    # temp = adj.new(torch.clamp_(adj[chosen, :] + adj[tail_idx_neighbor, :], min=0.0, max=1.0))

                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                    # temp = adj.new(torch.clamp_(adj[chosen, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

                    # class_newnode_count_dict[i] += chosen.shape[0]
        class_newnode_dict[i] = tail_new_class_list
        adj_center[i, chosen] = 1


    labels_centernode = torch.tensor([c_largest - i for i in range(im_class_num)])
    idx_centernode = np.arange(embed.shape[0], embed.shape[0] + im_class_num)

    embed = torch.cat((embed,feature_centernode),dim=0)
    labels = torch.cat((labels,labels_centernode),dim=0)
    idx_train = torch.cat((idx_train,idx_train.new(idx_centernode)))
    # 把那些新加点的边的 adjancent matrix 放在后面
    if adj is not None:

        adj_new = torch.cat((adj_new,adj_center))

        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]   # original adjance matrix 左上

        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]    # new adjance matrix 左下

        # tail nodes connecting with new generated nodes (new adjance matrix) 右上
        # new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        for k,v in class_newnode_dict.items():
            new_adj[-(im_class_num-k), v] = 1     # 给新生成的点 与中心点连接 右下
            new_adj[v, -(im_class_num - k)] = 1   # 给新生成的点  与中心点连接 右上

        new_adj = new_adj.to_sparse()

        # column_index = adj.shape[0]
        # for i in range(im_class_num):
        #     new_adj[-(c_largest - i), column_index:column_index + class_newnode_count_dict[i]] = 1
        #     column_index = column_index + class_newnode_count_dict[i]
        #
        # # new_adj = torch.tensor(normalize(new_adj + torch.eye(new_adj.shape[0]))).to_sparse()
        # new_adj = new_adj + torch.eye(new_adj.shape[0])


        # add_num = adj_new.shape[0]
        # new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        # new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        # new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        # new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]
        # column_index = adj.shape[0]+3
        # for i in range(im_class_num):
        #     new_adj[adj.shape[0]+i, column_index:column_index+class_newnode_count_dict[i]] = 1
        #     column_index = column_index+class_newnode_count_dict[i]
        #
        # new_adj = normalize(new_adj+torch.eye(new_adj.shape[0]))
        # new_adj = sparse_mx_to_torch_sparse_tensor(new_adj)
        print("the number of new nodes is {}".format(add_num))
        return labels.long(), idx_train, new_adj.detach(),torch.tensor(chosen_tail_list).long(),torch.tensor(first_neighbor_list),torch.tensor(second_neighbor_list),n_tail_class

    else:
        return embed, labels, idx_train


def recon_upsample_degrees_dict(embed, labels, idx_train, k,smote_device,adj, im_class_num,n_node_degree=None, portion=0.0,dynamic=1,dynamic_model=0):
    embed = embed.to(smote_device)
    labels = labels.to(smote_device)
    idx_train = idx_train.to(smote_device)
    adj = adj.to(smote_device).to_dense()
    # adj = adj.detach()

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None
    chosen_tail_list,tail_new_list = [],[]
    tail_idx_neighbor_list, tail_idx_neighbor_second_list = [], []
    first_neighbor_list, second_neighbor_list = [], []
    num_egdes = np.sum(adj.cpu().numpy(),axis=1)
    tail_nodes = np.where(num_egdes<=k)[0]
    head_nodes = np.where(num_egdes>k)[0]
    tail_up_count = 0
    labels_other = torch.tensor([-1 for i in range(embed.shape[0]-labels.shape[0])])
    labels = torch.cat((labels,labels_other),dim=0)
    adj_center = torch.zeros(im_class_num,adj.shape[0])
    class_newnode_count_dict = {}
    feature_centernode = torch.empty((im_class_num,embed.shape[1]))
    # labels_centernode = torch.tensor((im_class_num,1))
    # idx_centernode = torch.tensor((im_class_num, 1))
    class_newnode_dict = {}
    n_tail_class = []  # the number of tail nodes in each class and the number of runs in each class
    tail_correspond_syn_array_dict = {}

    for i in range(im_class_num):
        tail_correspond_syn = []
        tail_new_class_list = []
        class_newnode_count_dict[i] = 0
        tail_list,tail_idx_list = [],[]
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        feature_centernode[i] = torch.mean(embed[chosen, :], dim=0)  # generate the embedding of center nodes

        chosen_copy = chosen.clone()
        chosen_list = chosen.tolist()
        for k in range(len(chosen_list)):
            if chosen_list[k] in tail_nodes:
                tail_list.append(chosen_list[k])
                tail_idx_list.append(k)
        chosen_tail = torch.tensor(tail_list)
        chosen_tail_idx = torch.tensor(tail_idx_list)
        # chosen_tail = torch.tensor([i for i in chosen.tolist() if i in tail_nodes])
        if chosen_tail.shape[0] != 0:
            chosen = chosen_tail
            print("the label is {}".format(c_largest - i))
            print("the number of tail nodes is {}".format(chosen.shape[0]))
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            # c_portion = int(500 / chosen.shape[0])
            c_portion = int(avg_number / chosen.shape[0] * 0.3)
            # c_portion = int(n_node_degree / chosen.shape[0])
            # c_portion = int(2000 / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1
        # c_portion = 20
        n_tail_class.append([chosen.shape[0],c_portion])
        print("the run number for fake node generation is {}".format(c_portion))
        for j in range(c_portion):
            # chosen = chosen[:num]  # chosen tail nodes to generate fake nodes
            chosen_embed_class = embed[chosen_copy,:]     # embed of nodes in this class
            distance = squareform(pdist(chosen_embed_class.cpu().detach()))   # calculate the distance among the same class
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)        # 130 get the index of min distance among the same class
            tail_idx_neighbor = idx_neighbor[chosen_tail_idx]     # 32 get the index of the tail nodes
            chosen_embed = embed[chosen_copy, :]  # 130 the embed of all nodes
            try:
                idx_neighbor_second = np.argsort(distance, axis=0)[1, :]   # 130 get the index of the second minimum distance
            except:
                idx_neighbor_second = idx_neighbor
            tail_idx_neighbor_second = idx_neighbor_second[chosen_tail_idx]  # 32 get the index of the tail nodes

            chosen_tail_list += chosen.tolist()
            first_neighbor_index = chosen_copy[tail_idx_neighbor].tolist()
            second_neighbor_index = chosen_copy[tail_idx_neighbor_second].tolist()
            first_neighbor_list += chosen_copy[tail_idx_neighbor].tolist()
            second_neighbor_list += chosen_copy[tail_idx_neighbor_second].tolist()
            # int(adj_new[i, chosen].shape[0]) * int(adj_new[i, chosen][0].tolist()) + int(adj_new[0].count()) * int(
            #     adj_new[i, chosen][0].tolist()) + int(adj_new[i, chosen].shape[0]) * int(adj_new[i, chosen][0].tolist())
           # chosen_embed = embed[chosen, :]
            # distance = squareform(pdist(chosen_embed.cpu().detach()))  # get the ecludiean distance matrix
            # np.fill_diagonal(distance, distance.max() + 100)  # set a large value for the diagnoal value

            # idx_neighbor = distance.argmin(axis=-1)  # sort distance in descending
            if dynamic == 0:
                interp_place_center = random.random()
                interp_place_first = random.random()
                interp_place_second = random.random()
                # new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
                # new_embed = embed[chosen, :] + feature_centernode[i] * interp_place_center + (chosen_embed[tail_idx_neighbor, :] - embed[chosen, :]) * interp_place_first + (chosen_embed[tail_idx_neighbor_second, :] - embed[chosen, :]) * interp_place_second
                new_embed = embed[chosen, :] + feature_centernode[i] * interp_place_center


            # elif dynamic_model!=0:
            else:
                # new_embed = dynamic_model(embed,chosen_embed,chosen,idx_neighbor,tail_up_count)
                new_embed = dynamic_model(embed, chosen_embed, chosen, tail_idx_neighbor, tail_idx_neighbor_second, tail_up_count)
                tail_idx_neighbor_list += tail_idx_neighbor.tolist()
                tail_idx_neighbor_second_list += tail_idx_neighbor_second.tolist()


            tail_up_count += len(chosen)

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            tail_new_list+= list(idx_new)
            tail_new_class_list += list(idx_new)
            idx_train_append = idx_train.new(idx_new)
            tail_correspond_syn.append(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[tail_idx_neighbor, :], min=0.0, max=1.0))

                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                    # adj_new = adj.new(torch.clamp_(adj[chosen, :], min=0.0, max=1.0))

                else:
                    # temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    # temp = adj.new(torch.clamp_(adj[chosen, :] + adj[tail_idx_neighbor, :], min=0.0, max=1.0))

                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                    # temp = adj.new(torch.clamp_(adj[chosen, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

                    # class_newnode_count_dict[i] += chosen.shape[0]
        class_newnode_dict[i] = tail_new_class_list
        adj_center[i, chosen] = 1

        tail_correspond_syn_array = np.array(tail_correspond_syn).T
        tail_correspond_syn_array = np.insert(tail_correspond_syn_array, 0, values=chosen.tolist(), axis=1)
        for i in range(chosen.shape[0]):
            tail_correspond_syn_array_dict[tail_correspond_syn_array[i, 0]] = tail_correspond_syn_array[i, 1:]

    labels_centernode = torch.tensor([c_largest - i for i in range(im_class_num)])
    idx_centernode = np.arange(embed.shape[0], embed.shape[0] + im_class_num)

    embed = torch.cat((embed,feature_centernode),dim=0)
    labels = torch.cat((labels,labels_centernode),dim=0)
    idx_train = torch.cat((idx_train,idx_train.new(idx_centernode)))
    # 把那些新加点的边的 adjancent matrix 放在后面
    if adj is not None:

        adj_new = torch.cat((adj_new,adj_center))

        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)

        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]   # original adjance matrix 左上

        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]    # new adjance matrix 左下

        # tail nodes connecting with new generated nodes (new adjance matrix) 右上
        # new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        for k,v in class_newnode_dict.items():
            new_adj[-(im_class_num-k), v] = 1     # 给新生成的点 与中心点连接 右下
            new_adj[v, -(im_class_num - k)] = 1   # 给新生成的点  与中心点连接 右上

        new_adj = new_adj.to_sparse()

        # column_index = adj.shape[0]
        # for i in range(im_class_num):
        #     new_adj[-(c_largest - i), column_index:column_index + class_newnode_count_dict[i]] = 1
        #     column_index = column_index + class_newnode_count_dict[i]
        #
        # # new_adj = torch.tensor(normalize(new_adj + torch.eye(new_adj.shape[0]))).to_sparse()
        # new_adj = new_adj + torch.eye(new_adj.shape[0])


        # add_num = adj_new.shape[0]
        # new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        # new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        # new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        # new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]
        # column_index = adj.shape[0]+3
        # for i in range(im_class_num):
        #     new_adj[adj.shape[0]+i, column_index:column_index+class_newnode_count_dict[i]] = 1
        #     column_index = column_index+class_newnode_count_dict[i]
        #
        # new_adj = normalize(new_adj+torch.eye(new_adj.shape[0]))
        # new_adj = sparse_mx_to_torch_sparse_tensor(new_adj)
        print("the number of new nodes is {}".format(add_num))
        return labels.long(), idx_train, new_adj.detach(),torch.tensor(chosen_tail_list).long(),torch.tensor(first_neighbor_list),torch.tensor(second_neighbor_list),n_tail_class,tail_correspond_syn_array_dict

    else:
        return embed, labels, idx_train

def upsample_nodes(embed, labels, idx_train, k,adj, im_class_num, portion=0.0):
    embed = embed
    labels = labels
    idx_train = idx_train
    adj = adj.to_dense()

    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    # ipdb.set_trace()
    adj_new = None
    chosen_tail_list,tail_new_list = [],[]
    first_neighbor_list = []
    num_egdes = np.sum(adj.cpu().numpy(),axis=1)
    tail_nodes = np.where(num_egdes<=k)[0]
    tail_up_count = 0
    labels_other = torch.tensor([-1 for i in range(embed.shape[0]-labels.shape[0])])
    labels = torch.cat((labels,labels_other),dim=0)
    adj_center = torch.zeros(im_class_num,adj.shape[0])
    class_newnode_count_dict = {}

    class_newnode_dict = {}
    n_tail_class = []  # the number of tail nodes in each class and the number of runs in each class
    tail_correspond_syn_array_dict = {}
    for i in range(im_class_num):
        tail_correspond_syn = []
        tail_new_class_list = []
        class_newnode_count_dict[i] = 0
        tail_list,tail_idx_list = [],[]
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        chosen_copy = chosen.clone()
        chosen_list = chosen.tolist()
        for k in range(len(chosen_list)):
            if chosen_list[k] in tail_nodes:
                tail_list.append(chosen_list[k])
                tail_idx_list.append(k)
        chosen_tail = torch.tensor(tail_list)
        chosen_tail_idx = torch.tensor(tail_idx_list)
        if chosen_tail.shape[0] != 0:
            chosen = chosen_tail
            print("the label is {}".format(c_largest - i))
            print("the number of tail nodes is {}".format(chosen.shape[0]))
        if portion == 0:
            # c_portion = int(500 / chosen.shape[0])
            c_portion = int(avg_number / chosen.shape[0])
            # c_portion = int(n_node_degree / chosen.shape[0])
            # c_portion = int(2000 / chosen.shape[0])
        else:
            c_portion = 1
        # c_portion = 20
        n_tail_class.append([chosen.shape[0],c_portion])
        print("the run number for fake node generation is {}".format(c_portion))
        for j in range(c_portion):
            # chosen = chosen[:num]  # chosen tail nodes to generate fake nodes
            chosen_embed_class = embed[chosen_copy,:]     # embed of chosen tail nodes
            distance = squareform(pdist(chosen_embed_class.cpu().detach()))   # calculate the distance among the same class
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)        # 130 get the index of min distance among the same class
            tail_idx_neighbor = idx_neighbor[chosen_tail_idx]     # 32 get the index of the tail nodes

            chosen_tail_list += chosen.tolist()
            first_neighbor_index = chosen_copy[tail_idx_neighbor].tolist()
            first_neighbor_list += chosen_copy[tail_idx_neighbor].tolist()

            tail_up_count += len(chosen)

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            tail_new_list+= list(idx_new)
            tail_new_class_list += list(idx_new)
            idx_train_append = idx_train.new(idx_new)
            tail_correspond_syn.append(idx_new)

            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[first_neighbor_index, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

        class_newnode_dict[i] = tail_new_class_list
        adj_center[i, chosen] = 1
        tail_correspond_syn_array = np.array(tail_correspond_syn).T
        tail_correspond_syn_array = np.insert(tail_correspond_syn_array,0,values=chosen.tolist(),axis=1)
        for i in range(chosen.shape[0]):
            tail_correspond_syn_array_dict[tail_correspond_syn_array[i,0]] = [tail_correspond_syn_array[i,1:]]


    labels_centernode = torch.tensor([c_largest - i for i in range(im_class_num)])
    idx_centernode = np.arange(embed.shape[0], embed.shape[0] + im_class_num)

    labels = torch.cat((labels,labels_centernode),dim=0)
    idx_train = torch.cat((idx_train,idx_train.new(idx_centernode)))
    # 把那些新加点的边的 adjancent matrix 放在后面
    if adj is not None:

        adj_new = torch.cat((adj_new,adj_center))

        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]   # original adjance matrix 左上

        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]    # new adjance matrix 左下

        # tail nodes connecting with new generated nodes (new adjance matrix) 右上
        # new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        for k,v in class_newnode_dict.items():
            new_adj[-(im_class_num-k), v] = 1     # 给新生成的点 与中心点连接 右下
            new_adj[v, -(im_class_num - k)] = 1   # 给新生成的点  与中心点连接 右上

        new_adj = new_adj.to_sparse()

        print("the number of new nodes is {}".format(add_num))
        return labels.long(), idx_train, new_adj.detach(),torch.tensor(chosen_tail_list).long(),torch.tensor(first_neighbor_list),n_tail_class


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def seed_torch(seed=1029):
    # ran_result.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

