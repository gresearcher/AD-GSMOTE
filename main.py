#!/usr/bin/env python
# encoding: utf-8

import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from model.utils import get_wrong_index,get_performance, recon_upsample_degrees_dict, seed_torch, load_twitter_data
from sklearn.metrics import classification_report,confusion_matrix
from model.dynamic_smote import AggHG
from model.logit_adjustment import get_augmentation
import datetime
import os
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.001,
#                     help='Initial learning rate.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--im_class_num', type=int, default=3,
                    help='# of minority classes')
parser.add_argument('--model', type=str, default='gcn',
                    help='gcn,sage,gat,gin')
parser.add_argument('--nhid', type=int, default=100)
parser.add_argument('--up_scale', type=float, default=0)
parser.add_argument('--rec_weight', type=float, default=0.000001)
parser.add_argument('--smote_device', type=float, default= torch.device(torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")))
parser.add_argument('--device', type=float, default= torch.device(torch.device("cuda:1" if  torch.cuda.is_available() else "cpu")))
parser.add_argument('--model_design', type=str, default='designed',
                    help='baseline,designed')
parser.add_argument('--data_name', type=str, default='twitter',
                    help='twitter,yelp')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

seed_torch(seed=5)
best_val_f1 = 0.0
best_epoch = 0
best_val_gmean = 0.0

tro = 1.2
temp_inter = 10.0
temp_intra = 18.0
att_dim = 64
batch_num = 5000
dropout = args.dropout


seed_list = [5,10,20,25,30]
f1_list,auc_list = [],[]
gmean_list = []

for run in range(5):
# for run in range(5):
    best_val_f1 = 0.0
    best_epoch = 0
    best_val_gmean = 0.0

    seed_torch(seed_list[run])

    # Load data
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
    dir_saver = r'D:\Twitter\saved_models\\' + timestamp
    path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, 'multi_smote_logit'))

    adjs, features,labels, idx_train, idx_val, idx_test = load_twitter_data()

    logits_augment = get_augmentation(labels[idx_train], tro=tro).to(args.device)
    degree_list = [10, 10, 10]

    center_dict_lists = [[{},{},{},{}],[{},{},{},{}],[{},{},{},{}]]
    adj_up_list, label_new_list, idx_train_new_list,chosen_tail_lists, first_neighbor_lists, second_neighbor_lists,n_tail_class_lists,tail_correspond_syn_array_dict_lists = [],[],[],[],[],[],[],[]

    for i in range(len(adjs)):

        labels_new, idx_train_new, adj_up, chosen_tail_list, first_neighbor_list, second_neighbor_list,n_tail_class,tail_correspond_syn_array_dict = recon_upsample_degrees_dict(
            features, labels,
            idx_train, k=degree_list[i], smote_device=args.smote_device,n_node_degree=300,
            adj=adjs[i],
            portion=args.up_scale, im_class_num=args.im_class_num, dynamic=0, dynamic_model=0)

        label_new_list.append(labels_new)
        idx_train_new_list.append(idx_train_new)
        adj_up_list.append(adj_up)

        chosen_tail_lists.append(chosen_tail_list)
        first_neighbor_lists.append(first_neighbor_list)
        second_neighbor_lists.append(second_neighbor_list)
        n_tail_class_lists.append(n_tail_class)
        tail_correspond_syn_array_dict_lists.append(tail_correspond_syn_array_dict)

    model = AggHG(fea_dim = features.shape[1],hid_dim=args.hidden,temp_inter = temp_inter, temp_intra=temp_intra,n_class=labels.max().item() + 1,att_dim=att_dim,n_nodes= features.shape[0],dropout=args.dropout,encoder=args.model,chosen_tail_lists=chosen_tail_lists,idx_train_new_list = idx_train_new_list,tail_corr_syn_dict_lists = tail_correspond_syn_array_dict_lists,n_tail_class = n_tail_class_lists,n_im_class = args.im_class_num,batch = batch_num,device=args.device,args=args)


    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    train_correct_indexs_list = [torch.empty((1, 0)),torch.empty((1, 0)),torch.empty((1, 0)),torch.empty((1, 0))]

    if args.cuda:
        model.to(args.device)
        features = features.to(args.device)
        # adjs = [adj.to(args.device) for adj in adjs]
        adj_up_list = [adj_up.to(args.device) for adj_up in adj_up_list]

        labels = labels.to(args.device)
        idx_train = idx_train.to(args.device)
        idx_val = idx_val.to(args.device)
        idx_test = idx_test.to(args.device)

        # logit_adjustment_list  = [logit_adjustment.to(args.device) for logit_adjustment in logit_adjustment_list]
        label_new_list = [label_new.to(args.device) for label_new in label_new_list]
        idx_train_new_list = [idx_train_new.to(args.device) for idx_train_new in idx_train_new_list]
        chosen_tail_lists = [chosen_tail.to(args.device) for chosen_tail in chosen_tail_lists]
        first_neighbor_lists = [first_neighbor.to(args.device) for first_neighbor in first_neighbor_lists]
        second_neighbor_lists = [second_neighbor.to(args.device) for second_neighbor in second_neighbor_lists]
        train_correct_indexs_list = [train_correct_indexs.to(args.device) for train_correct_indexs in train_correct_indexs_list]
        # tail_correspond_syn_array_dict_lists = [tail_correspond_syn_array_dict.to(args.device) for tail_correspond_syn_array_dict in tail_correspond_syn_array_dict_lists]
        n_tail_class_list = []
        for n_tail_class in n_tail_class_lists:
            sub_ = []
            for item in n_tail_class:
                sub_.append([torch.tensor(j).to(args.device) for j in item])
            n_tail_class_list.append(sub_)

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        loss_train,outputs =  model.loss(features,adj_up_list,labels,idx_train,label_new_list,idx_train_new_list,chosen_tail_lists,first_neighbor_lists,second_neighbor_lists,center_dict_lists,logits_augment)
        try:
            train_acc, train_auc, train_f1,train_gmean, train_corrects = get_performance(outputs[-1][idx_train], labels[idx_train],
                                                        pre='train')
        except ValueError:
            print('Input contains NaN')
            pass

        for i in range(len(adj_up_list)):

            train_acc, train_auc, train_f1, train_gmean, train_corrects = get_performance(
                outputs[i][idx_train_new_list[i]],
                label_new_list[i][idx_train_new_list[i]],
                pre='adj_{}'.format(i))
            train_correct_indexs_list[i] = torch.cat(
                (train_correct_indexs_list[i],
                 idx_train_new_list[i][torch.where(train_corrects == 1.0)[0].reshape(1, -1)]), dim=1)

            for j in range(1, args.im_class_num + 1):
                center_dict_lists[i][j] = \
                    train_correct_indexs_list[i][train_correct_indexs_list[i] < features.shape[0]].reshape(-1, 1)[
                        label_new_list[i][train_correct_indexs_list[i][
                            train_correct_indexs_list[i] < features.shape[0]].long()] == j].reshape(1,
                                                                                                    -1)

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            #

            model.eval()

            loss_val, outputs = model.loss(features, adj_up_list, labels, idx_train, label_new_list, idx_train_new_list,
                                       chosen_tail_lists, first_neighbor_lists, second_neighbor_lists,
                                       center_dict_lists, logits_augment)
        try:
            val_acc, val_auc, val_f1,val_gmean,_ = get_performance(outputs[-1][idx_val], labels[idx_val], pre='val')

            global best_val_f1
            global best_epoch
            global best_val_gmean
            if best_val_f1 < val_f1:

                best_val_gmean = val_gmean
                best_val_f1 = val_f1
                best_epoch = epoch
                print(' Saving model ...')
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                torch.save(model.state_dict(), path_saver)

            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  # 'loss_rec: {:.4f}'.format(loss_rec.item()),
                  # 'loss_total: {:.4f}'.format(loss_total.item()),

                  'acc_train: {:.4f}'.format(train_acc.item()),
                  'F1_train: {:.4f}'.format(train_f1.item()),
                  'auc_train: {:.4f}'.format(train_auc),
                  'gmean_train: {:.4f}'.format(train_gmean),

                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(val_acc.item()),
                  'F1_val: {:.4f}'.format(val_f1.item()),
                  'auc_val: {:.4f}'.format(val_auc),
                  'gmean_val: {:.4f}'.format(val_gmean),
                  'best_f1_val: {:.4f}'.format(best_val_f1),
                  'best_gmean_val: {:.4f}'.format(best_val_gmean),
                  'time: {:.4f}s'.format(time.time() - t))


            log_dict = {"epoch":epoch+1,"acc_train score": train_acc.item(), "acc_val score": val_acc.item(),"f1_train score": train_f1.item(), "f1_val score": val_f1.item(),"auc_train score": train_auc, "auc_val score": val_auc,"gmean_train score": train_gmean, "gmean_val score": val_gmean,"loss_train score": loss_train.item(), "loss_val score": loss_val.item()
             }

        except ValueError:
            print('Input contains NaN')
            pass


    def tes():

        print("Load model from epoch {}".format(best_epoch))
        print("Model path: {}".format(path_saver))


        model.load_state_dict(torch.load(path_saver))
        model.eval()

        # loss_test, outputs = model.loss(features, adj_up_list, labels, idx_train, label_new_list, idx_train_new_list,
        #                                chosen_tail_lists, first_neighbor_lists, second_neighbor_lists,
        #                                center_dict_lists,logit_adjustment_list)
        loss_test, outputs = model.loss(features, adj_up_list, labels, idx_train, label_new_list, idx_train_new_list,
                                       chosen_tail_lists, first_neighbor_lists, second_neighbor_lists,
                                       center_dict_lists,logits_augment)

        test_acc, test_auc, test_f1,test_gmean,_ = get_performance(outputs[-1][idx_test], labels[idx_test], pre='test')
        wrong_index, wrong_index_label = get_wrong_index(outputs[-1][idx_test], labels[idx_test], pre='test')
        wrong_index_df = pd.DataFrame((wrong_index,wrong_index_label))
        wrong_index_df.to_csv(r'./data/wrong_index_df.csv')
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(test_acc.item()),
              "F1= {:.4f}".format(test_f1.item()),
              'auc_test: {:.4f}'.format(test_auc),
              'gmean_test: {:.4f}'.format(test_gmean)
              )

        print("confusion matrix:")
        print(confusion_matrix(y_true=labels[idx_test].cpu().detach(),y_pred=torch.argmax(outputs[-1][idx_test], dim=-1).cpu().detach()))
        print("classification report:")
        print(classification_report(y_true=labels[idx_test].cpu().detach(),y_pred=torch.argmax(outputs[-1][idx_test],dim=-1).cpu().detach(),target_names=['negative','seller','user','related']))

        # with torch.no_grad():
            # np.savetxt("m.txt",output.cpu(),fmt="%s",delimiter=",")
            # np.savetxt( r"E:\code\Muco\data\for embedding\gcn_embedding.txt",output.cpu(),fmt="%s",delimiter=",")
            # np.savetxt(r"D:\Muco\binary_gcn_embedding.txt", output.cpu(), fmt="%s", delimiter=",")
        # print(output)

        auc_list.append(test_auc)
        f1_list.append(test_f1)
        gmean_list.append(test_gmean)

    # Train model
    t_total = time.time()
    # for run in range(20):


    for epoch in range(args.epochs):
        # if args.model_design == 'baseline':
        #     train_correct_index = None
        # train_correct_index1,train_correct_index2,train_correct_index = train(epoch,train_correct_index1,train_correct_index2,train_correct_index)
        train(epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    tes()

auc_ave = np.mean(auc_list)
auc_std = np.std(auc_list)

f1_ave = np.mean(f1_list)
f1_std = np.std(f1_list)

gmean_ave = np.mean(gmean_list)
gmean_std = np.std(gmean_list)

print(
    "test_auc_ave: %.4f, test_auc_std: %.4f, test_f1_ave: %.4f, test_f1_std: %.4f, test_gmean_ave: %.4f, test_gmean_std: %.4f, " % (
    auc_ave, auc_std, f1_ave, f1_std, gmean_ave, gmean_std))

end = time.time()
period = (end - start)
print(period)