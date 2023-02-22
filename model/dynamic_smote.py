#!/usr/bin/env python
# encoding: utf-8


from model import  DynamicSmote,SemanticAggregate,InterCL1,IntraCL1
from pygcn.models import GCN,GAT,SAGE,Classifier,SAGE_1
import torch
import torch.nn as nn
import numpy as np


class EdgeGenerator(nn.Module):

    def __init__(self, tail_corr_syn_dict_lists,args,threshold):

        super(EdgeGenerator, self).__init__()

        syn_corr_tail_list = []
        for i in range(len(tail_corr_syn_dict_lists)):
            syn_corr_tail = []
            for k,v in tail_corr_syn_dict_lists[i].items():
                for syn in v:
                    syn_corr_tail.append([syn,k])
            syn_corr_tail_list.append(syn_corr_tail)
        self.weight = [nn.Parameter(torch.FloatTensor(len(syn_corr_tail_list[i]),1)).to(args.device) for i in range(len(syn_corr_tail_list))]
        self.syn_corr_tail_list = [np.array(syn_corr_tail_list[i]) for i in range(len(syn_corr_tail_list))]
        self.threshold = threshold
        self.args = args

    def reset_parameters(self):
        for i in range(len(self.weight)):
            nn.init.xavier_uniform_(self.weight[i])

    def forward(self, embed_list):
        edge_score_list = []
        for i in range(len(self.syn_corr_tail_list)):
            edge_score = torch.sigmoid(torch.log(embed_list[i][self.syn_corr_tail_list[i][:,0]] @ embed_list[i][self.syn_corr_tail_list[i][:,1]].T @ self.weight[i])+1e-5)
            # edge_score = embed_list[i][0]@ embed_list[i][1]@ self.weight[i]
            edge_score_list.append(edge_score)

        return edge_score_list


    def get_new_adj(self,embed_list,adj_list):

        adj_new_list = []
        self.edge_score_list = self.forward(embed_list)

        for i in range(len(self.edge_score_list)):
            count = 0
            adj_new =adj_list[i].to_dense().cpu()
            for j in range(self.edge_score_list[i].shape[0]):
                if self.edge_score_list[i][j][0] >= self.threshold:
                    row = self.syn_corr_tail_list[i][j][0]
                    col = self.syn_corr_tail_list[i][j][1]
                    adj_new[row,col] = 1
                    count +=1
                    # adj_new[col,row] = 1

            adj_new_list.append(adj_new.to_sparse().to(self.args.device))

            print("{} new edges are generated for relation {}".format(count,i+1))

        return adj_new_list



class dSmote(nn.Module):

    def __init__(self, fea_dim, hid_dim, n_class, dropout, encoder, args,tail_corr_syn_dict_lists):

        super(dSmote, self).__init__()

        self.dynamicsmote = DynamicSmote(fea_dim, hid_dim)
        if args.data_name == 'twitter':
            self.encoder = GCN(nfeat=fea_dim, nhid=hid_dim, dropout=dropout)
        if args.data_name == 'yelp':
            self.encoder = SAGE_1(nfeat=fea_dim, nhid=hid_dim, dropout=dropout)

        self.edgegen = EdgeGenerator(tail_corr_syn_dict_lists,args,threshold=0.7)


    def forward(self, feature, adj_new_list, labels, chosen_tail_lists, first_neighbor_lists, second_neighbor_lists,
                center_dict_lists):

        embed_ds = []
        for i in range(len(adj_new_list)):
            features_ds = self.dynamicsmote(feature, labels, chosen_tail_lists[i], first_neighbor_lists[i],
                                            second_neighbor_lists[i], center_dict_lists[i])
            embed_ds.append(self.encoder(features_ds, adj_new_list[i]))

        adj_new_list = self.edgegen.get_new_adj(embed_ds,adj_new_list)

        return embed_ds,adj_new_list

class SmoteHG(nn.Module):

    def __init__(self,fea_dim,hid_dim,n_class,dropout,encoder,args):

        super(SmoteHG,self).__init__()

        self.dynamicsmote = DynamicSmote(fea_dim,hid_dim)
        if args.data_name == 'twitter':
            self.encoder = GCN(nfeat=fea_dim,nhid=hid_dim,dropout=dropout)
        if args.data_name == 'yelp':
            self.encoder = SAGE_1(nfeat=fea_dim, nhid=hid_dim, dropout=dropout)



    def forward(self,feature,adj_new_list,labels,chosen_tail_lists,first_neighbor_lists, second_neighbor_lists, center_dict_lists):

        embed_ds = []
        for i in range(len(adj_new_list)):
            features_ds = self.dynamicsmote(feature,labels,chosen_tail_lists[i],first_neighbor_lists[i],second_neighbor_lists[i],center_dict_lists[i])
            embed_ds.append(self.encoder(features_ds,adj_new_list[i]))
            # outputs.append(self.classifier(embed_ds))

        return embed_ds


    # def loss(self,):


class AggHG(nn.Module):
     def __init__(self,fea_dim,hid_dim,att_dim,n_class,n_nodes,dropout,encoder,chosen_tail_lists,tail_corr_syn_dict_lists,idx_train_new_list,n_tail_class,n_im_class,batch,device,temp_inter,temp_intra,args):
         super(AggHG,self).__init__()

         self.hid_dim = hid_dim
         self.n_nodes = n_nodes
         self.smotehg = SmoteHG(fea_dim,hid_dim,n_class,dropout,encoder,args)
         self.fc = nn.Linear(hid_dim,n_class)
         self.semanticaggregator = SemanticAggregate(inp_dim=hid_dim,att_dim=att_dim)

         self.device = device
         self.criterion = nn.CrossEntropyLoss()

         self.intercl = InterCL1(chosen_tail_lists,n_nodes,batch,temp_inter,args)
         self.intracl = IntraCL1(idx_train_new_list,n_tail_class,n_nodes,chosen_tail_lists,tail_corr_syn_dict_lists,n_im_class,temp_intra,args)

     def forward(self,feature,adj_new_list,labels,chosen_tail_lists,first_neighbor_lists, second_neighbor_lists, center_dict_lists):

        embed_ds_list = self.smotehg(feature,adj_new_list,labels,chosen_tail_lists,first_neighbor_lists, second_neighbor_lists, center_dict_lists)

        embeds = embed_ds_list[0][:self.n_nodes].reshape(self.n_nodes,1,-1).to(self.device)
        for i in range(1,len(embed_ds_list)):
            embeds = torch.cat((embeds, embed_ds_list[i][:self.n_nodes].reshape(self.n_nodes,1,-1)), dim=1)
        output = self.semanticaggregator(embeds)


        return embed_ds_list,output


     def loss(self,feature,adj_new_list,labels,idx_train,labels_new_lists,idx_train_new_list,chosen_tail_lists,first_neighbor_lists, second_neighbor_lists, center_dict_lists,logits_augment):

        loss = []
        output_list = []
        embeds_ds_list,embeds = self.forward(feature,adj_new_list,labels,chosen_tail_lists,first_neighbor_lists, second_neighbor_lists, center_dict_lists)

        output_task = self.fc(embeds) + logits_augment

        for i in range(len(embeds_ds_list)):

            outputs = self.fc(embeds_ds_list[i]) + logits_augment
            output_list.append(outputs)
            loss.append(self.criterion(outputs[idx_train_new_list[i]],labels_new_lists[i][idx_train_new_list[i]]))

        # nn.CrossEntropyLoss()
        output_list.append(output_task)
        loss_task = self.criterion(output_task[idx_train], labels[idx_train])

        inter_loss = self.intercl(embeds_ds_list)
        intra_loss = self.intracl(embeds_ds_list)

        loss_final = loss_task + torch.sum(torch.stack(loss)) + (inter_loss + intra_loss)

        return loss_final,output_list