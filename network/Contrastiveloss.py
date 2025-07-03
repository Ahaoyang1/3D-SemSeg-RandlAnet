import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import network.pytorch_utils as pt_utils
from network.GNN_Module import GNN
from network.loss_func import compute_loss
from network.storeArray import Store
import time

class Contrastive:
    def __init__(self,
                 clustering_items_per_class,
                 clustering_start_iter,
                 clustering_update_mu_iter,
                 clustering_momentum,
                 feature_store_update_iter_iteraval,
                 margin,
                 loss_weight_clustering,
                 train_dataset,
                 criterion,
                 enable_clustering,
                 device,
                 start_sample_weight_iter,
                 ):

        self.clustering_items_per_class = clustering_items_per_class
        self.clustering_start_iter = clustering_start_iter
        self.clustering_update_mu_iter = clustering_update_mu_iter
        self.clustering_momentum = clustering_momentum

        self.num_labels = 19
        self.clustering_items_per_class = clustering_items_per_class
        self.feature_store_update_iter_iteraval = feature_store_update_iter_iteraval
        self.margin = margin
        self.hingeloss = nn.HingeEmbeddingLoss(self.margin)
        self.means = [None for _ in range(19)]
        self.feature_store_is_stored = False
        self.feature_store = Store(self.num_labels, self.clustering_items_per_class)
        self.loss_weight_clustering = loss_weight_clustering
        self.loss_weight = {"loss_cls": 1.0, "loss_clustering": loss_weight_clustering}
        self.enable_clustering = enable_clustering
        self.device=device

        self.sample_weight_iter = start_sample_weight_iter

        self.sample_weight = torch.ones(19, dtype=torch.float)

        self.feat_store_path = os.path.join(os.path.abspath('.'), 'feature_store7')
        if not os.path.exists(self.feat_store_path):
            os.mkdir(self.feat_store_path)
        self.feature_store_is_stored = False
        self.feature_store_save_loc = os.path.join(self.feat_store_path, 'feat.pt')
        self.centroids_store_save_loc = os.path.join(self.feat_store_path, 'centroids.pt')


    def getloss(self, cls_loss, targets, iter,input_features=None):

        losses = {}
        losses["loss_cls"] = cls_loss

        if input_features is not None:
            losses["loss_clustering"] = self.get_clustering_loss(input_features, targets,iter)
        # loss_weight = default=1.0
        # return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        # print(losses['loss_cls'],losses['loss_clustering'])
        total_loss=losses["loss_cls"]*self.loss_weight['loss_cls']+losses["loss_clustering"]*self.loss_weight['loss_clustering']
        return total_loss

    def get_save_name(self, iter):
        tmp_path = os.path.join(self.feat_store_path, str(iter)+'feat.pt')
        return tmp_path

    def update_feature_store(self, features, gt_classes, iter):
        self.feature_store.add(features, gt_classes)


    def clstr_loss_l2_cdist(self, input_features, gt_classes):
        """
        Get the foreground input_features, generate distributions for the class,
        get probability of each feature from each distribution;
        Compute loss: if belonging to a class -> likelihood should be higher
                      else -> lower
        :param input_features:
        :param proposals:
        :return:
        """
        all_means = self.means
        for item in all_means:
            if item!=None:
                length = item.shape
                break

        # print(length)
        for i, item in enumerate(all_means):
            # if len(item) == 0:
            if item==None:
                all_means[i] = torch.zeros(length)
        # print(input_features.device)
        # print(torch.stack(all_means).to(self.device).device)
        distances = torch.cdist(input_features.to(self.device), torch.stack(all_means).to(self.device), p=2.0)

        labels = -1 * torch.ones((input_features.shape[0],self.num_labels))
        for i in range(self.num_labels):
            idx = torch.where(gt_classes ==  i)[0]
            labels[idx, i] = 1

        loss = self.hingeloss(distances, labels.to(self.device))
        return loss

    def get_clustering_loss(self,input_features, targets, iter):
        if not self.enable_clustering:
            return 0

        c_loss = 0

        # print("iteration={}".format(storage.iter))
        if iter == self.clustering_start_iter:
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if item.shape[0] == 0:
                    self.means[index] = None
                else:
                    item = item.data
                    mu = item.mean(dim=0)
                    self.means[index] = mu
            c_loss = self.clstr_loss_l2_cdist(input_features, targets)

        elif iter > self.clustering_start_iter:
            if iter % self.clustering_update_mu_iter == 0:
                # Compute new MUs
                items = self.feature_store.retrieve(-1)
                new_means = [None for _ in range(self.num_labels + 1)]
                for index, item in enumerate(items):
                    if item.shape[0]==0:
                        new_means[index] = None
                    else:
                        # print(item)
                        item = item.data
                        # print(item)
                        new_means[index] = item.mean(dim=0)
                # Update the MUs
                if iter > self.sample_weight_iter:
                    dist_total = []
                    flag = False
                    for i, mean in enumerate(self.means):
                        if mean != None and new_means[i] != None:
                            dist_per_class = torch.dist(mean, new_means[i], p=2)
                            dist_total.append(dist_per_class)
                            self.means[i] = self.clustering_momentum * mean + (1 - self.clustering_momentum) * new_means[i]
                        else:
                            flag = True
                    if flag == False:
                        self.sample_weight = torch.tensor(dist_total)
                    else:
                        self.sample_weight = torch.ones(19, dtype=torch.float)
                else:
                    for i, mean in enumerate(self.means):
                        if mean!=None and new_means[i] !=None:
                            self.means[i] = self.clustering_momentum * mean + (1 - self.clustering_momentum) * new_means[i]
                        # print(self.means[i].shape, new_means[i].shape)
                # print('finish update mean vector')
            c_loss = self.clstr_loss_l2_cdist(input_features, targets)
        self.update_feature_store(input_features, targets, iter)

        return c_loss