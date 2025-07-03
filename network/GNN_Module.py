import os
import sys
import glob
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import network.pytorch_utils as pt_utils


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))
        # self.conv.append(Conv1D_noRelu(channels[-2], channels[-1], ksize))
    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))
        # self.conv.append(Conv2D_noRelu(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
    # gather the coordinates or features of neighboring points
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
    return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class Gnn(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Gnn, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)
        self.att_pooling_1 = Att_pooling(emb_dims,emb_dims)

    def forward(self, x,idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        # print(batch_idx,batch_idx.shape)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)  # [B, C, N, k]
        # center =x.unsqueeze(-1).repeat(1,1,1,idx.size(-1))
        # print(center.shape)
        x = nn_feat - x.unsqueeze(-1)  # [B, c, N, k] - [B, c, N, 1] = [B, c, N, k]
        # x = x.permute(0, 3, 1, 2)
        # print(x.shape,center.shape)
        # x=torch.concat([center,x],dim=1)
        x = self.conv2d(x)  # [B, emb_dims, N, k]

        # x = x.max(-1)[0]  # [B, C, N] //attention
        x=self.att_pooling_1(x).squeeze(-1)

        x = self.conv1d(x)
        return x



class GNN(nn.Module):
    def __init__(self, in_dim,dim):
        super(GNN, self).__init__()
        # in_dim=dim[0]
        # out_dim=dim[-1]
        self.propogate1 = Gnn(in_dim, dim[0])
        self.propogate2 = Gnn(dim[0], dim[1])  # Conv2DBNReLU(3,64)->Conv2DBNReLU(64,64)->Conv1DBNReLU(64,64)
        # self.propogate3 = Gnn(dim[1], dim[1])  ###

        self.shortcut = Gnn(in_dim, dim[1])

        # # self.shortcut=pt_utils.Conv2d(in_dim, dim[-1], kernel_size=(1, 1), bn=True, activation=None)
        # self.bn=nn.BatchNorm1d(dim[-1])
        # self.propogate4 = Propagate(64, 64)
        # self.propogate5 = Propagate(64, emb_dims)

    def forward(self, x,nn_idx):
        # [B, 3, N]
        # nn_idx = knn(x, k=12)  # [B, N, k], 最近邻索引
        x1 = self.propogate1(x, nn_idx)
        x2 = self.propogate2(x1, nn_idx)
        # x3 = self.propogate3(x2, nn_idx)

        shortcut=self.shortcut(x,nn_idx)
        return F.leaky_relu(x2+shortcut,negative_slope=0.2)

        # return x1

        # x2=self.bn(x2)           #加上bn?
        #
        # shortcut = self.shortcut(x, nn_idx)
        # # x = self.propogate4(x, nn_idx)
        # # x = self.propogate5(x, nn_idx)  # [B, emb_dims, N]
        # return F.leaky_relu(x2+shortcut,negative_slope=0.2)



# class GNN3(nn.Module):
#     def __init__(self, dim,mlp_list):
#         super(GNN3, self).__init__()
#
#         self.mlp1=pt_utils.Conv2d(dim, mlp_list[0], kernel_size=(1, 1), bn=True)
#         self.mlp2=pt_utils.Conv2d(mlp_list[0], mlp_list[0], kernel_size=(1, 1), bn=True,activation=None)
#         self.mlp3 = pt_utils.Conv2d(mlp_list[0], mlp_list[1], kernel_size=(1, 1), bn=True)
#         self.mlp4 = pt_utils.Conv2d(mlp_list[1], mlp_list[1], kernel_size=(1, 1), bn=True,activation=None)
#         self.mlp5 = pt_utils.Conv2d(mlp_list[1], mlp_list[2], kernel_size=(1, 1), bn=True)
#         self.mlp6 = pt_utils.Conv2d(mlp_list[2], mlp_list[2], kernel_size=(1, 1), bn=True,activation=None)
#
#
#         self.shortcut1=pt_utils.Conv2d(dim, mlp_list[0], kernel_size=(1, 1), bn=True,activation=None)
#         self.shortcut2=pt_utils.Conv2d(mlp_list[0], mlp_list[1], kernel_size=(1, 1), bn=True,activation=None)
#         self.shortcut3=pt_utils.Conv2d(mlp_list[1], mlp_list[2], kernel_size=(1, 1), bn=True,activation=None)
#
#     def forward(self,x,idx):
#
#         print(x.shape)
#         x1=self.mlp1(x)
#         print(x1.shape)
#         x2=self.mlp2(x1)
#         shortcut1=self.shortcut1(x)
#         x2=F.leaky_relu(x2+shortcut1,negative_slope=0.2)
#
#         x3=self.mlp3(x2)
#         x4=self.mlp4(x3)
#         shortcut2=self.shortcut2(x2)
#         x4=F.leaky_relu(x4+shortcut2)
#
#         x5=self.mlp5(x4)
#         x6=self.mlp6(x5)
#         shortcut3=self.shortcut3(x4)
#
#         return F.leaky_relu(x6+shortcut3,negative_slope=0.2)
#
# class Multi_MLP(nn.Module):
#     def __init__(self,indim,outdim):
#         super(Multi_MLP, self).__init__()
#         self.mlp1 = pt_utils.Conv2d(indim, outdim, kernel_size=(1, 1), bn=True)
#         self.mlp2 = pt_utils.Conv2d(outdim, outdim, kernel_size=(1, 1), bn=True, activation=None)
#         self.shortcut=pt_utils.Conv2d(outdim, outdim, kernel_size=(1, 1), bn=True, activation=None)
#
#     def forward(self,x):
#         x1=self.mlp1(x)
#         x2=self.mlp2(x1)
#         shortcut=self.shortcut(x)
#         return F.leaky_relu(x2+shortcut,negative_slope=0.2)

