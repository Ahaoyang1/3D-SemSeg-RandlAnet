import torch
import torch.nn as nn
import torch.nn.functional as F
import network.pytorch_utils as pt_utils
from network.GNN_Module import GNN
from network.loss_func import compute_loss
from network.store import Store
from network.gcn import SelfAttention, MultiHeadedAttention

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)
        self.self_attention_layer = SelfAttention(feature_dim=512)

        self.dilated_res_blocks = nn.ModuleList()
        self.dilated_res_blocks.append(Dilated_res_block(8, 16, [8, 16], 10))
        self.dilated_res_blocks.append(Dilated_res_block(32, 64, [32, 64], 20))
        self.dilated_res_blocks.append(Dilated_res_block(128, 128, [64, 128], 30))
        self.dilated_res_blocks.append(Dilated_res_block(256, 256, [128, 256], 40))

        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-4]
                d_out = 2 * self.config.d_out[-4]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.embedding_layer = pt_utils.Conv2d(32, config.embedding_size, kernel_size=(1, 1), bn=True)
        self.fc3 = pt_utils.Conv2d(32, config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, end_points):
        features = end_points['features']
        coords = end_points['xyz'][-1]
        coords = coords.transpose(2, 1)

        features = self.fc0(features)
        features = features.unsqueeze(dim=3)

        f_encoder_list = []
        encode_list = []
        for i in range(self.config.num_layers):
            f_encoder_i, xyz_encode = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i], encode_list)
            encode_list.append(xyz_encode)
            for j in range(len(encode_list)):
                encode_list[j] = self.random_sample(encode_list[j], end_points['sub_idx'][i])
            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        features = f_encoder_list[-1].squeeze(dim=3)
        features = self.self_attention_layer(coords, features)
        features = features.unsqueeze(dim=3)

        features = self.decoder_0(features)

        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))
            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        features = self.fc1(f_decoder_list[-1])
        features = self.fc2(features)

        logits = self.fc3(features)
        logits = logits.squeeze(3)

        embeddings = self.dropout(features)
        embeddings = self.embedding_layer(embeddings)
        embeddings = embeddings.squeeze(3)

        end_points['logits'] = logits
        end_points['embeddings'] = embeddings

        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        feature = feature.squeeze(dim=3)
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        feature = feature.squeeze(dim=3)
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)
        return interpolated_features


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out, mlp_lst, encode_dim):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out, mlp_lst, encode_dim)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx, encode_list):
        f_pc = self.mlp1(feature)
        f_pc, xyz_encode = self.lfa(xyz, f_pc, neigh_idx, encode_list)
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2), xyz_encode


class Building_block(nn.Module):
    def __init__(self, d_out, mlp_lst, encode_dim):
        super(Building_block, self).__init__()
        self.gnn = GNN(encode_dim + d_out // 2, mlp_lst)
        self.att_pooling_3 = Att_pooling(10, 10)

    def forward(self, xyz, feature, neigh_idx, encode_list):
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.att_pooling_3(f_xyz)
        f_xyz = f_xyz.squeeze(-1)
        rt_xyzencode = f_xyz
        feature = feature.squeeze(-1)
        if len(encode_list) != 0:
            last_encode = encode_list[0]
            for ii in range(1, len(encode_list)):
                last_encode = torch.cat([last_encode, encode_list[ii]], dim=1)
            last_encode = last_encode.squeeze(-1)
            f_concat = torch.cat([f_xyz, last_encode, feature], dim=1)
        else:
            f_concat = torch.cat([f_xyz, feature], dim=1)

        f_pc_agg = self.gnn(f_concat, neigh_idx)
        return f_pc_agg.unsqueeze(-1), rt_xyzencode.unsqueeze(-1)

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
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
