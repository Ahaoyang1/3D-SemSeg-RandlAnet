from utils.data_process import DataProcessing as DP
from utils.config import ConfigDs as cfg
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch


class SemanticKITTI(torch_data.Dataset):
    def __init__(self, mode, data_list=None):
        self.name = 'Ds'
        self.dataset_path = '/lustre/home/jtyang1/yangah/Data/tomato2'
        self.num_classes = cfg.num_classes
        self.ignored_labels = np.sort([0])
        self.mode = mode

        if data_list is None:
            if mode == 'training':
                seq_list = ['00_0']
            elif mode == 'validation':
                seq_list = ['08']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        else:
            self.data_list = data_list

        self.data_list = sorted(self.data_list)

    def get_class_weight(self):
        return DP.get_class_weights(self.dataset_path, self.data_list, self.num_classes)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.data_list)
        num_points = selected_labels.shape[0]

        masks = np.zeros((num_points, 2), dtype=np.float32)
        masks[np.arange(num_points), selected_labels - 1] = 1

        size = np.unique(selected_labels).size
        return selected_pc, selected_labels, selected_idx, cloud_ind, masks, size

    def spatially_regular_gen(self, item, data_list):
        cloud_ind = item
        pc_path = data_list[cloud_ind]
        pc, tree, labels = self.get_data(pc_path)
        pick_idx = np.random.choice(len(pc), 1)
        selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)
        return selected_pc, selected_labels, selected_idx, np.array([cloud_ind], dtype=np.int32)

    def get_data(self, file_path):
        seq_id = file_path[0]
        frame_id = file_path[1]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
        labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels

    @staticmethod
    def crop_pc(points, labels, search_tree, pick_idx):
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        select_idx = DP.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        return features

    def index_points(self, points, idx):
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = np.arange(B).reshape(view_shape).repeat(repeat_shape[0], axis=0).repeat(repeat_shape[1], axis=1)
        new_points = points[batch_indices, idx, :]
        return new_points

    def idw(self, xyz, num, neigh_idx):
        xyz = torch.from_numpy(xyz)
        neigh_idx = torch.from_numpy(neigh_idx)
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz
        dist_mean = torch.mean(relative_xyz, dim=2)
        xyz_norm = torch.sqrt(torch.sum(torch.pow(dist_mean, 2), dim=-1))
        idx = torch.multinomial(1 / xyz_norm, num)
        return idx.numpy()

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []



        for i in range(cfg.num_layers):

            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            if i == 3:
                sub_points = batch_pc
                pool_i = neighbour_idx
            else:
                idx = self.idw(batch_pc, batch_pc.shape[1] // cfg.sub_sampling_ratio[i], neighbour_idx)
                sub_points = self.index_points(batch_pc, idx)
                pool_i = self.index_points(neighbour_idx, idx)

            # sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            # pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]

            up_i = DP.knn_search(sub_points, batch_pc, 1)

            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)

            batch_pc = sub_points


        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):

        selected_pc, selected_labels, selected_idx, cloud_ind, masks, sizes = [], [], [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])
            masks.append(batch[i][4])
            sizes.append(batch[i][5])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)
        masks = np.stack(masks)
        sizes = np.array(sizes)

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        # inputs['sub_idx'][-1]=inputs['sub_idx'][-2]
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        # inputs['sub_idx'][-1]=inputs['sub_idx'][-2]
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['masks'] = torch.from_numpy(masks).float()
        inputs['sizes'] = torch.from_numpy(sizes).long()

        return inputs



