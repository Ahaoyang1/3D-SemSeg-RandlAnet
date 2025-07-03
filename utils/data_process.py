from os.path import join
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import utils.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class DataProcessing:

    # @staticmethod
    # def load_pc_kitti(pc_path):
    #     scan = np.fromfile(pc_path, dtype=np.float32)
    #     scan = scan.reshape((-1, 4))
    #     points = scan[:, 0:3]  # get xyz
    #     return points

    @staticmethod
    def load_pc_kitti(pc_path):

        scan = np.fromfile(pc_path, dtype=np.float32)

        scan = scan.reshape((-1, 4))

        points = scan[:, 0:3]
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):

        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF
        inst_label = label >> 16
        assert ((sem_label + (inst_label << 16) == label).all())

        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def get_file_list(dataset_path, seq_list):

        data_list = []
        for seq_id in seq_list:

            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')

            new_data = [(seq_id, f[:-4]) for f in np.sort(os.listdir(pc_path))]

            data_list.extend(new_data)

        return data_list

    def get_active_list(list_root):

        train_list = []
        pool_list = []

        with open(join(list_root, 'label_data.json')) as f:
            train_list = json.load(f)

        with open(join(list_root, 'ulabel_data.json')) as f:
            pool_list = json.load(f)

        pool_list += train_list
        train_list = []
        return train_list, pool_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending on the input
        """


        points = np.asarray(points, dtype=np.float32)


        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            features = np.asarray(features, dtype=np.float32) if features is not None else None
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            labels = np.asarray(labels, dtype=np.int32)
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            features = np.asarray(features, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int32)
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """


        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)


        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)


        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(data_root, paths, num_of_class):
        num_per_class = [0 for _ in range(num_of_class)]

        for file_path in tqdm(paths, total=len(paths)):
            label_path = join(data_root, file_path[0], 'labels', file_path[1] + '.npy')
            label = np.load(label_path).reshape(-1)
            inds, counts = np.unique(label, return_counts=True)
            for i, c in zip(inds, counts):
                if i - 1 >= num_of_class:
                    continue
                num_per_class[i - 1] += c



        num_per_class = np.array(num_per_class)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)

