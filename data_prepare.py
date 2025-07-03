import os
import yaml
import pickle
import argparse
import numpy as np
from os.path import join, exists
from sklearn.neighbors import KDTree
from utils.data_process import DataProcessing as DP


def write_log(message, file_path):
    """将消息写入指定的日志文件."""
    with open(file_path, 'a') as f:
        f.write(message + '\n')
    print(message)

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', default='/lustre/home/jtyang1/yangah/Data/tomato2',
                    help='source dataset path [default: None]')
parser.add_argument('--dst_path', default='/lustre/home/jtyang1/yangah/Data/tomato2',
                    help='destination dataset path [default: None]')
parser.add_argument('--grid_size', type=float, default=0.01, help='Subsample Grid Size [default: 0.01]')
parser.add_argument('--yaml_config', default='utils/semantic-kitti.yaml', help='semantic-kitti.yaml path')
FLAGS = parser.parse_args()

data_config = FLAGS.yaml_config
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

grid_size = FLAGS.grid_size
dataset_path = FLAGS.src_path
output_path = FLAGS.dst_path
seq_list = np.sort(os.listdir(dataset_path))
# seq_list=['00_0']

for seq_id in seq_list:
    log_file_path = join(output_path, seq_id, 'process_log.txt')  # 定义日志文件路径

    if seq_id == '0':
        continue

    write_log('sequence ' + seq_id + ' start', log_file_path)

    seq_path = join(dataset_path, seq_id)
    seq_path_out = join(output_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    pc_path_out = join(seq_path_out, 'velodyne')
    KDTree_path_out = join(seq_path_out, 'KDTree')


    os.makedirs(seq_path_out) if not exists(seq_path_out) else None
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None
    os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None

    if int(seq_id) < 11:
        label_path = join(seq_path, 'labels')
        label_path_out = join(seq_path_out, 'labels')
        os.makedirs(label_path_out) if not exists(label_path_out) else None

        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            points_with_labels = np.loadtxt(join(pc_path, scan_id))  # Load points and labels from text file
            points = points_with_labels[:, :3]  # Extract only the xyz coordinates
            labels = points_with_labels[:, -1].astype(np.int32)  # Extract labels from the last column

            sub_points, sub_labels = DP.grid_sub_sampling(points, labels=labels, grid_size=grid_size)
            write_log(f'Point cloud {scan_id} loaded and subsampled with shape: {sub_points.shape}', log_file_path)

            search_tree = KDTree(sub_points)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')

            np.save(join(pc_path_out, scan_id)[:-4], sub_points)
            np.save(join(label_path_out, scan_id)[:-4], sub_labels)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            write_log(f'KDTree for {scan_id} saved', log_file_path)

            if seq_id == '08':
                proj_path = join(seq_path_out, 'proj')
                os.makedirs(proj_path) if not exists(proj_path) else None
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_inds], f)
                write_log(f'Projection indices for {scan_id} saved', log_file_path)
    else:
        proj_path = join(seq_path_out, 'proj')
        os.makedirs(proj_path) if not exists(proj_path) else None
        scan_list = np.sort(os.listdir(pc_path))
        for scan_id in scan_list:
            points_with_labels = np.loadtxt(join(pc_path, scan_id))  # Load points from text file
            points = points_with_labels[:, :3]  # Extract only the xyz coordinates
            sub_points = DP.grid_sub_sampling(points, grid_size=0.01)
            search_tree = KDTree(sub_points)
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
            proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
            np.save(join(pc_path_out, scan_id)[:-4], sub_points)
            with open(KDTree_save, 'wb') as f:
                pickle.dump(search_tree, f)
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_inds], f)
            write_log(f'Projection indices for {scan_id} saved', log_file_path)

    write_log(f'Sequence {seq_id} processed successfully', log_file_path)