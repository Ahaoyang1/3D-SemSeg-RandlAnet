from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))  # 获取当前文件所在目录的绝对路径
ROOT_DIR = dirname(BASE_DIR)  # 获取当前文件所在目录的上一级目录的绝对路径
sys.path.append(BASE_DIR)  # 将BASE_DIR添加到sys.path中，以便在import时能够找到对应的模块
sys.path.append(ROOT_DIR)  # 将ROOT_DIR添加到sys.path中，以便在import时能够找到对应的模块
from download.ply import write_ply  # 导入一个名为write_ply的函数，该函数定义在download目录下的ply.py中
# from download.tool import DataProcessing as DP
from utils.data_process import DataProcessing as DP
# 导入一个名为DataProcessing的类，该类定义在utils目录下的data_process.py中，用于数据处理相关的操作

dataset_path = '/lustre/home/jtyang1/WangZhengWen/Dataset'  # 数据集文件夹路径
anno_paths = [line.rstrip() for line in open(join(BASE_DIR, 'meta/anno_paths.txt'))]  # 从文件中读取注释文件的路径列表
anno_paths = [join(dataset_path, p) for p in anno_paths]  # 根据注释文件的路径列表构建完整路径

gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'meta/class_names.txt'))]  # 物体类别列表
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}  # 构建物体类别到标签的映射字典

sub_grid_size = 0.04  # 子网格大小
original_pc_folder = join(dirname(dataset_path), 'original_ply')  # 原始点云文件夹路径
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(sub_grid_size))  # 子采样后的点云文件夹路径
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None  # 如果原始点云文件夹不存在，则创建文件夹
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None  # 如果子采样后的点云文件夹不存在，则创建文件夹
out_format = '.ply'  # 输出文件格式


def convert_pc2ply(anno_path, save_path):
    """
    将原始的数据集文件转换为ply文件（每行为XYZRGBL）。
    我们将每个实例中的所有点聚合到一个房间中。
    :param anno_path: 注释文件的路径，例如 Area_1/office_2/Annotations/
    :param save_path: 保存原始点云的路径（每行为XYZRGBL）
    :return: None
    """
    data_list = []

    for f in glob.glob(join(anno_path, '*.txt')):
        # 获取类别名称
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:  # note: in some room there is 'staris' class..
            class_name = 'clutter'
        # 从文件中读取点云数据和标签
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        # 将点云数据和标签连接在一起，并添加到数据列表中
        data_list.append(np.concatenate([pc, labels], 1))  # Nx7

    # 将数据列表中的数据连接在一起，形成完整的点云和标签矩阵
    pc_label = np.concatenate(data_list, 0)
    # 减去点云数据的最小坐标值，使得点云的坐标范围从0开始
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:6].astype(np.uint8)
    labels = pc_label[:, 6].astype(np.uint8)
    # 将点云数据、颜色和标签保存为ply文件
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # 保存子采样后的点云和KD树文件
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # 构建并保存KD树
    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    # 根据KD树获取点云数据在子采样点云中的投影索引，并保存为文件
    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


if __name__ == '__main__':
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    # 遍历注释文件路径列表
    for annotation_path in anno_paths:
        print(annotation_path)
        # 提取路径中的元素
        elements = str(annotation_path).split('/')
        # 构建输出文件名
        out_file_name = elements[-3] + '_' + elements[-2] + out_format
        # 调用convert_pc2ply函数将注释文件转换为ply文件，并保存到原始点云文件夹中
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name))