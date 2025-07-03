import random
import open3d
import colorsys
import numpy as np


color={
    '0':[0,255,0],
    '1': [0, 0, 255],
    '2': [0, 255, 255],
    # '3': [254, 255, 0],
    # '4': [255, 0, 255],
    # '5': [100, 100, 255],
    # # '5': [255, 0, 0],
    # '6': [50, 50, 50],
    # # '8': [90, 30, 150],
    # '7': [200, 200, 100],
    # '8': [170, 121, 200],
    # '9': [255, 0, 0],
    # '10': [200, 100, 100],
    # '11': [10, 200, 100],
    # '12': [200, 200, 200],
}

color_list=[
    [0,0,0],
    [0, 255, 0],
    [0, 0, 255],
    # [150, 60, 30],
    # [180, 30, 80],
    # [255, 0, 0],
    # [30, 30, 255],
    # [200, 40, 255],
    # [90, 30, 150],
    # [255, 0, 255],
    # [255, 150, 255],
    # [75, 0, 75],
    # [75, 0, 175],
    # [0, 200, 255],
    # [50, 120, 255],
    # [0, 175, 0],
    # [0, 60, 135],
    # [80, 240, 150],
    # [150, 240, 255],
    # [0, 0, 255]
]

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):  # 生成随机颜色，其中N为需要的类别数
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        # print(colors)
        random.seed(seed)
        # random.shuffle(colors)
        return colors

    # @staticmethod
    # def draw_pc(pc_xyzrgb):  # 输入数据格式为x-y-z-r-g-b六维
    #     pc = open3d.geometry.PointCloud()
    #     pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
    #     if pc_xyzrgb.shape[1] == 3:
    #         open3d.visualization.draw_geometries([pc])
    #         return 0
    #     if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
    #         pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
    #     else:
    #         pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
    #     open3d.visualization.draw_geometries([pc])
    #
    #     return 0
    #
    # @staticmethod
    # def draw_pc_4_class(pc_xyz, label, plot_colors = None):
    #     pc_sem_ins = np.empty(label.shape[0], dtype=label.dtype)
    #     idx_00 = np.where(label==0)
    #     idx_01 = np.where(label==1)
    #     idx_02 = np.where((label==2) | (label==3) | (label==4) | (label==5) | (label==6) | (label==11))
    #     idx_03 = np.where((label==7) | (label==8) | (label==9) | (label==10) | (label==12))
    #     pc_sem_ins[idx_00] = 0
    #     pc_sem_ins[idx_01] = 1
    #     pc_sem_ins[idx_02] = 2
    #     pc_sem_ins[idx_03] = 3
    #
    #     # idx = np.where(pc_sem_ins == 2)
    #     # pc_sem_ins = pc_sem_ins[idx]
    #     # pc_xyz = pc_xyz[idx]
    #
    #     ins_colors = []
    #     sem_ins_labels = np.unique(pc_sem_ins)
    #     for i in sem_ins_labels:
    #         # print()
    #         ins_colors.append(color[str(i)])
    #
    #     lst = sem_ins_labels.tolist()
    #     # idx=lst.index(8)
    #     # print(idx)
    #     # print(sem_ins_labels)
    #     sem_ins_bbox = []
    #     Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
    #     for id, semins in enumerate(sem_ins_labels):
    #         valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
    #         if semins <= -1:
    #             tp = [0, 0, 0]
    #         else:
    #             if plot_colors is not None:
    #                 tp = ins_colors[semins]
    #             else:
    #                 tp = ins_colors[id]
    #
    #         Y_colors[valid_ind] = tp
    #         # if id==idx:
    #         #     Y_colors[valid_ind]=[0.05,0.05,0.05]
    #         # print(id,tp)
    #
    #         ### bbox
    #         valid_xyz = pc_xyz[valid_ind]
    #         xmin = np.min(valid_xyz[:, 0]);
    #         xmax = np.max(valid_xyz[:, 0])
    #         ymin = np.min(valid_xyz[:, 1]);
    #         ymax = np.max(valid_xyz[:, 1])
    #         zmin = np.min(valid_xyz[:, 2]);
    #         zmax = np.max(valid_xyz[:, 2])
    #         sem_ins_bbox.append(
    #             [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])
    #
    #     Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
    #     Plot.draw_pc(Y_semins)


    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins,
                        plot_colors=None):  # pc_xyz是坐标信息 pc_sem_ins是label plot_colors是用每类的颜色[r, g, b]/255
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        # ins_colors=color_list


        ##############################
        ins_colors=[]
        sem_ins_labels = np.unique(pc_sem_ins)
        for i in sem_ins_labels:
            # print()
            ins_colors.append(color[str(i)])

        lst=sem_ins_labels.tolist()
        # idx=lst.index(8)
        # print(idx)
        # print(sem_ins_labels)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp
            # if id==idx:
            #     Y_colors[valid_ind]=[0.05,0.05,0.05]
            # print(id,tp)

            ### bbox
            valid_xyz = pc_xyz[valid_ind]
            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)

        return Y_semins

