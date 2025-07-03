from utils.data_process import DataProcessing as DP
from utils.config import ConfigSemanticKITTI as cfg
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch
# from distance import Distance
import math
import time
from sklearn.neighbors import KDTree
from visual import Plot

def spatially_regular_gen1( item, data_list):
    # Generator loop
    cloud_ind = item
    # print(item)
    pc_path = data_list[cloud_ind]

    pc, tree, labels = get_data(pc_path)
    return pc,tree,labels

def crop_pc(points, labels, search_tree, pick_idx):
    # crop a fixed size point cloud for training
    center_point = points[pick_idx, :].reshape(1, -1)
    select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
    # print("center_point:",center_point.shape,"selected_idx:",select_idx.shape)

    select_idx = DP.shuffle_idx(select_idx)  ###########################################################

    select_points = points[select_idx]
    select_labels = labels[select_idx]
    return select_points, select_labels, select_idx

def get_data( file_path):
    # print(file_path)
    seq_id = file_path[0]
    frame_id = file_path[1]

    #########

    kd_tree_path = join(dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
    # read pkl with search tree
    with open(kd_tree_path, 'rb') as f:
        search_tree = pickle.load(f)
    points = np.array(search_tree.data, copy=False)
    # load labels
    label_path = join(dataset_path, seq_id, 'labels', frame_id + '.npy')
    labels = np.squeeze(np.load(label_path))
    return points, search_tree, labels


def get_data1(seq_id,frame_id):
    kd_tree_path = join(dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
    # read pkl with search tree
    with open(kd_tree_path, 'rb') as f:
        search_tree = pickle.load(f)
    points = np.array(search_tree.data, copy=False)
    # load labels
    label_path = join(dataset_path, seq_id, 'labels', frame_id + '.npy')
    labels = np.squeeze(np.load(label_path))
    return points, search_tree, labels


def save(id,new_xyz,new_label,str):
    s = '%06d' % id
    xyz_path = join(dataset_path, str, 'velodyne', s+'.npy')
    label_path = join(dataset_path, str, 'labels', s+'.npy')
    np.save(xyz_path, new_xyz)
    np.save(label_path, new_label)
    search_tree = KDTree(new_xyz)
    KDTree_save = join(dataset_path, str, 'KDTree', s+'.pkl')
    with open(KDTree_save, 'wb') as f:
        pickle.dump(search_tree, f)

def judge(xyz):
    for i in range(xyz.shape[0]):
        x=xyz[i][0]
        y=xyz[i][1]
        z=xyz[i][2]
        if x>=x_min and x<=x_max and y>=y_min and y<=y_max and z>=z_min and z<=z_max:
            return False
    return True

dataset_path = '/media/jtyang/Backup2/xzwang/semanticiTTi/data_odometry_velodyne/dataset/sequences'

seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
# seq_list=['08'] #269*3
# seq_list=['00', '01', '02', '03', '04', '05', '06', '07']
data_list = DP.get_file_list(dataset_path, seq_list)

# print(data_list)
data_list=sorted(data_list)
# print(data_list)
id=0
cnt=0

logg=open("/media/jtyang/Backup2/xzwang/data.txt",'w+')

# xyz1, tree1, label1 = get_data1('01', '000359')

xyz1, tree1, label1 = get_data1('02', '001000')

# Plot().draw_pc_sem_ins(xyz1, label1, plot_colors=None)
# xyz1, tree1, label1 = get_data1('02', '000990')
# Plot().draw_pc_sem_ins(xyz1, label1, plot_colors=None)


print(xyz1.shape)

idx = np.where(label1 == 8)
moter_xyz = xyz1[idx]
moter_label = label1[idx]
x_max=0
x_min=999
y_max=0
y_min=999
z_max=0
z_min=999
for i in range(moter_xyz.shape[0]):
    x_max=max(x_max,moter_xyz[i][0])
    x_min=min(x_min,moter_xyz[i][0])
    y_max=max(y_max,moter_xyz[i][1])
    y_min=min(y_min,moter_xyz[i][1])
    z_max=max(z_max,moter_xyz[i][2])
    z_min=min(z_min,moter_xyz[i][2])
print(x_max,x_min,y_max,y_min,z_max,z_min)

size90=int(moter_xyz.shape[0]*0.9)
size80=int(moter_xyz.shape[0]*0.8)
size70=int(moter_xyz.shape[0]*0.7)
size60=int(moter_xyz.shape[0]*0.6)
print(size90)

moter_xyz90=moter_xyz[0:size90,:]
moter_label90=moter_label[0:size90]

moter_xyz80=moter_xyz[0:size80,:]
moter_label80=moter_label[0:size80]

# moter_xyz70=moter_xyz[0:size70,:]
# moter_label70=moter_label[0:size70]
#
# moter_xyz60=moter_xyz[0:size60,:]
# moter_label60=moter_label[0:size60]

is_jump=0
seq=0
res=0

for i in range(0,len(data_list),2):

    if data_list[i][0]!='00':
        break

    if is_jump>0:
        is_jump=is_jump-1
        xyz, tree, label = spatially_regular_gen1(i, data_list)
        save(id, xyz, label, str='000')
        save(id, xyz, label, str='000_80')
        save(id, xyz, label, str='000_90')
        print('跳过的已保存',id)
        id = id + 1
        continue

    start=time.time()
    xyz,tree,label = spatially_regular_gen1(i, data_list)


    if judge(xyz)==False:
        save(id,xyz,label,str='000')
        save(id, xyz, label, str='000_80')
        save(id, xyz, label, str='000_90')
        print('不符合的已保存', id)
        id=id+1
        if seq>0:
            seq=seq-1
            print('连续帧中存在不符合的',seq)
        # save(id, xyz, label, str='000')
        continue

    if seq>0:
        new_xyz = np.concatenate([xyz, moter_xyz], axis=0)
        new_label = np.concatenate([label, moter_label], axis=0)
        save(id, new_xyz, new_label, str='000')
        # id=id+1

        new_xyz = np.concatenate([xyz, moter_xyz90], axis=0)
        new_label = np.concatenate([label, moter_label90], axis=0)
        save(id, new_xyz, new_label, str='000_90')
        # id=id+1

        new_xyz = np.concatenate([xyz, moter_xyz80], axis=0)
        new_label = np.concatenate([label, moter_label80], axis=0)
        save(id, new_xyz, new_label, str='000_80')
        seq=seq-1
        print('已保存', id,seq)
        id=id+1
        res=res+1
        continue

    new_xyz=np.concatenate([xyz,moter_xyz],axis=0)
    new_label=np.concatenate([label,moter_label],axis=0)
    Plot().draw_pc_sem_ins(new_xyz, new_label, plot_colors=None)


    # search_tree = KDTree(new_xyz)
    # # KDTree_save = join(dataset_path, str, 'KDTree', s + '.pkl')
    # select_points, select_labels, select_idx=crop_pc(new_xyz,new_label,search_tree,1356)
    # Plot().draw_pc_sem_ins(select_points, select_labels, plot_colors=None)


    print(data_list[i],i/2)

    is_save=input("0:不保存，1:保存，2:跳过10帧，3:跳过20帧，4:跳过40帧，5:跳过100帧")

    if is_save=='2':
        save(id, xyz, label, str='000')
        save(id, xyz, label, str='000_80')
        save(id, xyz, label, str='000_90')
        print('已保存跳过的', id)
        id = id + 1
        is_jump=10
        print("跳过10帧")
    elif is_save=='3':
        is_jump=20
        save(id, xyz, label, str='000')
        save(id, xyz, label, str='000_80')
        save(id, xyz, label, str='000_90')
        print('已保存跳过的', id)
        id = id + 1
        print("跳过20帧")
    elif is_save=='4':
        is_jump=40
        save(id, xyz, label, str='000')
        save(id, xyz, label, str='000_80')
        save(id, xyz, label, str='000_90')
        print('已保存跳过的', id)
        id = id + 1
        print("跳过40帧")
    elif is_save=='5':
        is_jump=100
        save(id, xyz, label, str='000')
        save(id, xyz, label, str='000_80')
        save(id, xyz, label, str='000_90')
        print('已保存跳过的', id)
        id = id + 1
        print("跳过100帧")
    elif is_save=='1':
        n=input("连续多少帧")
        seq=int(n)
        save(id,new_xyz,new_label,str='000')
        # id=id+1

        new_xyz=np.concatenate([xyz,moter_xyz90],axis=0)
        new_label=np.concatenate([label,moter_label90],axis=0)
        save(id,new_xyz,new_label,str='000_90')
        # id=id+1

        new_xyz = np.concatenate([xyz, moter_xyz80], axis=0)
        new_label = np.concatenate([label, moter_label80], axis=0)
        save(id, new_xyz, new_label,str='000_80')
        # id=id+1

        # new_xyz = np.concatenate([xyz, moter_xyz70], axis=0)
        # new_label = np.concatenate([label, moter_label70], axis=0)
        # save(id, new_xyz, new_label,str='004')
        # # id=id+1
        #
        # new_xyz = np.concatenate([xyz, moter_xyz60], axis=0)
        # new_label = np.concatenate([label, moter_label60], axis=0)
        # save(id, new_xyz, new_label,str='005')

        print('已保存添加的', id)
        id = id + 1
        res=res+1
    else:
        save(id, new_xyz, new_label, str='000')
        # id=id+1

        new_xyz = np.concatenate([xyz, moter_xyz90], axis=0)
        new_label = np.concatenate([label, moter_label90], axis=0)
        save(id, new_xyz, new_label, str='000_90')
        # id=id+1

        new_xyz = np.concatenate([xyz, moter_xyz80], axis=0)
        new_label = np.concatenate([label, moter_label80], axis=0)
        save(id, new_xyz, new_label, str='000_80')
        print('已保存当前帧', id)
        res=res+1
        id = id + 1

    print("当前已改变%d帧，共%d帧" %(res,id))

    # break

'''
    # print(xyz.shape,label.shape)
    f1= 8 in label
    num=(label==8).sum()
    # print(num)
    # print(xyz.shape)
    # print(i, cnt)
    if f1==True: #1000
        cnt=cnt+1
        print(num,data_list[i],file=logg)
        # print(i)
        # print(i, cnt)
        # break'''










