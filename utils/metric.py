import numpy as np
import threading
from sklearn.metrics import confusion_matrix

# 定义 IoUCalculator 类
class IoUCalculator:
    def __init__(self, cfg):
        # 初始化各个类别数量为 0
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg
        self.lock = threading.Lock()  # 创建一个锁，用于线程同步

    # 添加数据的方法
    def add_data(self, end_points):
        # 获取预测结果和标签
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]  # 获取预测标签（取 logits 中最大值对应的索引）
        pred_valid = pred.detach().cpu().numpy()  # 转换为 numpy 数组
        labels_valid = labels.detach().cpu().numpy()  # 转换为 numpy 数组

        # 累积计算正确预测的数量
        val_total_correct = 0
        val_total_seen = 0
        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))

        # 使用锁保证多线程操作的互斥
        self.lock.acquire()
        self.gt_classes += np.sum(conf_matrix, axis=1)  # 计算每个类别的真实样本数量之和
        self.positive_classes += np.sum(conf_matrix, axis=0)  # 计算每个类别被预测为正样本的数量之和
        self.true_positive_classes += np.diagonal(conf_matrix)  # 计算每个类别真正被预测为正样本的数量
        self.lock.release()

    # 计算 IoU
    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                # 计算每个类别的 IoU
                iou = self.true_positive_classes[n] / \
                    float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        # 计算平均 IoU
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list

# 计算准确率的函数
def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]  # 获取预测标签（取 logits 中最大值对应的索引）
    acc = (logits == labels).sum().float() / float(labels.shape[0])  # 计算准确率
    end_points['acc'] = acc  # 将准确率存储在 end_points 字典中
    return acc, end_points
