#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import numpy as np  # 导入 numpy 库


class iouEval:  # 创建一个名为 iouEval 的类
    def __init__(self, n_classes, ignore=None):
        # 类别
        self.n_classes = n_classes  # 将输入的参数 n_classes 赋值给 self.n_classes

        # 从均值中包含和忽略的内容
        self.ignore = np.array(ignore, dtype=np.int64)  # 将 ignore 转换为 numpy 数组，并将其赋值给 self.ignore
        # 使用列表推导式创建一个包含在 n_classes 范围内但不包含在 ignore 中的数组，并将其赋值给 self.include
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
        print("[IOU EVAL] IGNORE: ", self.ignore)  # 打印输出 self.ignore
        print("[IOU EVAL] INCLUDE: ", self.include)  # 打印输出 self.include

        # 重置类别计数器
        self.reset()  # 调用 reset() 方法，将混淆矩阵重置为全零

    def num_classes(self):
        return self.n_classes  # 返回类别数量

    def reset(self):
        # 使用全零数组创建一个大小为 (n_classes, n_classes) 的混淆矩阵，并将其赋值给 self.conf_matrix
        self.conf_matrix = np.zeros((self.n_classes,
                                    self.n_classes),
                                    dtype=np.int64)

    def addBatch(self, x, y):  # x=preds, y=targets (添加批次)
        # 大小应该匹配
        x_row = x.reshape(-1)  # 将 x 转换为一维数组
        y_row = y.reshape(-1)  # 将 y 转换为一维数组

        # 检查
        assert(x_row.shape == x_row.shape)  # 确保 x_row 和 y_row 的形状相等

        # 创建索引
        idxs = tuple(np.stack((x_row, y_row), axis=0))  # 在第0轴上堆叠 x_row 和 y_row，并转换为元组

        # 创建混淆矩阵(cols = gt, rows = pred)
        np.add.at(self.conf_matrix, idxs, 1)  # 使用索引将1加到 self.conf_matrix 中对应位置的元素上

    def getStats(self):
        # 从忽略的类别列中删除 fp
        conf = self.conf_matrix.copy()  # 复制 self.conf_matrix 到 conf
        conf[:, self.ignore] = 0  # 将忽略的类别列置零

        # 获取干净的统计数据
        tp = np.diag(conf)  # 获取对角线元素，即正确预测的数量
        fp = conf.sum(axis=1) - tp  # 按行求和，并减去 tp，得到 fp
        fn = conf.sum(axis=0) - tp  # 按列求和，并减去 tp，得到 fn
        return tp, fp, fn  # 返回 tp、fp、fn

    def getIoU(self):
        tp, fp, fn = self.getStats()  # 获取 tp、fp、fn
        intersection = tp  # 交集为 tp
        union = tp + fp + fn + 1e-15  # 并集为 tp + fp + fn + 1e-15（加上一个小的值以避免除以零）
        iou = intersection / union  # 计算 IoU
        iou_mean = (intersection[self.include] / union[self.include]).mean()  # 计算均值
        return iou_mean, iou  # 返回 "iou mean"、"iou per class"（所有类别的 IoU）

    def getacc(self):
        tp, fp, fn = self.getStats()  # 获取 tp、fp、fn
        total_tp = tp.sum()  # 计算总的 tp
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15  # 计算总的预测数量
        acc_mean = total_tp / total  # 计算准确率均值
        return acc_mean  # 返回 "acc mean"（准确率均值）

    def get_confusion(self):
        return self.conf_matrix.copy()  # 返回混淆矩阵的副本
