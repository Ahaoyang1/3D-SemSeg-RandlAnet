import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from visualize import Plot


def main():
    base_dir = r'F:\实验结果\代码整合\2组数据\0.03\2\result\val_16\11\predictions'
    original_data_dir = r'F:\实验结果\代码整合\velodyne'

    prediction_all = []
    groundtruth_all = []
    total_correct = 0
    total_seen = 0
    visualization = True

    # 按品种分别存储
    prediction_dict = {'0': [], '1': [], '2': []}
    groundtruth_dict = {'0': [], '1': [], '2': []}
    correct_dict = {'0': 0, '1': 0, '2': 0}
    seen_dict = {'0': 0, '1': 0, '2': 0}

    # 获取所有预测文件路径
    data_path = [os.path.join(root, file)
                 for root, _, files in os.walk(base_dir)
                 for file in files if file.endswith('.npy')]

    for file_path in data_path:
        basename = os.path.basename(file_path)
        prefix = basename.split('-')[0]  # 提取前缀作为品种标识：'0', '1', '2'
        gt_path = os.path.join(original_data_dir, basename.replace('.npy', '.txt'))

        if not os.path.exists(gt_path):
            print(f"Warning: {gt_path} does not exist. Skipping.")
            continue

        pred = np.load(file_path).astype(int)
        origin = np.loadtxt(gt_path, dtype=float)
        labels = origin[:, -1].astype(int)
        points = origin[:, :3]

        acc = np.sum(pred == labels) / len(labels)
        print(f"{basename[:-4]} - Cloud_acc: {acc * 100:.8f}%")

        # 累加整体数据
        prediction_all.extend(pred)
        groundtruth_all.extend(labels)
        total_correct += np.sum(pred == labels)
        total_seen += len(labels)

        # 累加按品种数据
        prediction_dict[prefix].extend(pred)
        groundtruth_dict[prefix].extend(labels)
        correct_dict[prefix] += np.sum(pred == labels)
        seen_dict[prefix] += len(labels)

        # # 可视化
        # if visualization and acc > 0.8:
        #     Plot.draw_pc_sem_ins(points, labels)
        #     Plot.draw_pc_sem_ins(points, pred)

    all_preds = np.array(prediction_all)
    all_labels = np.array(groundtruth_all)
    num_classes = max(np.max(all_preds), np.max(all_labels)) + 1

    def evaluate(preds, gts, name):
        preds = np.array(preds)
        gts = np.array(gts)
        cm = confusion_matrix(gts, preds, labels=list(range(num_classes)))

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        Precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
        Recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
        F1 = np.divide(2 * Precision * Recall, Precision + Recall, out=np.zeros_like(TP, dtype=float),
                       where=(Precision + Recall) != 0)
        IoU = np.divide(TP, TP + FP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FP + FN) != 0)

        print(f"\n=== Evaluation for {name} ===")
        for i in range(num_classes):
            print(f"Class {i:2d}: TP={TP[i]:5d}, FP={FP[i]:5d}, TN={TN[i]:5d}, FN={FN[i]:5d}")
            print(f"    Precision: {Precision[i] * 100:.2f}%, Recall: {Recall[i] * 100:.2f}%, "
                  f"F1-score: {F1[i] * 100:.2f}%, IoU: {IoU[i] * 100:.2f}%")
        overall = np.sum(TP) / len(gts)
        print(f"Overall Accuracy: {overall * 100:.8f}%")

    for species_id in ['0', '1', '2']:
        if seen_dict[species_id] == 0:
            print(f"\n=== Species {species_id}: No data ===")
        else:
            evaluate(prediction_dict[species_id], groundtruth_dict[species_id], f"Species {species_id}")

    evaluate(prediction_all, groundtruth_all, "All Species")

    plt.show()


if __name__ == '__main__':
    main()