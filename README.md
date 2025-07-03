# 🌿 3D Plant Point Cloud Semantic Segmentation

This repository provides the implementation of a 3D semantic segmentation model for plant stem-leaf separation using **RandLA-Net** enhanced with **graph self-attention** and **semantic embedding** modules.

> 📌 **Validation Dataset:** Pheno4D / Plant-3D (SemanticKITTI format)  
> 📊 **Target Task:** Fine-grained semantic segmentation (e.g., stem vs. leaf)  
> 💡 **Features:** Lightweight architecture, multi-scale attention, non-destructive phenotyping

---


### Performance

> Results on Validation Set (seq 08)


## 🧱 A. Environment Setup

### 1. Install PyTorch

Please follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) and install version **≥1.4**, ensuring that the CUDA version matches your local GPU environment.

### 2. Install Required Python Packages

```bash
pip install -r requirements.txt
```

### 3. Compile C++ Wrappers

We use custom CUDA/C++ operations for neighbor search and grid subsampling:

```bash
bash compile_op.sh
```

---


## 📁 B. Data Preparation

### 1. Download Datasets

- 🌱 [Plant-3D dataset (Mendeley Data)](https://data.mendeley.com/datasets/9k7zctdyhs/1)
- 🌿 [Pheno4D dataset (IPB Bonn)](https://www.ipb.uni-bonn.de/data/pheno4d/)

### 2. Convert to SemanticKITTI Format

```bash
python data_prepare_semantickitti.py
```

> 🔧 Please **modify the dataset path** in `data_prepare_semantickitti.py` (look for `dataset_path = ...`) to your own local directory before running.

---


## 🧪 C. Training & Testing

### 1. Training

```bash
python3 train.py --batch_size 8 --log_dir log_train --gpu 0
```

You may add other args like `--epochs`, `--val_freq`, etc., based on your configuration.

### 2. Testing

```bash
python3 test.py   --checkpoint_path log_train/checkpoint.tar   --test_id 08   --index_to_label   --result_dir result/val_08
```

> ✅ If `--index_to_label` is set, predictions will be saved as `.label` files (for visualization).  
> ❎ If not set, results will be `.npy` files (for further evaluation or post-processing).

---

## 📊 D. Evaluation & Visualization

Evaluate predictions on the validation set:

```bash
python3 data_val.py --result_dir result/val_08
```

Optional: you can visualize `.label` predictions using tools from [SemanticKITTI devkit](http://semantic-kitti.org/).

---

## 📁 Project Structure

```text
├── train.py                  # Training pipeline
├── test.py                   # Inference pipeline
├── data_val.py               # Evaluation script
├── data_prepare_semantickitti.py
├── requirements.txt
│
├── dataset/
│   └── semkitti_testset.py   # Custom Dataset loader
├── network/
│   └── RandLANet.py          # Network definition with GSA modules
├── utils/
│   ├── data_process.py       # Preprocessing & KNN utilities
│   ├── config.py             # Hyperparameters
│   └── ...
```

---

## 🔍 Citation

If you use this code or datasets in your research, please consider citing the relevant datasets (Plant-3D, Pheno4D) and referring to this repository.

---


