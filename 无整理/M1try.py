import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import scipy.io as scio

# 在这里 import 你新的 modelM1.py 文件
from modelM1 import ChannelBandAttentionModelM1

# ================== 辅助函数 ==================
class eegDataset(Dataset):
    def __init__(self, data, label):
        super(eegDataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def load_DE_SEED(load_path):
    datasets = scio.loadmat(load_path)
    DE = datasets['DE']
    dataAll = np.transpose(DE, [1, 0, 2])  # [sample, channel, band]
    labelAll = datasets['labelAll'].flatten() + 1
    return dataAll, labelAll

def apply_standardization(train_data, test_data):
    shape = train_data.shape
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data.reshape(-1, shape[-1])).reshape(shape)
    test_data = scaler.transform(test_data.reshape(-1, shape[-1])).reshape(test_data.shape)
    return train_data, test_data

@torch.no_grad()
def evaluate(model, data_iter, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    preds_list = []
    labels_list = []

    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        pred = torch.argmax(outputs, dim=1)
        preds_list.append(pred)
        labels_list.append(y)
        correct += (pred == y).sum().item()
        total += len(y)

    all_preds = torch.cat(preds_list)
    all_labels = torch.cat(labels_list)
    overall_acc = correct / total

    # 各类别准确率
    class_acc = {}
    for c in torch.unique(all_labels):
        idx = (all_labels == c)
        class_correct = (all_preds[idx] == all_labels[idx]).sum().item()
        class_count = idx.sum().item()
        class_acc[int(c.item())] = class_correct / class_count if class_count>0 else 0

    return overall_acc, class_acc

# ========== 单频带训练函数 (类似 tryS3.py) ==========
def train_one_band(
    dir_path,
    band_idx,
    num_epochs=300,
    batch_size=64,
    lr=0.0005,
    weight_decay=0.005,
    patience=100,
    device='cuda',
    model_class=ChannelBandAttentionModelM1
):
    if not os.path.exists(dir_path):
        print(f"数据目录不存在: {dir_path}")
        return []

    file_list = [f for f in os.listdir(dir_path) if f.endswith('.mat')]
    if not file_list:
        print(f"在目录 {dir_path} 中未找到 .mat 文件")
        return []

    subject_acc_list = []
    start_time = time.time()

    print(f"\n=== [Band={band_idx}] 开始训练 ===")
    
    for sub_i, file_name in enumerate(file_list):
        file_path = os.path.join(dir_path, file_name)
        print(f"\n处理被试 {sub_i+1}/{len(file_list)}: {file_name}")

        # 1. 加载
        data, labels = load_DE_SEED(file_path)
        data = data[:, :, band_idx]  # [样本数, 通道数, len(band_idx)]

        # 2. trial划分
        total_samples = data.shape[0]
        num_trials = 15
        samples_per_trial = total_samples // num_trials
        train_indices = []
        test_indices = []
        for t in range(num_trials):
            start_idx = t * samples_per_trial
            end_idx = min((t+1) * samples_per_trial, total_samples)
            if t < 9:
                train_indices.extend(range(start_idx, end_idx))
            else:
                test_indices.extend(range(start_idx, end_idx))
        
        data_train = data[train_indices]
        label_train = labels[train_indices]
        data_test = data[test_indices]
        label_test = labels[test_indices]
        
        print(f"训练集: {data_train.shape}, 测试集: {data_test.shape}")

        # 3. 标准化
        data_train, data_test = apply_standardization(data_train, data_test)

        # 4. 转Tensor
        data_train_t = torch.tensor(data_train, dtype=torch.float32)
        label_train_t = torch.tensor(label_train, dtype=torch.long)
        data_test_t = torch.tensor(data_test, dtype=torch.float32)
        label_test_t = torch.tensor(label_test, dtype=torch.long)

        train_set = eegDataset(data_train_t, label_train_t)
        test_set = eegDataset(data_test_t, label_test_t)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # 5. 构建模型
        num_nodes = data_train.shape[1]
        num_bands = data_train.shape[2]
        xdim = [None, num_nodes, num_bands]

        model = model_class(
            xdim=xdim,
            k_adj=2,
            num_out=88,
            nclass=3
        ).to(device)

        # 6. 定义优化器/损失
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 7. 训练
        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        no_improve_count = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            test_acc, _ = evaluate(model, test_loader, device)
            avg_loss = total_loss / len(train_loader)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if (epoch+1)%5==0 or epoch==0 or epoch==num_epochs-1:
                print(f"[Epoch {epoch+1:02d}/{num_epochs}] Loss={avg_loss:.4f}  TestAcc={test_acc:.4f}")

            if no_improve_count >= patience:
                print(f"Early stopping at epoch={epoch+1}")
                break

        model.load_state_dict(best_model_wts)
        final_acc, _ = evaluate(model, test_loader, device)
        subject_acc_list.append(final_acc)
        print(f"被试 {sub_i+1} 的最佳准确率: {final_acc:.4f}")

        save_name = f"subject_{sub_i+1}_band{band_idx}_M1.pth"
        torch.save(best_model_wts, save_name)
        print(f"已保存最优模型: {save_name}")

    duration = (time.time() - start_time)/60
    print(f"\n[Band={band_idx}] 训练结束, 耗时: {duration:.2f} 分钟.")
    return subject_acc_list


def main_multi_band_experiment():
    band_sets = {
        'delta': [0],
        'theta': [1],
        'alpha': [2],
        'beta':  [3],
        'gamma': [4],
        'all':   [0,1,2,3,4]
    }

    dir_path = "D:/SEED/SEED_code/DE/session1/"
    num_epochs=300
    batch_size=64
    lr=0.0005
    weight_decay=0.005
    patience=100
    device='cuda'

    all_results = {}
    for band_name, band_idx in band_sets.items():
        print(f"\n====== 处理频带: {band_name} => 索引 {band_idx} ======")
        subject_accs = train_one_band(
            dir_path=dir_path,
            band_idx=band_idx,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            model_class=ChannelBandAttentionModelM1
        )
        if subject_accs:
            mean_acc = np.mean(subject_accs)
            std_acc = np.std(subject_accs)
            all_results[band_name] = (mean_acc, std_acc, subject_accs)
            print(f"[Band={band_name}] 各被试准确率: {[f'{x:.4f}' for x in subject_accs]}")
            print(f"[Band={band_name}] 平均准确率={mean_acc:.4f}, 标准差={std_acc:.4f}")
        else:
            print(f"[Band={band_name}] 无结果!")
    
    print("\n======== 实验总结 ========")
    for band_name, (mean_acc, std_acc, acc_list) in all_results.items():
        print(f"{band_name} => Mean={mean_acc:.4f}, Std={std_acc:.4f}, details={acc_list}")
def train_loso_band(
    dir_path,
    band_idx,
    num_epochs=300,
    batch_size=64,
    lr=0.0005,
    weight_decay=0.005,
    patience=100,
    device='cuda',
    model_class=ChannelBandAttentionModelM1
):
    file_list = [f for f in os.listdir(dir_path) if f.endswith('.mat')]
    subject_accs = []

    for i, test_file in enumerate(file_list):
        print(f"\n=== LOSO 被试 {i+1}/{len(file_list)}: {test_file} 作为测试集 ===")
        test_path = os.path.join(dir_path, test_file)
        test_data, test_label = load_DE_SEED(test_path)
        test_data = test_data[:, :, band_idx]

        train_data_list = []
        train_label_list = []

        for j, train_file in enumerate(file_list):
            if j == i:
                continue
            path = os.path.join(dir_path, train_file)
            data, label = load_DE_SEED(path)
            data = data[:, :, band_idx]
            train_data_list.append(data)
            train_label_list.append(label)

        train_data = np.concatenate(train_data_list, axis=0)
        train_label = np.concatenate(train_label_list, axis=0)

        train_data, test_data = apply_standardization(train_data, test_data)

        train_loader = DataLoader(eegDataset(torch.tensor(train_data, dtype=torch.float32),
                                             torch.tensor(train_label, dtype=torch.long)),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(eegDataset(torch.tensor(test_data, dtype=torch.float32),
                                            torch.tensor(test_label, dtype=torch.long)),
                                 batch_size=batch_size, shuffle=False)

        xdim = [None, train_data.shape[1], train_data.shape[2]]
        model = model_class(xdim=xdim, k_adj=2, num_out=88, nclass=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            acc, _ = evaluate(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping on epoch {epoch+1}")
                break

        print(f"被试 {i+1} 最佳准确率: {best_acc:.4f}")
        subject_accs.append(best_acc)

    return subject_accs


if __name__ == "__main__":
    main_multi_band_experiment()