"""Train ChannelBandAttentionModelM1 on SEED dataset.

Examples
--------
# 被试内 (DEP) Alpha 频带
python scripts/train.py --data ./data --band alpha --mode dep

# Leave‑One‑Subject‑Out (LOSO) Gamma 频带
python scripts/train.py --data ./data --band gamma --mode loso

# 多频带循环
python scripts/train.py --data ./data --mode multi
"""
import argparse, json, time, copy, numpy as np
from pathlib import Path
import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader

from datasets.seed import load_seed_mat, standardize, SEEDDataset
from models.model_m1 import ChannelBandAttentionModelM1
from engine.trainer import Trainer

BANDS = {'delta':[0],'theta':[1],'alpha':[2],'beta':[3],'gamma':[4],'all':[0,1,2,3,4]}

def build_model(num_nodes, num_bands):
    return ChannelBandAttentionModelM1(num_nodes=num_nodes, num_bands=num_bands)

@torch.no_grad()
def evaluate(model, loader, device='cuda'):
    model.eval(); c,t = 0,0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        c += (model(xb).argmax(1)==yb).sum().item(); t += len(yb)
    return c/t if t else 0

# ---------------- DEP ----------------
def run_dep(mat_paths, band_idx, hp):
    accs = []
    for p in mat_paths:
        data, lab = load_seed_mat(p); data = data[:,:,band_idx]
        n = int(0.9 * len(data))
        tr_x, te_x = data[:n], data[n:]; tr_y, te_y = lab[:n], lab[n:]
        tr_x, te_x = standardize(tr_x, te_x)
        tr_dl = DataLoader(SEEDDataset(tr_x,tr_y), batch_size=hp.bs, shuffle=True)
        te_dl = DataLoader(SEEDDataset(te_x,te_y), batch_size=hp.bs)
        model = build_model(tr_x.shape[1], tr_x.shape[2])
        best = Trainer(model, lr=hp.lr, wd=hp.wd, patience=hp.pat).fit(tr_dl, te_dl, hp.epochs)
        accs.append(best); print(p.name, best)
    print('DEP Mean', np.mean(accs))

# ---------------- LOSO ---------------
def run_loso(mat_paths, band_idx, hp):
    accs = []
    for i, test_p in enumerate(mat_paths):
        print(f'\nLOSO {i+1}/{len(mat_paths)}: {test_p.name} as test')
        te_x, te_y = load_seed_mat(test_p); te_x = te_x[:,:,band_idx]
        tr_x_list, tr_y_list = [], []
        for j, p in enumerate(mat_paths):
            if j==i: continue
            x,y = load_seed_mat(p); tr_x_list.append(x[:,:,band_idx]); tr_y_list.append(y)
        tr_x, tr_y = np.concatenate(tr_x_list), np.concatenate(tr_y_list)
        tr_x, te_x = standardize(tr_x, te_x)
        tr_dl = DataLoader(SEEDDataset(tr_x,tr_y), batch_size=hp.bs, shuffle=True)
        te_dl = DataLoader(SEEDDataset(te_x,te_y), batch_size=hp.bs)
        model = build_model(tr_x.shape[1], tr_x.shape[2])
        best = Trainer(model, lr=hp.lr, wd=hp.wd, patience=hp.pat).fit(tr_dl, te_dl, hp.epochs)
        accs.append(best); print('Acc', best)
    print('LOSO Mean', np.mean(accs))

# -------------- Multi‑band over subjects ---------------
def run_multi(mat_paths, hp):
    results = {}
    for name, idx in BANDS.items():
        print(f'\n### Band {name} ###')
        accs = []
        for p in mat_paths:
            data, lab = load_seed_mat(p); data = data[:,:,idx]
            n = int(0.9*len(data)); tr_x, te_x = data[:n], data[n:]; tr_y, te_y = lab[:n], lab[n:]
            tr_x, te_x = standardize(tr_x, te_x)
            tr_dl = DataLoader(SEEDDataset(tr_x,tr_y), batch_size=hp.bs, shuffle=True)
            te_dl = DataLoader(SEEDDataset(te_x,te_y), batch_size=hp.bs)
            model = build_model(tr_x.shape[1], tr_x.shape[2])
            best = Trainer(model, lr=hp.lr, wd=hp.wd, patience=hp.pat).fit(tr_dl, te_dl, hp.epochs)
            accs.append(best)
        results[name] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
        print(name, results[name])
    print('\nSummary', json.dumps(results, indent=2))

# ---------------- CLI ----------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--band', default='alpha', choices=BANDS.keys())
    ap.add_argument('--mode', choices=['dep','loso','multi'], default='dep')
    ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=5e-3)
    ap.add_argument('--pat', type=int, default=50)
    hp = ap.parse_args()


    mats = sorted(hp.data.glob('*.mat'))
    if not mats: raise FileNotFoundError('No .mat in data dir')

    if hp.mode=='dep':
        run_dep(mats, BANDS[hp.band], hp)
    elif hp.mode=='loso':
        run_loso(mats, BANDS[hp.band], hp)
    else:
        run_multi(mats, hp)

