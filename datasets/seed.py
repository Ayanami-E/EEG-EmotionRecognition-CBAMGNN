import numpy as np, torch, scipy.io as scio
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

def load_seed_mat(path):
    d = scio.loadmat(path)
    de = np.transpose(d['DE'], [1,0,2])  # [sample, channel, band]
    labels = d['labelAll'].flatten() + 1
    return de, labels

def standardize(train, test):
    shp = train.shape; sc = StandardScaler()
    train = sc.fit_transform(train.reshape(-1, shp[-1])).reshape(shp)
    test  = sc.transform(test.reshape(-1, shp[-1])).reshape(test.shape)
    return train, test

class SEEDDataset(Dataset):
    def __init__(self, data, labels):
        self.x = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
