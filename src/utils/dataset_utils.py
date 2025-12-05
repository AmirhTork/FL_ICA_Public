import numpy as np
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype('float32')
        self.y = y.astype('int64')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def synthetic_binary_classification(n_samples=2000, n_features=20, imbalance=0.5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=(n_features,))
    logits = X.dot(w)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > (1 - imbalance)).astype(int)
    return X, y
