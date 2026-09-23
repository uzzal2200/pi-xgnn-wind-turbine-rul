"""PyTorch Dataset and DataLoader utilities."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler


class BearingDataset(Dataset):
    """Sliding-window dataset for one bearing."""

    def __init__(self, features: np.ndarray, rul: np.ndarray, window: int = 5):
        T          = len(features)
        self.X     = torch.FloatTensor(features)
        self.y     = torch.FloatTensor(rul)
        self.t_norm = torch.FloatTensor(np.arange(T, dtype=np.float32) / T)
        self.L     = window

    def __len__(self):
        return len(self.X) - self.L + 1

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.L]           # (L, d)
        y = self.y[idx + self.L - 1].unsqueeze(0)   # (1,)
        t = self.t_norm[idx + self.L - 1].unsqueeze(0)  # (1,)
        return x, y, t


def build_loaders(train_bearings: list,
                   test_bearing: tuple,
                   window: int = 5,
                   batch_size: int = 64):
    """
    Parameters
    ----------
    train_bearings : list of (features, rul)
    test_bearing   : (features, rul)

    Returns
    -------
    train_loader, test_dataset, scaler
    """
    # Fit StandardScaler on pooled training features
    all_feat = np.vstack([f for f, _ in train_bearings])
    scaler   = StandardScaler().fit(all_feat)

    # Training datasets
    train_ds = []
    for feat, rul in train_bearings:
        feat_s = scaler.transform(feat).astype(np.float32)
        train_ds.append(BearingDataset(feat_s, rul, window))

    train_loader = DataLoader(
        ConcatDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Test dataset
    feat_s  = scaler.transform(test_bearing[0]).astype(np.float32)
    test_ds = BearingDataset(feat_s, test_bearing[1], window)

    return train_loader, test_ds, scaler
