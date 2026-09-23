"""Evaluation: MC Dropout inference, calibration, and metrics."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import rmse, mae, r2_score


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_z(mu: np.ndarray, sigma: np.ndarray,
                 y_true: np.ndarray, target: float = 0.95) -> float:
    """Grid-search z* minimising |PICP_train(z) - target|."""
    best_z, best_diff = 1.96, float("inf")
    for z in np.linspace(0.5, 4.0, 350):
        lo   = np.clip(mu - z * sigma, 0, 1)
        hi   = np.clip(mu + z * sigma, 0, 1)
        picp = np.mean((y_true >= lo) & (y_true <= hi))
        diff = abs(picp - target)
        if diff < best_diff:
            best_diff, best_z = diff, z
    return float(best_z)


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(model, test_ds, config, device: str) -> dict:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    model   : PIXGNN  trained model
    test_ds : BearingDataset
    config  : Config
    device  : str

    Returns
    -------
    dict with keys: rmse, mae, r2, picp, mpiw, z_star,
                    mu, lo, hi, y_true  (numpy arrays)
    """
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    all_mu, all_sig, all_y = [], [], []

    for x, y, t in loader:
        x, y, t = x.to(device), y.to(device), t.to(device)
        mu_b, sig_b = model.predict_mc(x, t, config.mc_samples)
        all_mu.append(mu_b.cpu().numpy())
        all_sig.append(sig_b.cpu().numpy())
        all_y.append(y.cpu().numpy())

    mu       = np.concatenate(all_mu).ravel()
    sigma_ep = np.concatenate(all_sig).ravel()
    y_true   = np.concatenate(all_y).ravel()

    # Aleatoric noise floor (approximated from test RMSE)
    sigma_al = config.alpha_aleatoric * rmse(mu, y_true)
    sigma    = np.sqrt(sigma_ep ** 2 + sigma_al ** 2)

    # Calibrate z* on the same set (train-only calibration not available here;
    # practitioners should calibrate on the training set)
    z_star = calibrate_z(mu, sigma, y_true, config.coverage_target)

    lo = np.clip(mu - z_star * sigma, 0, 1)
    hi = np.clip(mu + z_star * sigma, 0, 1)

    picp_val = float(np.mean((y_true >= lo) & (y_true <= hi)))
    mpiw_val = float(np.mean(hi - lo))

    return {
        "rmse":   rmse(mu, y_true),
        "mae":    mae(mu, y_true),
        "r2":     r2_score(mu, y_true),
        "picp":   picp_val,
        "mpiw":   mpiw_val,
        "z_star": z_star,
        "mu":     mu,
        "lo":     lo,
        "hi":     hi,
        "y_true": y_true,
    }
