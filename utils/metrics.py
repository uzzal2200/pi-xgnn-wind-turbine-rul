"""Scalar evaluation metrics."""
import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def picp(lo: np.ndarray, hi: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((true >= lo) & (true <= hi)))


def mpiw(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def print_metrics(results: dict) -> None:
    border = "─" * 42
    print(f"\n{border}")
    print(f"  {'Metric':<8}  {'Value':>10}")
    print(border)
    for k in ("rmse", "mae", "r2", "picp", "mpiw", "z_star"):
        if k in results:
            print(f"  {k.upper():<8}  {results[k]:>10.4f}")
    print(border)
