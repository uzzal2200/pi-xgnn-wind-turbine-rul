"""PI-XGNN Configuration."""
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "./data/XJTU-SY"
    condition: int = 1            # 1 | 2 | 3
    n_train_bearings: int = 4     # bearings 1-4 → train
    test_bearing_id: int = 5      # bearing 5 → test

    # ── Feature / Window ──────────────────────────────────────────────────
    n_features: int = 12
    window_length: int = 5
    irrms_alpha: float = 0.1
    fpt_eta: float = 1.1
    pelt_penalty_mult: float = 3.0
    pelt_min_size: int = 5

    # ── Model ─────────────────────────────────────────────────────────────
    latent_dim: int = 8
    n_heads: int = 4
    dropout: float = 0.20

    # ── Physics weights ───────────────────────────────────────────────────
    lambda_pde: float = 0.15
    lambda_mono: float = 0.15

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    n_seeds: int = 3
    grad_clip: float = 1.0

    # ── Inference / Calibration ───────────────────────────────────────────
    mc_samples: int = 100
    alpha_aleatoric: float = 0.10
    coverage_target: float = 0.95

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
