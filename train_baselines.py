"""
Train all (or selected) baseline models and compare with PI-XGNN.

Usage
-----
# All baselines + PI-XGNN
python train_baselines.py --data_dir ./data/XJTU-SY --condition 1

# Select specific models
python train_baselines.py --models mlp lstm pi_tenn

# Synthetic demo (no dataset needed)
python train_baselines.py --demo --epochs 20 --seeds 1
"""
import argparse
import copy
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from config                   import Config
from datasets.bearing_dataset import build_loaders
from datasets.preprocess      import preprocess_bearing
from models.pi_xgnn           import PIXGNN
from models.baselines         import BASELINE_REGISTRY
from engine.losses            import pi_loss
from engine.evaluator         import evaluate
from utils.metrics            import print_metrics


# ── Loss dispatch ─────────────────────────────────────────────────────────────

def compute_loss(model_name, out, y, config):
    """
    Dispatch the correct loss depending on model type.

    out = (pred, aux1, aux2) from model.forward()
    """
    pred, aux1, aux2 = out
    _, uses_pde = BASELINE_REGISTRY.get(model_name, (None, False))

    if model_name == "attn_pinn":
        # aux1 = softplus monotonicity loss computed inside AttnPINN.forward
        mono_loss = aux1 if aux1 is not None else pred.new_zeros(1).squeeze()
        loss = torch.nn.functional.mse_loss(pred, y) + config.lambda_mono * mono_loss
        return loss

    if uses_pde and aux1 is not None and aux2 is not None:
        # PI-TENN / PI-XGNN style: aux1 = du/dt, aux2 = G_pred
        loss, _ = pi_loss(pred, y, aux1, aux2,
                           config.lambda_pde, config.lambda_mono)
        return loss

    # Pure data-driven baselines
    return torch.nn.functional.mse_loss(pred, y)


# ── Single model trainer ──────────────────────────────────────────────────────

def train_model(model_class, config, train_loader, device,
                model_name="model", n_seeds=3, epochs=200, verbose=True):
    """Train n_seeds times, keep best train-loss model."""
    best_loss  = float("inf")
    best_state = None

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model     = model_class(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=config.lr,
                                       weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        for epoch in range(epochs):
            model.train()
            total, nb = 0.0, 0
            for x, y, t in train_loader:
                x, y, t = x.to(device), y.to(device), t.to(device)
                optimizer.zero_grad()
                out  = model(x, t)
                loss = compute_loss(model_name, out, y, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                total += loss.item(); nb += 1
            scheduler.step()
            avg = total / max(nb, 1)

        if avg < best_loss:
            best_loss  = avg
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    seed {seed+1}/{n_seeds}  final_loss={avg:.4f}"
                  + ("  ← best" if avg == best_loss else ""))

    best = model_class(config).to(device)
    best.load_state_dict(best_state)
    return best


# ── Quick predict_mc shim for non-probabilistic baselines ─────────────────────

def add_mc_shim(model):
    """Attach a predict_mc method to baselines that lack one."""
    if not hasattr(model, "predict_mc"):
        def predict_mc(x, t, n_samples=100):
            model.train()
            preds = []
            with torch.no_grad():
                for _ in range(n_samples):
                    out = model(x, t)
                    preds.append(out[0])
            stacked = torch.stack(preds)
            return stacked.mean(0), stacked.std(0)
        model.predict_mc = predict_mc
    return model


# ── Results table printer ─────────────────────────────────────────────────────

def print_table(all_results: dict):
    header = f"{'Model':<14} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'PICP':>7} {'MPIW':>7}"
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, res in all_results.items():
        tag = " ← PI-XGNN" if name == "pi_xgnn" else ""
        print(f"{name:<14} "
              f"{res['rmse']:>7.4f} "
              f"{res['mae']:>7.4f} "
              f"{res['r2']:>7.4f} "
              f"{res['picp']:>7.3f} "
              f"{res['mpiw']:>7.4f}{tag}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data/XJTU-SY")
    p.add_argument("--condition",  type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--seeds",      type=int, default=3)
    p.add_argument("--models",     nargs="+",
                   default=list(BASELINE_REGISTRY.keys()) + ["pi_xgnn"],
                   help="Subset of models to run")
    p.add_argument("--demo",       action="store_true",
                   help="Use synthetic data (no dataset required)")
    p.add_argument("--save_dir",   default="checkpoints",
                   help="Directory to save trained model checkpoints")
    return p.parse_args()


def load_data(args, cfg):
    if args.demo:
        from datasets.synthetic import generate_bearing
        print("Using synthetic bearing data (demo mode).")
        train_bearings, test_bearing = [], None
        for s in range(4):
            h, v = generate_bearing(T=80, seed=s)
            feat, rul, fpt, _ = preprocess_bearing(h, v)
            train_bearings.append((feat, rul))
            print(f"  Train bearing {s+1}: T=80  FPT={fpt}")
        h, v = generate_bearing(T=100, seed=99)
        feat, rul, fpt, _ = preprocess_bearing(h, v)
        test_bearing = (feat, rul)
        print(f"  Test  bearing 5: T=100  FPT={fpt}")
    else:
        from datasets.xjtu_sy import load_condition
        raw = load_condition(cfg.data_dir, cfg.condition)
        if not raw:
            raise RuntimeError("No bearing data found. Check --data_dir.")
        processed = {}
        for bid, (h, v) in raw.items():
            feat, rul, fpt, _ = preprocess_bearing(h, v)
            processed[bid] = (feat, rul)
            print(f"  Bearing{cfg.condition}_{bid}: T={len(feat)}  FPT={fpt}")
        train_ids      = [i for i in range(1, cfg.n_train_bearings+1) if i in processed]
        train_bearings = [processed[i] for i in train_ids]
        test_bearing   = processed[cfg.test_bearing_id]

    return build_loaders(train_bearings, test_bearing,
                         window=cfg.window_length,
                         batch_size=cfg.batch_size)


def main():
    args   = parse_args()
    cfg    = Config(data_dir=args.data_dir, condition=args.condition,
                    epochs=args.epochs, n_seeds=args.seeds)
    device = torch.device(cfg.device)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nPI-XGNN Baseline Comparison")
    print(f"Condition {cfg.condition}  |  device={device}  |  "
          f"epochs={cfg.epochs}  |  seeds={cfg.n_seeds}")
    print(f"Models: {args.models}\n")

    print("[1/2] Loading data …")
    train_loader, test_ds, _ = load_data(args, cfg)

    all_results = {}
    models_to_run = [m for m in args.models
                     if m in BASELINE_REGISTRY or m == "pi_xgnn"]

    print(f"\n[2/2] Training {len(models_to_run)} model(s) …")

    for name in models_to_run:
        print(f"\n── {name.upper()} ──")

        if name == "pi_xgnn":
            from engine.trainer import train_with_seeds
            model = train_with_seeds(PIXGNN, {"config": cfg},
                                      train_loader, cfg, device)
        else:
            model_class, _ = BASELINE_REGISTRY[name]
            model = train_model(model_class, cfg, train_loader, device,
                                model_name=name,
                                n_seeds=cfg.n_seeds,
                                epochs=cfg.epochs)
            add_mc_shim(model)

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"{name}_C{cfg.condition}.pt")
        torch.save({"state_dict": model.state_dict(), "config": cfg,
                    "model_name": name}, ckpt_path)

        # Evaluate
        results = evaluate(model, test_ds, cfg, device)
        all_results[name] = results
        print(f"  RMSE={results['rmse']:.4f}  R²={results['r2']:.4f}  "
              f"PICP={results['picp']:.3f}")

    print_table(all_results)


if __name__ == "__main__":
    main()
