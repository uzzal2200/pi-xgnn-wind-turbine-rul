"""
Quick demo with synthetic data — no real dataset required.

Usage:
    python demo.py
    python demo.py --epochs 100 --seeds 1
"""
import argparse
import torch

from config                   import Config
from datasets.synthetic       import generate_bearing
from datasets.preprocess      import preprocess_bearing
from datasets.bearing_dataset import build_loaders
from models.pi_xgnn           import PIXGNN
from engine.trainer           import train_with_seeds
from engine.evaluator         import evaluate
from utils.metrics            import print_metrics
from utils.visualization      import plot_rul


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seeds",  type=int, default=1)
    p.add_argument("--plot",   action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config(epochs=args.epochs, n_seeds=args.seeds)
    device = torch.device(cfg.device)

    print(f"PI-XGNN Demo  |  device={device}  |  epochs={cfg.epochs}")
    print("Generating synthetic bearing data …")

    # 4 training bearings, 1 test bearing (all synthetic)
    train_bearings = []
    for seed in range(4):
        h, v = generate_bearing(T=100, seed=seed)
        feat, rul, fpt, _ = preprocess_bearing(h, v)
        train_bearings.append((feat, rul))
        print(f"  Train bearing {seed + 1}: T=100  FPT={fpt}")

    h, v = generate_bearing(T=120, seed=99)
    feat, rul, fpt, _ = preprocess_bearing(h, v)
    test_bearing = (feat, rul)
    print(f"  Test  bearing 5: T=120  FPT={fpt}")

    train_loader, test_ds, _ = build_loaders(
        train_bearings, test_bearing,
        window=cfg.window_length, batch_size=cfg.batch_size)

    print(f"\nTraining PI-XGNN ({cfg.n_seeds} seed × {cfg.epochs} epochs) …")
    model = train_with_seeds(PIXGNN, {"config": cfg},
                              train_loader, cfg, device)

    print("\nEvaluating …")
    results = evaluate(model, test_ds, cfg, device)
    print_metrics(results)

    if args.plot:
        plot_rul(results, title="PI-XGNN – Synthetic Demo",
                 save_path="demo_rul.png")
        print("Plot saved → demo_rul.png")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
