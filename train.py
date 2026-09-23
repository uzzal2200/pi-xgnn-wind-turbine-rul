"""Main training script for PI-XGNN on XJTU-SY."""
import argparse
import torch

from config                    import Config
from datasets.xjtu_sy         import load_condition
from datasets.preprocess       import preprocess_bearing
from datasets.bearing_dataset  import build_loaders
from models.pi_xgnn            import PIXGNN
from engine.trainer            import train_with_seeds
from engine.evaluator          import evaluate
from utils.metrics             import print_metrics
from utils.visualization       import plot_rul


def parse_args():
    p = argparse.ArgumentParser(description="Train PI-XGNN on XJTU-SY")
    p.add_argument("--data_dir",   default="./data/XJTU-SY",
                   help="Root directory of XJTU-SY dataset")
    p.add_argument("--condition",  type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--seeds",      type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--plot",       action="store_true",
                   help="Save RUL prediction plot after evaluation")
    p.add_argument("--save",       default="pi_xgnn.pt",
                   help="Checkpoint output path")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config(
        data_dir   = args.data_dir,
        condition  = args.condition,
        epochs     = args.epochs,
        n_seeds    = args.seeds,
        batch_size = args.batch_size,
        lr         = args.lr,
    )
    device = torch.device(cfg.device)
    print(f"PI-XGNN  |  Condition {cfg.condition}  |  Device: {device}")

    # ── Load & preprocess ────────────────────────────────────────────────
    print("\n[1/3] Loading XJTU-SY data …")
    raw = load_condition(cfg.data_dir, cfg.condition)
    if not raw:
        raise RuntimeError("No bearing data found. Check --data_dir.")

    processed = {}
    for bid, (h, v) in raw.items():
        feat, rul, fpt, _ = preprocess_bearing(
            h, v, cfg.irrms_alpha, cfg.fpt_eta,
            cfg.pelt_penalty_mult, cfg.pelt_min_size)
        processed[bid] = (feat, rul)
        print(f"  Bearing{cfg.condition}_{bid}: {len(feat)} cycles  FPT={fpt}")

    train_ids = [i for i in range(1, cfg.n_train_bearings + 1) if i in processed]
    if cfg.test_bearing_id not in processed:
        raise RuntimeError(f"Test bearing {cfg.test_bearing_id} not found.")

    train_bearings = [processed[i] for i in train_ids]
    test_bearing   = processed[cfg.test_bearing_id]

    train_loader, test_ds, _ = build_loaders(
        train_bearings, test_bearing,
        window=cfg.window_length, batch_size=cfg.batch_size)

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\n[2/3] Training  ({cfg.n_seeds} seeds × {cfg.epochs} epochs) …")
    model = train_with_seeds(PIXGNN, {"config": cfg},
                              train_loader, cfg, device)

    torch.save({"state_dict": model.state_dict(), "config": cfg}, args.save)
    print(f"  Checkpoint saved → {args.save}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Evaluating Bearing{cfg.condition}_{cfg.test_bearing_id} …")
    results = evaluate(model, test_ds, cfg, device)
    print_metrics(results)

    if args.plot:
        plot_rul(results,
                 title=f"PI-XGNN · Condition {cfg.condition} · "
                       f"Bearing{cfg.condition}_{cfg.test_bearing_id}",
                 save_path=f"rul_C{cfg.condition}.png")


if __name__ == "__main__":
    main()
