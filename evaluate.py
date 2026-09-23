"""Evaluate a saved PI-XGNN checkpoint on XJTU-SY."""
import argparse
import torch

from config                   import Config
from datasets.xjtu_sy        import load_condition
from datasets.preprocess      import preprocess_bearing
from datasets.bearing_dataset import build_loaders
from models.pi_xgnn           import PIXGNN
from engine.evaluator         import evaluate
from utils.metrics            import print_metrics
from utils.visualization      import plot_rul, plot_feature_importance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pi_xgnn.pt")
    p.add_argument("--plot",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg  = ckpt.get("config", Config())
    device = torch.device(cfg.device)

    model = PIXGNN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    raw = load_condition(cfg.data_dir, cfg.condition)
    processed = {}
    for bid, (h, v) in raw.items():
        feat, rul, fpt, _ = preprocess_bearing(
            h, v, cfg.irrms_alpha, cfg.fpt_eta)
        processed[bid] = (feat, rul)

    train_ids      = [i for i in range(1, cfg.n_train_bearings + 1) if i in processed]
    train_bearings = [processed[i] for i in train_ids]
    test_bearing   = processed[cfg.test_bearing_id]

    _, test_ds, _ = build_loaders(
        train_bearings, test_bearing,
        window=cfg.window_length, batch_size=cfg.batch_size)

    results = evaluate(model, test_ds, cfg, device)
    print_metrics(results)

    if args.plot:
        plot_rul(results,
                 title=f"PI-XGNN · C{cfg.condition} · Bearing{cfg.test_bearing_id}",
                 save_path=f"rul_C{cfg.condition}_eval.png")

        # Feature importance on test set
        import numpy as np
        from torch.utils.data import DataLoader
        loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
        x_all, _, t_all = next(iter(loader))
        x_all, t_all = x_all.to(device), t_all.to(device)
        imp = model.node_importance(x_all, t_all, n_samples=50).cpu().numpy()
        plot_feature_importance(imp, save_path="feature_importance.png")


if __name__ == "__main__":
    main()
