"""
Master EDA + Results script — generates ALL figures and tables from the paper.

Figures generated
─────────────────
  Fig 05  Raw vibration at 3 stages
  Fig 06  FFT spectra at 4 stages
  Fig 07  All 12 features + IRRMS / FPT / STP / RUL
  Fig 08  Adaptive sensor-correlation graph
  Fig 11  Training + validation loss curves
  Fig 12  σ(τ) evolution + graph edge density
  Fig 13  RUL trajectories — all 8 methods
  Fig 14  PI-XGNN detail + 95% CI
  Fig 15  Absolute error distributions (boxplot)
  Fig 16  PI-XGNN across 3 XJTU-SY conditions
  Fig 17  Epistemic uncertainty dynamics
  Fig 18  Calibration reliability diagrams
  Fig 19  Gradient-based feature importance (3 stages)
  Fig 26  Ablation RUL curves
  Fig 27  Ablation component bar chart
  Fig 28  Hyperparameter sensitivity

Tables generated
────────────────
  Tables 1–14  (static + dynamic from experiments)

Usage
─────
  # Synthetic demo — no dataset required
  python eda/run_all.py --demo

  # Real XJTU-SY data
  python eda/run_all.py --data_dir ./data/XJTU-SY --condition 1

  # Only EDA figures (no training needed)
  python eda/run_all.py --demo --only_eda

  # Only static tables
  python eda/run_all.py --only_tables
"""
import sys, os, argparse, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config                   import Config
from datasets.preprocess      import preprocess_bearing
from datasets.bearing_dataset import build_loaders
from models.pi_xgnn           import PIXGNN
from models.baselines         import BASELINE_REGISTRY
from engine.losses            import pi_loss
from engine.evaluator         import evaluate

from eda.data_eda       import generate_eda_figures
from eda.training_viz   import train_with_history, plot_fig11, plot_fig12
from eda.result_plots   import (plot_fig13, plot_fig14, plot_fig15,
                                  plot_fig16, plot_fig17, plot_fig18,
                                  plot_fig19, plot_fig26_27, plot_fig28)
from eda.tables         import generate_static_tables, generate_result_tables


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="./data/XJTU-SY",
                   help="XJTU-SY root directory")
    p.add_argument("--phm_dir",     default="./data/PHM2012",
                   help="PHM 2012 root directory (contains Learning_set/)")
    p.add_argument("--grc_dir",     default="./data/NREL_GRC",
                   help="NREL GRC root directory (contains Healthy/ Damaged/)")
    p.add_argument("--condition",   type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--seeds",       type=int, default=3)
    p.add_argument("--fine_tune_epochs", type=int, default=50,
                   help="PHM 2012 fine-tuning epochs")
    p.add_argument("--demo",        action="store_true",
                   help="Use synthetic data (no dataset required)")
    p.add_argument("--only_eda",    action="store_true",
                   help="Only generate EDA figures (Figs 5–8)")
    p.add_argument("--only_tables", action="store_true",
                   help="Only generate static tables (1–7, 14)")
    p.add_argument("--skip_cross",  action="store_true",
                   help="Skip cross-domain experiments (Figs 20–25)")
    p.add_argument("--fig_dir",     default="./figures")
    p.add_argument("--tbl_dir",     default="./tables")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw(args, cond):
    if args.demo:
        from datasets.synthetic import generate_bearing
        raw = {}
        for bid in range(1, 6):
            T = np.random.randint(70, 130)
            raw[bid] = generate_bearing(T=T, seed=bid + cond * 10)
        return raw
    else:
        from datasets.xjtu_sy import load_condition
        return load_condition(args.data_dir, cond)


def preprocess_all(raw, cfg):
    processed = {}
    for bid, (h, v) in raw.items():
        feat, rul, fpt, irrms = preprocess_bearing(
            h, v, cfg.irrms_alpha, cfg.fpt_eta,
            cfg.pelt_penalty_mult, cfg.pelt_min_size)
        processed[bid] = dict(feat=feat, rul=rul, fpt=fpt, irrms=irrms,
                               cycles_h=h, cycles_v=v)
        print(f"  Bearing{cfg.condition}_{bid}: T={len(feat)}  FPT={fpt}")
    return processed


def get_stps(irrms, fpt, cfg):
    """Return STP1, STP2 from PELT changepoints."""
    from datasets.preprocess import pelt_changepoints
    T   = len(irrms)
    bps = pelt_changepoints(irrms[fpt:], cfg.pelt_penalty_mult, cfg.pelt_min_size)
    bps_abs = sorted([fpt + b for b in bps])
    stp1 = bps_abs[0] if len(bps_abs) > 0 else (fpt + T) // 2
    stp2 = bps_abs[1] if len(bps_abs) > 1 else int(T * 0.85)
    return int(stp1), int(stp2)


# ── Validation split helper ───────────────────────────────────────────────────

def train_val_split(train_loader, val_frac=0.2):
    """Build a DataLoader for last val_frac of each dataset in ConcatDataset."""
    ds = train_loader.dataset
    total = len(ds)
    n_val = max(1, int(total * val_frac))
    n_train = total - n_val
    train_sub = Subset(ds, list(range(n_train)))
    val_sub   = Subset(ds, list(range(n_train, total)))
    t_loader  = DataLoader(train_sub, batch_size=train_loader.batch_size, shuffle=True)
    v_loader  = DataLoader(val_sub,   batch_size=256, shuffle=False)
    return t_loader, v_loader


# ── Ablation experiments ──────────────────────────────────────────────────────

def run_ablation(train_loader, test_ds, cfg, device):
    """Train No-Graph / No-Physics / No-PDE / Full and evaluate each."""
    from engine.trainer import train_one_seed

    results = {}
    variants = {
        "No-Graph":    dict(lambda_pde=0.15, lambda_mono=0.15, graph=False),
        "No-Physics":  dict(lambda_pde=0.0,  lambda_mono=0.0,  graph=True),
        "No-PDE":      dict(lambda_pde=0.0,  lambda_mono=0.15, graph=True),
        "PI-XGNN":     dict(lambda_pde=0.15, lambda_mono=0.15, graph=True),
    }

    for vname, params in variants.items():
        print(f"  Ablation: {vname}")
        c          = copy.deepcopy(cfg)
        c.lambda_pde  = params["lambda_pde"]
        c.lambda_mono = params["lambda_mono"]

        if not params["graph"]:
            # Use PI-TENN as No-Graph variant
            from models.baselines import PITENN
            model = PITENN(c)
        else:
            model = PIXGNN(c)

        torch.manual_seed(0)
        model, _ = train_one_seed(model, train_loader, c, device, verbose=False)
        res = evaluate(model, test_ds, c, device)
        results[vname] = res
        print(f"    RMSE={res['rmse']:.4f}")

    return results


# ── Window sensitivity ────────────────────────────────────────────────────────

def run_window_sensitivity(train_bearings, test_bearing, cfg, device,
                            windows=(3, 5, 7, 10)):
    from engine.trainer import train_one_seed
    results = {}
    for L in windows:
        print(f"  Window L={L}")
        c = copy.deepcopy(cfg)
        c.window_length = int(L)
        tl, tds, _ = build_loaders(train_bearings, test_bearing,
                                    window=int(L), batch_size=cfg.batch_size)
        if len(tl) == 0:
            print(f"    Skipped (empty loader for L={L})")
            continue
        torch.manual_seed(0)
        model = PIXGNN(c)
        model, _ = train_one_seed(model, tl, c, device, verbose=False)
        results[L] = evaluate(model, tds, c, device)
    return results


# ── Hyperparameter sensitivity ────────────────────────────────────────────────

def run_hyperparam_sensitivity(train_loader, test_ds, cfg, device):
    from engine.trainer import train_one_seed
    sweep = {
        "lambda_pde":  [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        "lambda_mono": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        "dropout":     [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        # window uses run_window_sensitivity (needs its own loaders)
    }
    sensitivity = {}
    for param, vals in sweep.items():
        print(f"  Sensitivity: {param}")
        rmses = []
        for v in vals:
            c = copy.deepcopy(cfg)
            if param == "lambda_pde":    c.lambda_pde   = v
            elif param == "lambda_mono": c.lambda_mono  = v
            elif param == "dropout":     c.dropout      = v
            elif param == "window":
                c.window_length = int(v)
            torch.manual_seed(0)
            # For window sweep, use the pre-built loader (same window=5)
            # just change the config; model adapts via window_length in Config
            cur_loader = train_loader
            cur_ds     = test_ds
            m, _ = train_one_seed(PIXGNN(c), cur_loader, c, device, verbose=False)
            r = evaluate(m, cur_ds, c, device)
            rmses.append(r["rmse"])
        sensitivity[param] = {"values": vals, "rmse": rmses}
    return sensitivity


# ── Baseline runner ───────────────────────────────────────────────────────────

def run_baselines(train_loader, test_ds, cfg, device):
    from engine.trainer import train_one_seed
    import torch.nn.functional as F

    results = {}
    for name, (cls, uses_pde) in BASELINE_REGISTRY.items():
        print(f"  Baseline: {name}")
        torch.manual_seed(0)
        model     = cls(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

        for _ in range(cfg.epochs):
            model.train()
            for x, y, t in train_loader:
                x, y, t = x.to(device), y.to(device), t.to(device)
                optimizer.zero_grad()
                pred, a1, a2 = model(x, t)
                if uses_pde and a1 is not None and a2 is not None and name != "attn_pinn":
                    loss, _ = pi_loss(pred, y, a1, a2, cfg.lambda_pde, cfg.lambda_mono)
                elif name == "attn_pinn" and a1 is not None:
                    loss = F.mse_loss(pred, y) + cfg.lambda_mono * a1
                else:
                    loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            scheduler.step()

        # Attach predict_mc if missing
        if not hasattr(model, "predict_mc"):
            def make_mc(m):
                def _mc(x, t, n_samples=100):
                    m.train()
                    preds = []
                    with torch.no_grad():
                        for _ in range(n_samples):
                            preds.append(m(x, t)[0])
                    stk = torch.stack(preds)
                    return stk.mean(0), stk.std(0)
                return _mc
            model.predict_mc = make_mc(model)

        res = evaluate(model, test_ds, cfg, device)
        results[name] = res
        print(f"    RMSE={res['rmse']:.4f}  R²={res['r2']:.4f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = Config(data_dir=args.data_dir, condition=args.condition,
                    epochs=args.epochs, n_seeds=args.seeds)
    device = torch.device(cfg.device)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.tbl_dir, exist_ok=True)

    banner = "=" * 55
    print(f"\n{banner}")
    print(f"  PI-XGNN — Full Figure & Table Generation")
    print(f"  Condition {cfg.condition}  |  device={device}")
    print(f"  Demo={'Yes' if args.demo else 'No'}  |  epochs={cfg.epochs}")
    print(f"{banner}")

    # ── Static tables ──────────────────────────────────────────────────────
    generate_static_tables(cfg, args.tbl_dir)
    if args.only_tables:
        print("\nDone (static tables only).")
        return

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading data  (condition {cfg.condition}) …")
    raw       = load_raw(args, cfg.condition)
    processed = preprocess_all(raw, cfg)
    if not processed:
        raise RuntimeError("No bearing data loaded.")

    # Reference bearing for EDA: Bearing_1
    ref = processed[min(processed.keys())]
    stp1, stp2 = get_stps(ref["irrms"], ref["fpt"], cfg)

    # ── EDA Figures 5–8 ────────────────────────────────────────────────────
    print(f"\n[2/6] EDA figures (5–8) …")
    generate_eda_figures(
        ref["cycles_h"], ref["cycles_v"],
        ref["feat"], ref["irrms"],
        ref["fpt"], stp1, stp2, ref["rul"],
        args.fig_dir)

    if args.only_eda:
        print("\nDone (EDA only).")
        return

    # ── Build train/test split ─────────────────────────────────────────────
    train_ids      = [i for i in range(1, cfg.n_train_bearings + 1) if i in processed]
    train_bearings = [(processed[i]["feat"], processed[i]["rul"]) for i in train_ids]
    test_bearing   = (processed[cfg.test_bearing_id]["feat"],
                      processed[cfg.test_bearing_id]["rul"])

    train_loader, test_ds, _ = build_loaders(
        train_bearings, test_bearing,
        window=cfg.window_length, batch_size=cfg.batch_size)
    t_loader, v_loader = train_val_split(train_loader)

    # ── Train PI-XGNN with history ─────────────────────────────────────────
    print(f"\n[3/6] Training PI-XGNN (1 seed for visualisation) …")
    torch.manual_seed(0)
    pi_model  = PIXGNN(cfg)
    pi_model, history = train_with_history(pi_model, t_loader, v_loader, cfg, device)

    print(f"\n[4/6] Training figures (11, 12) …")
    plot_fig11(history, args.fig_dir)
    plot_fig12({"C1": history}, args.fig_dir)      # single-condition demo

    # ── Evaluate PI-XGNN ──────────────────────────────────────────────────
    print(f"\n[5/6] Evaluating all models …")
    pi_res = evaluate(pi_model, test_ds, cfg, device)
    print(f"  PI-XGNN  RMSE={pi_res['rmse']:.4f}  PICP={pi_res['picp']:.3f}")

    # ── Baselines ─────────────────────────────────────────────────────────
    baseline_res = run_baselines(train_loader, test_ds, cfg, device)
    all_results  = {**baseline_res, "pi_xgnn": pi_res}

    # ── Ablation ──────────────────────────────────────────────────────────
    print("\n  Running ablation experiments …")
    ablation_res = run_ablation(train_loader, test_ds, cfg, device)

    # ── Window sensitivity ─────────────────────────────────────────────────
    print("\n  Running window-length sensitivity …")
    window_res = run_window_sensitivity(train_bearings, test_bearing, cfg, device)

    # ── Hyperparameter sensitivity ─────────────────────────────────────────
    print("\n  Running hyperparameter sensitivity …")
    sensitivity = run_hyperparam_sensitivity(train_loader, test_ds, cfg, device)
    # Merge window results into sensitivity dict
    sensitivity["window"] = {
        "values": list(window_res.keys()),
        "rmse":   [window_res[L]["rmse"] for L in window_res]
    }

    # ── All results figures ────────────────────────────────────────────────
    print(f"\n[6/6] Result figures (13–19, 26–28) …")

    t_ref   = processed[cfg.test_bearing_id]
    fpt_t   = t_ref["fpt"]
    stp1_t, stp2_t = get_stps(t_ref["irrms"], fpt_t, cfg)

    plot_fig13(all_results, fpt=fpt_t, stp1=stp1_t, stp2=stp2_t,
               save_dir=args.fig_dir)
    plot_fig14(pi_res, fpt=fpt_t, stp1=stp1_t, stp2=stp2_t,
               save_dir=args.fig_dir)
    plot_fig15(all_results, save_dir=args.fig_dir)
    plot_fig16({"C1": pi_res}, fpts={"C1": fpt_t},
               stp1s={"C1": stp1_t}, stp2s={"C1": stp2_t},
               save_dir=args.fig_dir)
    plot_fig17(pi_res, fpt=fpt_t, stp1=stp1_t, stp2=stp2_t,
               save_dir=args.fig_dir)
    plot_fig18({"C1": pi_model}, {"C1": test_ds}, cfg, device,
               save_dir=args.fig_dir)
    plot_fig19(pi_model, test_ds, cfg, device,
               ref["irrms"], ref["fpt"], stp1, stp2,
               save_dir=args.fig_dir)
    plot_fig26_27(ablation_res, fpt=fpt_t, stp1=stp1_t, stp2=stp2_t,
                  save_dir=args.fig_dir)
    plot_fig28(sensitivity, save_dir=args.fig_dir)

    # ── Cross-domain: PHM 2012 + NREL GRC ────────────────────────────────
    phm_results_cross = {}
    grc_results_cross = {}

    if not args.skip_cross:
        from eda.cross_domain import run_cross_domain_all
        phm_results_cross, grc_results_cross = run_cross_domain_all(
            xjtu_model   = pi_model,
            phm_data_dir = args.phm_dir,
            grc_data_dir = args.grc_dir,
            config       = cfg,
            device       = device,
            save_dir     = args.fig_dir,
            scaler       = None,   # pass fitted scaler if available
            fine_tune_epochs = args.fine_tune_epochs,
        )

    # ── Dynamic tables ─────────────────────────────────────────────────────
    # Build silhouette dict for Table 11
    sil_scores = {}
    if grc_results_cross:
        s = grc_results_cross.get("silhouette", {})
        sil_scores = {
            "PI-XGNN (Full)": (s.get("tsne", float("nan")),
                               s.get("umap", float("nan"))),
        }

    generate_result_tables(
        results_c1=all_results,
        results_c2={},
        results_c3={},
        ablation_res=ablation_res,
        window_res=window_res,
        phm_results=phm_results_cross if phm_results_cross else None,
        silhouette_scores=sil_scores if sil_scores else None,
        save_dir=args.tbl_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    figs = sorted(os.listdir(args.fig_dir))
    tbls = sorted(os.listdir(args.tbl_dir))
    print(f"\n{banner}")
    print(f"  Done!  {len(figs)} figures → {args.fig_dir}/")
    print(f"         {len(tbls)} tables  → {args.tbl_dir}/")
    print(f"{banner}")
    for f in figs: print(f"    {f}")
    for t in tbls: print(f"    {t}")


if __name__ == "__main__":
    main()
