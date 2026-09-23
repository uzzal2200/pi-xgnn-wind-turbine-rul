"""
Result Figures (Section V of paper)

Fig 13 : RUL prediction trajectories — all 8 methods on Bearing1_5
Fig 14 : PI-XGNN prediction detail with 95% CI
Fig 15 : Absolute error distributions (box plots)
Fig 16 : PI-XGNN 3-condition evaluation with CI
Fig 17 : Epistemic uncertainty dynamics over bearing lifetime
Fig 18 : Calibration reliability diagrams (nominal vs empirical PICP)
Fig 19 : Gradient-based node importance attribution (3 stages)
Fig 26 : Ablation RUL prediction curves
Fig 27 : Ablation component contribution bar chart
Fig 28 : Hyperparameter sensitivity (4 panels)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats

FEATURE_NAMES = ["RMS", "SRA", "Peak", "P2P", "Crest",
                 "Kurtosis", "Skewness", "Impulse", "Waveform",
                 "Mean Freq", "Std Freq", "IRRMS"]


def _save(fig, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Figure 13 ─────────────────────────────────────────────────────────────────

def plot_fig13(all_results: dict, fpt=None, stp1=None, stp2=None, save_dir="./figures"):
    """
    Fig 13 — RUL prediction trajectories for all models.
    all_results : {model_name: {mu, y_true, ...}}
    """
    n_models = len(all_results)
    cols = 4; rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle("Fig 13 – RUL Prediction Trajectories", fontweight="bold", fontsize=12)
    axes = np.array(axes).ravel()

    for i, (name, res) in enumerate(all_results.items()):
        ax = axes[i]
        T  = np.arange(len(res["y_true"]))
        ax.plot(T, res["y_true"], "r-", lw=2, label="True RUL")
        ax.plot(T, res["mu"],     "b-", lw=1.2, label=f"Pred (RMSE={res['rmse']:.4f})")

        if "lo" in res and name == list(all_results.keys())[-1]:  # PI-XGNN last
            ax.fill_between(T, res["lo"], res["hi"], alpha=0.25, color="green",
                            label="95% CI")

        # Stage lines
        if fpt:  ax.axvline(fpt,  color="blue",   ls="--", lw=0.8)
        if stp1: ax.axvline(stp1, color="orange", ls="--", lw=0.8)
        if stp2: ax.axvline(stp2, color="red",    ls="--", lw=0.8)

        ax.set_title(name.upper(), fontsize=9)
        ax.set_xlabel("Sampling point", fontsize=8)
        ax.set_ylabel("RUL", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6)

        # Signed error subplot
        ax_err = ax.twinx()
        ax_err.fill_between(T, 0, res["mu"] - res["y_true"],
                            alpha=0.3, color="purple")
        ax_err.set_ylabel("Error", fontsize=6, color="purple")
        ax_err.tick_params(axis="y", labelcolor="purple", labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, save_dir, "fig13_rul_comparison.png")


# ── Figure 14 ─────────────────────────────────────────────────────────────────

def plot_fig14(res: dict, fpt=None, stp1=None, stp2=None, save_dir="./figures"):
    """Fig 14 — PI-XGNN detail: prediction + CI + error + 1.96σ envelope."""
    T   = np.arange(len(res["y_true"]))
    sig = (res["hi"] - res["lo"]) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Fig 14 – PI-XGNN Prediction Detail  "
        f"(RMSE={res['rmse']:.4f}, PICP={res['picp']:.3f}, MPIW={res['mpiw']:.4f})",
        fontweight="bold", fontsize=11)

    ax1.plot(T, res["y_true"], "r-",  lw=2,   label="True RUL")
    ax1.plot(T, res["mu"],     "g-",  lw=1.5, label="PI-XGNN mean")
    ax1.fill_between(T, res["lo"], res["hi"], alpha=0.3, color="green",
                     label="95% CI")
    for v, c, l in [(fpt, "blue", "FPT"), (stp1, "orange", "STP₁"),
                     (stp2, "red", "STP₂")]:
        if v: ax1.axvline(v, color=c, ls="--", lw=1.2, label=l)
    ax1.set_ylabel("Normalised RUL"); ax1.legend(fontsize=9); ax1.set_ylim(-0.05, 1.05)

    ax2.bar(T, np.abs(res["mu"] - res["y_true"]), color="grey", alpha=0.6, label="|Error|")
    ax2.plot(T, sig * 1.96, "r-", lw=1, label="1.96σ")
    ax2.set_ylabel("|Error|"); ax2.set_xlabel("Sampling cycle")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, save_dir, "fig14_pixgnn_detail.png")


# ── Figure 15 ─────────────────────────────────────────────────────────────────

def plot_fig15(all_results: dict, save_dir="./figures"):
    """Fig 15 — Absolute error distribution box plots for all methods."""
    names  = list(all_results.keys())
    errors = [np.abs(r["mu"] - r["y_true"]) for r in all_results.values()]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Fig 15 – Absolute Error Distributions", fontweight="bold")

    bp = ax.boxplot(errors, labels=names, patch_artist=True, notch=False,
                    medianprops=dict(color="red", lw=2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.6)

    # PI-XGNN RMSE reference line
    if "pi_xgnn" in all_results:
        rmse_pi = all_results["pi_xgnn"]["rmse"]
        ax.axhline(rmse_pi, color="red", ls="--", lw=1.5,
                   label=f"PI-XGNN RMSE={rmse_pi:.4f}")
    ax.set_xlabel("Method"); ax.set_ylabel("Absolute error")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_dir, "fig15_error_boxplot.png")


# ── Figure 16 ─────────────────────────────────────────────────────────────────

def plot_fig16(results_per_cond: dict, fpts=None, stp1s=None, stp2s=None,
               save_dir="./figures"):
    """
    Fig 16 — PI-XGNN 3-condition evaluation.
    results_per_cond : {"C1": res_dict, "C2": ..., "C3": ...}
    """
    n = len(results_per_cond)
    fig, axes = plt.subplots(2, n, figsize=(n * 5, 7))
    fig.suptitle("Fig 16 – PI-XGNN Across All Three XJTU-SY Conditions",
                 fontweight="bold")

    for j, (cond, res) in enumerate(results_per_cond.items()):
        T = np.arange(len(res["y_true"]))
        ax_top = axes[0, j] if n > 1 else axes[0]
        ax_bot = axes[1, j] if n > 1 else axes[1]

        ax_top.plot(T, res["y_true"], "r-", lw=2, label="True RUL")
        ax_top.plot(T, res["mu"],     "g-", lw=1.5, label="Predicted")
        ax_top.fill_between(T, res["lo"], res["hi"], alpha=0.25, color="green")
        ax_top.set_title(
            f"{cond}  RMSE={res['rmse']:.4f}  R²={res['r2']:.4f}  "
            f"PICP={res['picp']:.3f}", fontsize=9)
        ax_top.set_ylabel("RUL"); ax_top.legend(fontsize=8); ax_top.set_ylim(-0.05, 1.05)

        for v_dict, vl, vc in [(fpts, "FPT", "blue"), (stp1s, "STP₁", "orange"),
                                (stp2s, "STP₂", "red")]:
            if v_dict and cond in v_dict:
                ax_top.axvline(v_dict[cond], color=vc, ls="--", lw=1, label=vl)

        err = np.abs(res["mu"] - res["y_true"])
        sig = (res["hi"] - res["lo"]) / 2
        ax_bot.bar(T, err, color="steelblue", alpha=0.6, width=1)
        ax_bot.plot(T, sig * 1.96, "r-", lw=1, label="1.96σ")
        ax_bot.set_xlabel("Sampling cycle"); ax_bot.set_ylabel("|Error|")
        ax_bot.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, save_dir, "fig16_multicondition.png")


# ── Figure 17 ─────────────────────────────────────────────────────────────────

def plot_fig17(res: dict, fpt=None, stp1=None, stp2=None, save_dir="./figures"):
    """Fig 17 — Epistemic uncertainty dynamics (CI width over bearing lifetime)."""
    T      = np.arange(len(res["y_true"]))
    ci_w   = res["hi"] - res["lo"]
    mean_w = float(np.mean(ci_w))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Fig 17 – Epistemic Uncertainty Dynamics", fontweight="bold")

    ax1.plot(T, res["y_true"], "r-", lw=2, label="True RUL")
    ax1.plot(T, res["mu"],     "b-", lw=1.5, label="Predicted mean")
    ax1.fill_between(T, res["lo"], res["hi"], alpha=0.3, color="blue", label="95% CI")
    for v, c, l in [(fpt, "blue", "FPT"), (stp1, "orange", "STP₁"),
                     (stp2, "red",    "STP₂")]:
        if v: ax1.axvline(v, color=c, ls="--", lw=1.2, label=l)
    ax1.set_ylabel("RUL"); ax1.legend(fontsize=8); ax1.set_ylim(-0.05, 1.05)

    ax2.plot(T, ci_w, color="purple", lw=1.5)
    ax2.axhline(mean_w, color="grey", ls=":", lw=1.2,
                label=f"Mean MPIW = {mean_w:.4f}")
    if stp1: ax2.axvline(stp1, color="orange", ls="--", lw=1)
    if stp2: ax2.axvline(stp2, color="red",    ls="--", lw=1)
    ax2.set_ylabel("CI width (2z*σ̂)"); ax2.set_xlabel("Sampling cycle")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, save_dir, "fig17_uncertainty_dynamics.png")


# ── Figure 18 ─────────────────────────────────────────────────────────────────

def plot_fig18(models_per_cond: dict, test_ds_per_cond: dict,
               config, device, save_dir="./figures"):
    """
    Fig 18 — Calibration reliability diagrams: empirical PICP vs nominal level.
    models_per_cond : {"C1": model, ...}
    test_ds_per_cond: {"C1": test_dataset, ...}
    """
    from torch.utils.data import DataLoader

    nominal_levels = np.linspace(0.50, 0.95, 20)
    n  = len(models_per_cond)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1: axes = [axes]
    fig.suptitle("Fig 18 – Calibration Reliability Diagrams", fontweight="bold")

    for ax, (cond, model) in zip(axes, models_per_cond.items()):
        test_ds = test_ds_per_cond[cond]
        loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

        all_mu, all_sig, all_y = [], [], []
        for x, y, t in loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            mu, sig = model.predict_mc(x, t, n_samples=50)
            all_mu.append(mu.cpu().numpy())
            all_sig.append(sig.cpu().numpy())
            all_y.append(y.cpu().numpy())

        mu   = np.concatenate(all_mu).ravel()
        sig  = np.concatenate(all_sig).ravel()
        y    = np.concatenate(all_y).ravel()

        empirical = []
        for level in nominal_levels:
            from engine.evaluator import calibrate_z
            z  = calibrate_z(mu, sig, y, target=level)
            lo = np.clip(mu - z * sig, 0, 1)
            hi = np.clip(mu + z * sig, 0, 1)
            empirical.append(np.mean((y >= lo) & (y <= hi)))

        ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=1.5, label="Perfect calibration")
        ax.plot(nominal_levels, empirical, "o-", color="blue", lw=1.5,
                label="PI-XGNN (Full)", ms=4)
        picp_95 = empirical[-1]
        ax.set_title(f"{cond}  (PICP@95%={picp_95:.3f})", fontsize=9)
        ax.set_xlabel("Nominal coverage level")
        ax.set_ylabel("Empirical PICP")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(0.45, 1.0); ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    _save(fig, save_dir, "fig18_reliability_diagram.png")


# ── Figure 19 ─────────────────────────────────────────────────────────────────

def plot_fig19(model, test_ds, config, device, irrms, fpt, stp1, stp2,
               save_dir="./figures"):
    """
    Fig 19 — Gradient-based node importance at 3 degradation stages.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    all_x, all_t = [], []
    for x, _, t in loader:
        all_x.append(x); all_t.append(t)
    X_all = torch.cat(all_x).to(device)
    T_all = torch.cat(all_t).to(device)

    T      = len(test_ds) + config.window_length - 1  # approx
    h_end  = fpt
    m_end  = stp1
    s_end  = len(test_ds)

    stage_slices = {
        "Healthy":   slice(0, max(1, h_end)),
        "Moderate":  slice(h_end, max(h_end + 1, m_end)),
        "Severe":    slice(m_end, s_end),
    }

    imp_per_stage = {}
    for stage, sl in stage_slices.items():
        x_s = X_all[sl]
        t_s = T_all[sl]
        if len(x_s) == 0:
            imp_per_stage[stage] = np.ones(12) / 12
            continue
        imp = model.node_importance(x_s, t_s, n_samples=20).cpu().numpy()
        imp_per_stage[stage] = imp

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Fig 19 – Gradient-Based Node Importance Attribution",
                 fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#F44336"]
    for ax, (stage, imp), col in zip(axes, imp_per_stage.items(), colors):
        top3  = np.argsort(imp)[-3:]
        bar_c = [col if i in top3 else "#BDBDBD" for i in range(12)]
        ax.barh(FEATURE_NAMES, imp, color=bar_c)
        ax.axvline(1 / 12, color="red", ls="--", lw=1.2,
                   label=f"Uniform (1/12 = {1/12:.3f})")
        ax.set_title(stage, fontsize=10)
        ax.set_xlabel("Importance score")
        ax.legend(fontsize=8)
        for i, v in enumerate(imp):
            ax.text(v + 0.002, i, f"{v:.2f}", va="center", fontsize=7)

    plt.tight_layout()
    _save(fig, save_dir, "fig19_feature_importance.png")


# ── Figure 26 & 27 ────────────────────────────────────────────────────────────

def plot_fig26_27(ablation_results: dict, fpt=None, stp1=None, stp2=None,
                  save_dir="./figures"):
    """
    Fig 26 — Ablation RUL prediction curves.
    Fig 27 — Ablation component bar chart.
    ablation_results : {"No-Graph": res, "No-Physics": res, "No-PDE": res, "PI-XGNN": res}
    """
    # ── Fig 26 ──
    n = len(ablation_results)
    fig26, axes = plt.subplots(2, n, figsize=(n * 4, 6))
    fig26.suptitle("Fig 26 – Ablation RUL Prediction Curves", fontweight="bold")

    colors = {"No-Graph": "red", "No-Physics": "orange",
              "No-PDE": "purple", "PI-XGNN": "green"}

    for j, (name, res) in enumerate(ablation_results.items()):
        T  = np.arange(len(res["y_true"]))
        c  = colors.get(name, "blue")
        ax_t = axes[0, j] if n > 1 else axes[0]
        ax_b = axes[1, j] if n > 1 else axes[1]

        ax_t.plot(T, res["y_true"], "r-", lw=2, label="True RUL")
        ax_t.plot(T, res["mu"],      "-", color=c, lw=1.5,
                  label=f"Pred (RMSE={res['rmse']:.4f})")
        if "lo" in res and name == "PI-XGNN":
            ax_t.fill_between(T, res["lo"], res["hi"], alpha=0.25, color="green")
        for v, vc in [(fpt, "blue"), (stp1, "orange"), (stp2, "red")]:
            if v: ax_t.axvline(v, color=vc, ls="--", lw=0.8)
        ax_t.set_title(name, fontsize=9); ax_t.set_ylim(-0.05, 1.05)
        ax_t.legend(fontsize=7)

        ax_b.fill_between(T, 0, res["mu"] - res["y_true"], alpha=0.5, color=c)
        ax_b.axhline(0, color="black", lw=0.8)
        ax_b.set_xlabel("Sampling point"); ax_b.set_ylabel("Signed error")

    plt.tight_layout()
    _save(fig26, save_dir, "fig26_ablation_rul.png")

    # ── Fig 27 ──
    names  = list(ablation_results.keys())
    rmses  = [r["rmse"] for r in ablation_results.values()]
    maes   = [r["mae"]  for r in ablation_results.values()]
    r2s    = [r["r2"]   for r in ablation_results.values()]

    x      = np.arange(len(names))
    width  = 0.28

    fig27, axes27 = plt.subplots(1, 3, figsize=(13, 4))
    fig27.suptitle("Fig 27 – Ablation Component Contribution", fontweight="bold")

    for ax, vals, ylabel, lower_better in zip(
            axes27, [rmses, maes, r2s],
            ["RMSE ↓", "MAE ↓", "R² ↑"],
            [True, True, False]):
        bars = ax.bar(x, vals, width=0.6,
                      color=["#EF5350", "#FF9800", "#AB47BC", "#66BB6A"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig27, save_dir, "fig27_ablation_bar.png")


# ── Figure 28 ─────────────────────────────────────────────────────────────────

def plot_fig28(sensitivity_results: dict, save_dir="./figures"):
    """
    Fig 28 — Hyperparameter sensitivity (4 panels).
    sensitivity_results : {
        "lambda_pde":  {"values": [...], "rmse": [...]},
        "lambda_mono": {"values": [...], "rmse": [...]},
        "dropout":     {"values": [...], "rmse": [...]},
        "window":      {"values": [...], "rmse": [...]},
    }
    """
    params = {
        "lambda_pde":  ("λ_PDE", 0.15),
        "lambda_mono": ("λ_mono", 0.15),
        "dropout":     ("Dropout rate p", 0.20),
        "window":      ("Window length L", 5),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Fig 28 – Hyperparameter Sensitivity of PI-XGNN",
                 fontweight="bold")

    for ax, (key, (label, best_val)) in zip(axes, params.items()):
        if key not in sensitivity_results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        vals = sensitivity_results[key]["values"]
        rmse = sensitivity_results[key]["rmse"]
        best_rmse = min(rmse)
        band = best_rmse * 1.05   # ±5% band

        ax.plot(vals, rmse, "o-", color="steelblue", lw=1.5, ms=6)
        ax.axhline(band, color="grey", ls="--", lw=1, label="±5% of min RMSE")
        ax.axvline(best_val, color="red", ls="--", lw=1.5, label=f"Selected: {best_val}")
        ax.scatter([best_val], [best_rmse], marker="*", s=150, color="red", zorder=5)
        ax.set_xlabel(label); ax.set_ylabel("RMSE")
        ax.set_title(label); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, "fig28_sensitivity.png")
