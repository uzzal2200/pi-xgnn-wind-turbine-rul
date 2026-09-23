"""
Cross-Domain Experiments — PHM 2012 + NREL GRC

PHM 2012 (Figs 20, 21, 22 / Table 10)
  Zero-shot  : XJTU-SY trained model → PHM test bearings
  Fine-tuned : update only last 2 decoder layers on PHM learning set

NREL GRC (Figs 23, 24, 25 / Table 11)
  Zero-shot latent-space analysis : healthy vs damaged cluster separation
  Adaptive graph density evolution per configuration
  t-SNE + UMAP projections + silhouette scores
"""
import os, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# PHM 2012 experiments
# ──────────────────────────────────────────────────────────────────────────────

def fine_tune_phm(model, train_loader, config, device, epochs=50):
    """
    Fine-tune only the last two decoder layers (F_net final layers)
    on PHM 2012 training data.
    """
    # Freeze everything except F_net[-2:] and G_net[-1]
    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable = []
    # Unfreeze last 2 linear layers of F_net
    f_layers = [m for m in model.F_net.net if isinstance(m, nn.Linear)]
    for layer in f_layers[-2:]:
        for p in layer.parameters():
            p.requires_grad_(True); trainable.append(p)
    # Unfreeze last linear layer of G_net
    g_layers = [m for m in model.G_net.net if isinstance(m, nn.Linear)]
    for p in g_layers[-1].parameters():
        p.requires_grad_(True); trainable.append(p)

    optimizer = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        for x, y, t in train_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            optimizer.zero_grad()
            u, du_dt, g_pred = model(x, t)
            from engine.losses import pi_loss
            loss, _ = pi_loss(u, y, du_dt, g_pred,
                               config.lambda_pde, config.lambda_mono)
            loss.backward()
            optimizer.step()

    # Re-enable all params
    for p in model.parameters():
        p.requires_grad_(True)

    return model


def run_phm_experiments(xjtu_model, phm_raw: dict, config, device,
                          fine_tune_epochs: int = 50):
    """
    Zero-shot + fine-tuned evaluation on PHM 2012.

    Parameters
    ----------
    xjtu_model : PIXGNN trained on XJTU-SY
    phm_raw    : output of load_all_phm()
    config     : Config

    Returns
    -------
    results : {"C1": {"zero_shot": {model: res}, "fine_tuned": {model: res}}, ...}
    """
    from datasets.preprocess      import preprocess_bearing
    from datasets.bearing_dataset import build_loaders
    from engine.evaluator         import evaluate

    results = {}

    for cond_key, cond_data in phm_raw.items():          # "C1", "C2", "C3"
        print(f"\n  PHM {cond_key}")
        learn = cond_data.get("learning", {})
        test  = cond_data.get("test",     {})

        if not learn and not test:
            print(f"    No data for {cond_key}, skipping.")
            continue

        # Preprocess
        proc_learn = {}
        for bid, (h, v) in learn.items():
            feat, rul, fpt, _ = preprocess_bearing(h, v)
            proc_learn[bid]   = (feat, rul)
            print(f"    Learning Bearing_{bid}: T={len(feat)}  FPT={fpt}")

        proc_test = {}
        for bid, (h, v) in test.items():
            feat, rul, fpt, _ = preprocess_bearing(h, v)
            proc_test[bid]    = (feat, rul)
            print(f"    Test Bearing_{bid}: T={len(feat)}")

        # Use first test bearing as reference if no dedicated test data
        if not proc_test and proc_learn:
            last_bid = max(proc_learn.keys())
            proc_test = {last_bid: proc_learn.pop(last_bid)}

        if not proc_learn or not proc_test:
            continue

        # Build loaders
        train_bearings = list(proc_learn.values())
        ref_bid        = min(proc_test.keys())
        test_bearing   = proc_test[ref_bid]

        train_loader, test_ds, _ = build_loaders(
            train_bearings, test_bearing,
            window=config.window_length, batch_size=config.batch_size)

        # ── Zero-shot ──
        print(f"    Zero-shot evaluation …")
        zs_model = copy.deepcopy(xjtu_model)
        zs_res   = evaluate(zs_model, test_ds, config, device)
        print(f"    ZS RMSE={zs_res['rmse']:.4f}  PICP={zs_res['picp']:.3f}")

        # ── Fine-tuned ──
        print(f"    Fine-tuning ({fine_tune_epochs} epochs) …")
        ft_model = copy.deepcopy(xjtu_model)
        ft_model = fine_tune_phm(ft_model, train_loader, config, device,
                                  fine_tune_epochs)
        ft_res   = evaluate(ft_model, test_ds, config, device)
        print(f"    FT RMSE={ft_res['rmse']:.4f}  PICP={ft_res['picp']:.3f}")

        results[cond_key] = {
            "zero_shot":  {"pi_xgnn": zs_res},
            "fine_tuned": {"pi_xgnn": ft_res},
            "ft_model":   ft_model,
            "test_ds":    test_ds,
        }

    return results


# ── Figure 20 ─────────────────────────────────────────────────────────────────

def plot_fig20(phm_results: dict, save_dir):
    """Fig 20 — Fine-tuned PI-XGNN RUL on PHM 2012 (all conditions)."""
    conds = [k for k in phm_results if "fine_tuned" in phm_results[k]]
    n     = len(conds)
    if n == 0:
        print("  Fig 20: no PHM fine-tuned results, skipping.")
        return

    fig, axes = plt.subplots(2, n, figsize=(n * 5, 7))
    if n == 1: axes = axes.reshape(2, 1)
    fig.suptitle("Fig 20 – PI-XGNN Fine-Tuned RUL on PHM 2012", fontweight="bold")

    for j, cond in enumerate(conds):
        res = phm_results[cond]["fine_tuned"]["pi_xgnn"]
        T   = np.arange(len(res["y_true"]))

        ax_t = axes[0, j]; ax_b = axes[1, j]
        ax_t.plot(T, res["y_true"], "r-", lw=2, label="True RUL")
        ax_t.plot(T, res["mu"],     "g-", lw=1.5,
                  label=f"Pred (RMSE={res['rmse']:.4f})")
        ax_t.fill_between(T, res["lo"], res["hi"], alpha=0.25, color="green")
        ax_t.set_title(f"{cond}  R²={res['r2']:.4f}  PICP={res['picp']:.3f}",
                       fontsize=9)
        ax_t.set_ylabel("Normalised RUL"); ax_t.legend(fontsize=8)
        ax_t.set_ylim(-0.05, 1.05)

        err = np.abs(res["mu"] - res["y_true"])
        ax_b.bar(T, err, color="steelblue", alpha=0.6, width=1)
        sig = (res["hi"] - res["lo"]) / 2
        ax_b.plot(T, sig * 1.96, "r-", lw=1)
        ax_b.set_xlabel("Sampling cycle"); ax_b.set_ylabel("|Error|")

    plt.tight_layout()
    _save(fig, save_dir, "fig20_phm_rul.png")


# ── Figure 21 ─────────────────────────────────────────────────────────────────

def plot_fig21(phm_results: dict, save_dir):
    """Fig 21 — RMSE reduction (%) by fine-tuning vs zero-shot."""
    conds = [k for k in phm_results
             if "zero_shot" in phm_results[k] and "fine_tuned" in phm_results[k]]
    if not conds:
        print("  Fig 21: no data, skipping.")
        return

    zs_rmse = [phm_results[c]["zero_shot"]["pi_xgnn"]["rmse"]  for c in conds]
    ft_rmse = [phm_results[c]["fine_tuned"]["pi_xgnn"]["rmse"] for c in conds]
    pct_red = [100 * (z - f) / z for z, f in zip(zs_rmse, ft_rmse)]

    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Fig 21 – RMSE Reduction (%) by Fine-Tuning", fontweight="bold")

    bars = ax.bar(x, pct_red, color=["#2196F3", "#4CAF50", "#FF9800"][:len(conds)],
                  alpha=0.85, width=0.5)
    for bar, val, zs, ft in zip(bars, pct_red, zs_rmse, ft_rmse):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%\n(ZS={zs:.4f}→FT={ft:.4f})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("RMSE reduction (%)"); ax.set_xlabel("PHM Condition")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_dir, "fig21_phm_rmse_reduction.png")


# ── Figure 22 ─────────────────────────────────────────────────────────────────

def plot_fig22(phm_results: dict, config, device, save_dir):
    """Fig 22 — Calibration reliability diagrams for PHM 2012 (fine-tuned)."""
    from engine.evaluator import calibrate_z

    conds = [k for k in phm_results if "ft_model" in phm_results[k]]
    n     = len(conds)
    if n == 0:
        print("  Fig 22: no data, skipping.")
        return

    nominal = np.linspace(0.50, 0.95, 15)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1: axes = [axes]
    fig.suptitle("Fig 22 – Calibration Reliability Diagrams (PHM 2012 Fine-Tuned)",
                 fontweight="bold")

    for ax, cond in zip(axes, conds):
        model   = phm_results[cond]["ft_model"]
        test_ds = phm_results[cond]["test_ds"]
        loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

        all_mu, all_sig, all_y = [], [], []
        for x, y, t in loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            mu, sig  = model.predict_mc(x, t, n_samples=50)
            all_mu.append(mu.cpu().numpy())
            all_sig.append(sig.cpu().numpy())
            all_y.append(y.cpu().numpy())

        mu  = np.concatenate(all_mu).ravel()
        sig = np.concatenate(all_sig).ravel()
        y   = np.concatenate(all_y).ravel()

        empirical = []
        for level in nominal:
            z  = calibrate_z(mu, sig, y, target=level)
            lo = np.clip(mu - z * sig, 0, 1)
            hi = np.clip(mu + z * sig, 0, 1)
            empirical.append(np.mean((y >= lo) & (y <= hi)))

        picp_95 = empirical[-1]
        ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=1.5, label="Perfect")
        ax.plot(nominal, empirical, "o-", color="blue", lw=1.5,
                label=f"PI-XGNN (PICP={picp_95:.3f})", ms=4)
        ax.set_title(f"{cond}  PICP@95%={picp_95:.3f}", fontsize=9)
        ax.set_xlabel("Nominal level"); ax.set_ylabel("Empirical PICP")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(0.45, 1.0); ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    _save(fig, save_dir, "fig22_phm_calibration.png")


# ──────────────────────────────────────────────────────────────────────────────
# NREL GRC experiments
# ──────────────────────────────────────────────────────────────────────────────

def _extract_latent(model, cycles_h, cycles_v, config, device,
                    scaler=None, proxy_start=1.0, proxy_end=0.0):
    """
    Run the encoder on GRC cycles and return latent codes + proxy RUL.

    Parameters
    ----------
    proxy_start, proxy_end : RUL values assigned to first/last cycle
                             (1.0→0.0 for damaged, 1.0→1.0 for healthy)
    """
    from datasets.preprocess      import preprocess_bearing
    from datasets.bearing_dataset import BearingDataset

    feat, _, fpt, _ = preprocess_bearing(
        cycles_h, cycles_v,
        alpha=config.irrms_alpha, eta=config.fpt_eta)

    T   = len(feat)
    rul = np.linspace(proxy_start, proxy_end, T).astype(np.float32)

    if scaler is not None:
        feat = scaler.transform(feat).astype(np.float32)

    ds     = BearingDataset(feat, rul, window=config.window_length)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    latents, proxy_ruls = [], []
    model.eval()
    with torch.no_grad():
        for x, y, t in loader:
            x = x.to(device)
            h = model._encode(x)
            latents.append(h.cpu().numpy())
            proxy_ruls.append(y.numpy())

    return np.concatenate(latents), np.concatenate(proxy_ruls).ravel()


def run_grc_experiments(xjtu_model, grc_raw: dict, config, device, scaler=None):
    """
    Zero-shot GRC domain validation.

    Parameters
    ----------
    grc_raw : output of load_all_grc()  {"healthy": (h, v, fs), "damaged": (...)}

    Returns
    -------
    grc_results : {
        "healthy_latents": np.ndarray  (T_h, 8)
        "damaged_latents": np.ndarray  (T_d, 8)
        "healthy_rul":     np.ndarray  (T_h,)
        "damaged_rul":     np.ndarray  (T_d,)
        "graph_density":   {"healthy": [...], "damaged": [...]}
        "silhouette":      {"tsne": float, "umap": float, "pca": float}
    }
    """
    latents, proxy_ruls = {}, {}

    for cfg_name, (h, v, fs) in grc_raw.items():
        print(f"  GRC {cfg_name}: extracting latent codes …")
        p_start = 1.0
        p_end   = 1.0 if cfg_name == "healthy" else 0.0
        Z, rul  = _extract_latent(xjtu_model, h, v, config, device,
                                   scaler=scaler,
                                   proxy_start=p_start, proxy_end=p_end)
        latents[cfg_name]    = Z
        proxy_ruls[cfg_name] = rul
        print(f"    {Z.shape[0]} windows  latent_dim={Z.shape[1]}")

    # Graph density evolution
    graph_density = _compute_graph_density(xjtu_model, grc_raw, config, device, scaler)

    # Silhouette scores
    silhouette = _compute_silhouette(latents)

    return {
        "healthy_latents": latents.get("healthy"),
        "damaged_latents": latents.get("damaged"),
        "healthy_rul":     proxy_ruls.get("healthy"),
        "damaged_rul":     proxy_ruls.get("damaged"),
        "graph_density":   graph_density,
        "silhouette":      silhouette,
    }


def _compute_graph_density(model, grc_raw, config, device, scaler):
    """Compute edge density per degradation stage for each GRC config."""
    from datasets.preprocess import preprocess_bearing
    density = {}

    for cfg_name, (h, v, fs) in grc_raw.items():
        feat, _, fpt, irrms = preprocess_bearing(h, v)
        if scaler: feat = scaler.transform(feat).astype(np.float32)

        T = len(feat)
        stp = fpt + (T - fpt) // 2

        stage_dens = []
        for stage_idx in [fpt // 2, (fpt + stp) // 2, min(stp + 5, T - 1)]:
            window = feat[max(0, stage_idx-2): stage_idx+3]
            x = torch.FloatTensor(window).unsqueeze(0).to(device)
            with torch.no_grad():
                tau = float(torch.sigmoid(model.graph.tau).item())
                C   = (x.transpose(1, 2) @ x).abs() / x.shape[1]
                I   = torch.eye(12, device=device).unsqueeze(0)
                A   = ((C - tau) > 0).float() * (1 - I) + I
                d   = float((A.sum(-1) / 12).mean().item())
            stage_dens.append(d)
        density[cfg_name] = stage_dens

    return density


def _compute_silhouette(latents: dict):
    """Compute silhouette scores under PCA, t-SNE, UMAP projections."""
    if "healthy" not in latents or "damaged" not in latents:
        return {}

    H = latents["healthy"]
    D = latents["damaged"]
    n = min(500, len(H), len(D))    # subsample for speed
    idx_h = np.random.choice(len(H), n, replace=False)
    idx_d = np.random.choice(len(D), n, replace=False)
    Z     = np.vstack([H[idx_h], D[idx_d]])
    y     = np.array([0] * n + [1] * n)

    scores = {}

    # PCA
    try:
        Z_pca = PCA(n_components=3).fit_transform(Z)
        scores["pca"] = float(silhouette_score(Z_pca, y))
    except Exception:
        scores["pca"] = float("nan")

    # t-SNE
    if HAS_TSNE:
        try:
            Z_tsne = TSNE(n_components=2, perplexity=min(30, n-1),
                          random_state=42).fit_transform(Z)
            scores["tsne"] = float(silhouette_score(Z_tsne, y))
        except Exception:
            scores["tsne"] = float("nan")
    else:
        scores["tsne"] = float("nan")
        print("  Note: install scikit-learn for t-SNE silhouette.")

    # UMAP
    if HAS_UMAP:
        try:
            Z_umap = UMAP(n_components=2, random_state=42).fit_transform(Z)
            scores["umap"] = float(silhouette_score(Z_umap, y))
        except Exception:
            scores["umap"] = float("nan")
    else:
        scores["umap"] = float("nan")
        print("  Note: install umap-learn for UMAP silhouette: pip install umap-learn")

    print(f"  Silhouette: PCA={scores['pca']:.3f}  "
          f"t-SNE={scores['tsne']:.3f}  UMAP={scores['umap']:.3f}")
    return scores


# ── Figure 23 ─────────────────────────────────────────────────────────────────

def plot_fig23(grc_results: dict, save_dir):
    """Fig 23 — PCA latent-space visualisation: healthy vs damaged (GRC)."""
    H   = grc_results.get("healthy_latents")
    D   = grc_results.get("damaged_latents")
    if H is None or D is None:
        print("  Fig 23: no GRC latents, skipping.")
        return

    n   = min(300, len(H), len(D))
    H_s = H[np.random.choice(len(H), n, replace=False)]
    D_s = D[np.random.choice(len(D), n, replace=False)]
    Z   = np.vstack([H_s, D_s])
    rul = np.hstack([grc_results["healthy_rul"][:n],
                     grc_results["damaged_rul"][:n]])

    pca   = PCA(n_components=3).fit_transform(Z)
    label = np.array(["Healthy"] * n + ["Damaged"] * n)

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("Fig 23 – Latent Feature Visualisation on NREL GRC (PCA)",
                 fontweight="bold")
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2],
                     c=rul, cmap="RdYlBu", alpha=0.6, s=20,
                     marker=["o"  if l == "Healthy" else "^"
                             for l in label])
    plt.colorbar(sc, ax=ax, label="Proxy RUL (red=low, blue=high)", shrink=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    from matplotlib.lines import Line2D
    legend_elem = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
               markersize=8, label="Healthy"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="grey",
               markersize=8, label="Damaged"),
    ]
    ax.legend(handles=legend_elem, loc="upper right")
    plt.tight_layout()
    _save(fig, save_dir, "fig23_grc_pca_latent.png")


# ── Figure 24 ─────────────────────────────────────────────────────────────────

def plot_fig24(grc_results: dict, save_dir):
    """Fig 24 — Adaptive graph density evolution on NREL GRC."""
    density = grc_results.get("graph_density", {})
    if not density:
        print("  Fig 24: no graph density data, skipping.")
        return

    stages = ["Healthy", "Moderate", "Severe"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 24 – Adaptive Graph Evolution on NREL GRC", fontweight="bold")

    colors = {"healthy": "blue", "damaged": "red"}
    for cfg_name, dens in density.items():
        x   = np.arange(len(dens))
        col = colors.get(cfg_name, "grey")
        ax1.plot(x, dens, "o-", color=col, lw=2, ms=8, label=cfg_name.capitalize())
        for xi, yi in zip(x, dens):
            ax1.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", fontsize=8, color=col)

    ax1.set_xticks(range(len(stages))); ax1.set_xticklabels(stages)
    ax1.set_ylabel("Graph edge density"); ax1.set_title("(a) Edge density")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Bar chart of density per stage
    x  = np.arange(len(stages)); w = 0.35
    if "healthy" in density and "damaged" in density:
        ax2.bar(x - w/2, density["healthy"], w, label="Healthy", color="blue",  alpha=0.7)
        ax2.bar(x + w/2, density["damaged"], w, label="Damaged", color="red",   alpha=0.7)
        ax2.set_xticks(x); ax2.set_xticklabels(stages)
        ax2.set_ylabel("Edge density"); ax2.set_title("(b) Density comparison")
        ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, "fig24_grc_graph_density.png")


# ── Figure 25 ─────────────────────────────────────────────────────────────────

def plot_fig25(grc_results: dict, save_dir):
    """Fig 25 — t-SNE + UMAP projections of GRC latent codes."""
    H = grc_results.get("healthy_latents")
    D = grc_results.get("damaged_latents")
    if H is None or D is None:
        print("  Fig 25: no GRC latents, skipping.")
        return

    n    = min(300, len(H), len(D))
    H_s  = H[np.random.choice(len(H), n, replace=False)]
    D_s  = D[np.random.choice(len(D), n, replace=False)]
    Z    = np.vstack([H_s, D_s])
    rul  = np.hstack([grc_results["healthy_rul"][:n],
                      grc_results["damaged_rul"][:n]])
    labs = np.array([0] * n + [1] * n)   # 0=healthy, 1=damaged

    projections = {}
    if HAS_TSNE:
        try:
            projections["t-SNE"] = TSNE(n_components=2, perplexity=min(30, n-1),
                                         random_state=42).fit_transform(Z)
        except Exception as e:
            print(f"  t-SNE failed: {e}")
    if HAS_UMAP:
        try:
            projections["UMAP"] = UMAP(n_components=2, random_state=42).fit_transform(Z)
        except Exception as e:
            print(f"  UMAP failed: {e}")

    if not projections:
        print("  Fig 25: neither t-SNE nor UMAP available, skipping.")
        return

    n_proj = len(projections)
    fig, axes = plt.subplots(1, n_proj, figsize=(n_proj * 5, 4))
    if n_proj == 1: axes = [axes]
    fig.suptitle("Fig 25 – t-SNE / UMAP Projections of PI-XGNN Latent Codes (NREL GRC)",
                 fontweight="bold")

    sil_scores = grc_results.get("silhouette", {})
    markers    = {0: "o", 1: "^"}
    labels_str = {0: "Healthy", 1: "Damaged"}

    for ax, (proj_name, Z_2d) in zip(axes, projections.items()):
        for lab in [0, 1]:
            mask = labs == lab
            sc   = ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                              c=rul[mask], cmap="RdYlBu",
                              marker=markers[lab], alpha=0.6, s=20,
                              label=labels_str[lab])
        sil_key = proj_name.lower().replace("-", "")
        sil_val = sil_scores.get(sil_key, float("nan"))
        ax.set_title(f"{proj_name}  Silhouette={sil_val:.3f}", fontsize=9)
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="Proxy RUL", shrink=0.7)

    plt.tight_layout()
    _save(fig, save_dir, "fig25_grc_tsne_umap.png")


# ──────────────────────────────────────────────────────────────────────────────
# Master runner
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_domain_all(xjtu_model, phm_data_dir, grc_data_dir,
                          config, device, save_dir="./figures",
                          scaler=None,
                          fine_tune_epochs=50):
    """
    Run all cross-domain experiments and generate Figs 20–25 + Tables 10–11.

    Parameters
    ----------
    xjtu_model    : PIXGNN trained on XJTU-SY
    phm_data_dir  : path to PHM 2012 root (contains Learning_set/)
    grc_data_dir  : path to NREL GRC root (contains Healthy/ and Damaged/)
    scaler        : StandardScaler fitted on XJTU-SY training data
    """
    phm_results = {}
    grc_results = {}

    # ── PHM 2012 ──────────────────────────────────────────────────────────────
    if phm_data_dir and os.path.isdir(phm_data_dir):
        print("\n[PHM 2012] Loading dataset …")
        from datasets.phm2012 import load_all_phm
        phm_raw     = load_all_phm(phm_data_dir)
        phm_results = run_phm_experiments(xjtu_model, phm_raw, config, device,
                                           fine_tune_epochs)
        print("\n[PHM 2012] Generating figures 20–22 …")
        plot_fig20(phm_results, save_dir)
        plot_fig21(phm_results, save_dir)
        plot_fig22(phm_results, config, device, save_dir)
    else:
        print(f"\n[PHM 2012] Skipped — directory not found: {phm_data_dir}")

    # ── NREL GRC ──────────────────────────────────────────────────────────────
    if grc_data_dir and os.path.isdir(grc_data_dir):
        print("\n[NREL GRC] Loading dataset …")
        from datasets.nrel_grc import load_all_grc
        grc_raw     = load_all_grc(grc_data_dir)
        grc_results = run_grc_experiments(xjtu_model, grc_raw, config, device, scaler)
        print("\n[NREL GRC] Generating figures 23–25 …")
        plot_fig23(grc_results, save_dir)
        plot_fig24(grc_results, save_dir)
        plot_fig25(grc_results, save_dir)
    else:
        print(f"\n[NREL GRC] Skipped — directory not found: {grc_data_dir}")

    return phm_results, grc_results
