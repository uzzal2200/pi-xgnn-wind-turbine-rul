"""
Training Visualization (Section V-D of paper)

Fig 11 : Training + validation loss curves (total, L_data, L_PDE, L_mono)
Fig 12 : Learnable adjacency threshold σ(τ) evolution + graph edge density

These figures need a modified trainer that tracks history.
Use train_with_history() from this module instead of the standard trainer.
"""
import os, copy
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.losses import pi_loss


# ── History-tracking trainer ──────────────────────────────────────────────────

def train_with_history(model, train_loader, val_loader, config, device):
    """
    Train one seed and record per-epoch metrics for plotting.

    Returns
    -------
    model   : trained model
    history : dict  {train_total, val_total, train_data, train_pde, train_mono,
                     tau_c1, tau_c2, tau_c3, density}
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),
                             lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    hist = {k: [] for k in ("train_total", "val_total",
                             "L_data", "L_pde", "L_mono", "tau", "density")}

    for epoch in range(config.epochs):
        # ── train ──
        model.train()
        t_tot = t_dat = t_pde = t_mon = 0.0; nb = 0
        for x, y, t in train_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            optimizer.zero_grad()
            u, du_dt, g_pred = model(x, t)
            loss, parts = pi_loss(u, y, du_dt, g_pred,
                                   config.lambda_pde, config.lambda_mono)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            t_tot += loss.item(); t_dat += parts["data"]
            t_pde += parts["pde"]; t_mon += parts["mono"]
            nb += 1
        scheduler.step()

        # ── val (data loss only — no PDE autograd needed) ──
        model.eval(); v_tot = 0.0; vb = 0
        with torch.no_grad():
            for x, y, t in val_loader:
                x, y, t = x.to(device), y.to(device), t.to(device)
                u_val = model.predict(x, t)
                v_tot += torch.nn.functional.mse_loss(u_val, y).item()
                vb += 1
        model.train()

        # ── tau and density ──
        tau_val  = float(torch.sigmoid(model.graph.tau).item())
        # Edge density using current tau
        with torch.no_grad():
            sample_x = next(iter(train_loader))[0][:8].to(device)
            g_out    = model.graph(sample_x)
            C        = (sample_x.transpose(1, 2) @ sample_x).abs() / 5
            I        = torch.eye(12, device=device).unsqueeze(0)
            A        = ((C - tau_val) > 0).float() * (1 - I) + I
            density  = float((A.sum(-1) / 12).mean().item())

        hist["train_total"].append(t_tot / max(nb, 1))
        hist["val_total"].append(v_tot / max(vb, 1))
        hist["L_data"].append(t_dat / max(nb, 1))
        hist["L_pde"].append(t_pde / max(nb, 1))
        hist["L_mono"].append(t_mon / max(nb, 1))
        hist["tau"].append(tau_val)
        hist["density"].append(density)

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}  "
                  f"train={hist['train_total'][-1]:.4f}  "
                  f"val={hist['val_total'][-1]:.4f}  "
                  f"τ={tau_val:.3f}")

    return model, hist


# ── Figure 11 ─────────────────────────────────────────────────────────────────

def plot_fig11(history, save_dir):
    """Fig 11 — Training and validation loss curves (total + 3 components)."""
    epochs = np.arange(1, len(history["train_total"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Fig 11 – Training Convergence of PI-XGNN", fontweight="bold")

    # Left: total train vs val
    ax1.semilogy(epochs, history["train_total"], "b-",  lw=1.5, label="Training loss")
    ax1.semilogy(epochs, history["val_total"],   "r--", lw=1.5, label="Validation loss")
    ax1.axhline(history["train_total"][-1], color="grey", ls=":", lw=1)
    ax1.text(epochs[-1] * 0.6, history["train_total"][-1] * 1.5,
             f"Final train: {history['train_total'][-1]:.4f}", fontsize=8)
    ax1.text(epochs[-1] * 0.6, history["val_total"][-1] * 1.5,
             f"Final val:   {history['val_total'][-1]:.4f}", fontsize=8, color="red")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("(a) Training vs. validation loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Right: loss components
    ax2.semilogy(epochs, history["L_data"], "b-",  lw=1.5, label="$\\mathcal{L}_{data}$")
    ax2.semilogy(epochs, history["L_pde"],  "r-",  lw=1.5, label="$\\mathcal{L}_{PDE}$")
    ax2.semilogy(epochs, history["L_mono"], "g-",  lw=1.5, label="$\\mathcal{L}_{mono}$")
    for k in ("L_data", "L_pde", "L_mono"):
        ax2.text(epochs[-1] * 0.7, history[k][-1],
                 f"{history[k][-1]:.4f}", fontsize=8)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Component loss (log scale)")
    ax2.set_title("(b) Loss component breakdown")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, "fig11_loss_curves.png")


# ── Figure 12 ─────────────────────────────────────────────────────────────────

def plot_fig12(histories: dict, save_dir):
    """
    Fig 12 — σ(τ) evolution (left) + graph edge density (right).

    Parameters
    ----------
    histories : {condition_label: history_dict}  e.g. {"C1": h1, "C2": h2, "C3": h3}
    """
    epochs = None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 12 – Learnable Threshold σ(τ) and Graph Edge Density",
                 fontweight="bold")

    colors = {"C1": "blue", "C2": "orange", "C3": "green"}
    for label, hist in histories.items():
        ep  = np.arange(1, len(hist["tau"]) + 1)
        col = colors.get(label, "black")
        ax1.plot(ep, hist["tau"],     color=col, lw=1.5, label=label)
        ax2.plot(ep, hist["density"], color=col, lw=1.5, label=label)

    ax1.axhline(0.5, color="grey", ls="--", lw=1, label="Threshold=0.5")
    ax1.set_xlabel("Training epoch"); ax1.set_ylabel("Learned threshold σ(τ)")
    ax1.set_title("(a) Evolution of σ(τ)"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.annotate("Sparse (healthy)", xy=(1, 0.72), fontsize=8, color="grey")
    ax1.annotate("Dense (degraded)", xy=(1, 0.22), fontsize=8, color="grey")

    ax2.axhline(0.5, color="grey", ls="--", lw=1, label="Threshold=0.5")
    ax2.set_xlabel("Training epoch"); ax2.set_ylabel("Graph edge density")
    ax2.set_title("(b) Graph edge density"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_dir, "fig12_tau_evolution.png")


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
