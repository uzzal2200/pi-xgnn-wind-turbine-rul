"""
EDA Figures (Section III of paper) — raw data, no model required.

Fig 5  : Raw vibration at healthy / moderate / severe stages
Fig 6  : FFT amplitude spectra at 4 degradation levels
Fig 7  : All 12 features over bearing lifetime + IRRMS / FPT / STP / RUL
Fig 8  : Adaptive sensor-correlation graph at 3 degradation stages
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

FEATURE_NAMES = ["RMS", "SRA", "Peak", "Peak-to-Peak", "Crest",
                 "Kurtosis", "Skewness", "Impulse", "Waveform",
                 "Mean Freq", "Std Freq", "IRRMS"]
COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
          "#9C27B0", "#00BCD4", "#795548", "#607D8B",
          "#E91E63", "#3F51B5", "#009688", "#FFC107"]


def _stage_indices(irrms, fpt, stp1=None, stp2=None, T=None):
    """Return representative cycle indices for healthy / moderate / severe."""
    T = T or len(irrms)
    healthy  = max(0, fpt // 2)
    moderate = stp1 if stp1 else (fpt + T) // 2
    severe   = stp2 if stp2 else int(T * 0.9)
    return healthy, moderate, severe


# ── Figure 5 ──────────────────────────────────────────────────────────────────

def plot_fig5(cycles_h, cycles_v, irrms, fpt, stp1, stp2, save_dir):
    """
    Fig 5 — Raw vibration signals at healthy, moderate, severe stages.
    2 rows (H / V) × 3 columns (stage).
    """
    T  = len(cycles_h)
    h_idx, m_idx, s_idx = _stage_indices(irrms, fpt, stp1, stp2, T)
    stages  = [(h_idx, "Healthy"),
               (m_idx, "Moderate Degradation"),
               (s_idx, "Severe Degradation (Failure)")]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle("Fig 5 – Raw Vibration Signals at Three Degradation Stages",
                 fontsize=13, fontweight="bold")

    for col, (idx, label) in enumerate(stages):
        t_ax = np.arange(len(cycles_h[idx]))
        for row, (sig, ch) in enumerate([(cycles_h[idx], "Horizontal"),
                                          (cycles_v[idx], "Vertical")]):
            ax = axes[row, col]
            ax.plot(t_ax, sig, lw=0.6, color=COLORS[col])
            pk = np.max(np.abs(sig))
            ax.set_title(f"{label}\n{ch}  |  Peak: {pk:.3f} g", fontsize=9)
            ax.set_xlabel("Sample"); ax.set_ylabel("Acceleration (g)")
            ax.set_ylim(-pk * 1.3, pk * 1.3)

    plt.tight_layout()
    _save(fig, save_dir, "fig05_vibration_stages.png")


# ── Figure 6 ──────────────────────────────────────────────────────────────────

def plot_fig6(cycles_h, cycles_v, irrms, fpt, stp1, stp2, save_dir):
    """
    Fig 6 — FFT amplitude spectra at 4 degradation levels.
    """
    T = len(cycles_h)
    h_idx, m_idx, s_idx = _stage_indices(irrms, fpt, stp1, stp2, T)
    sl_idx = (h_idx + m_idx) // 2              # slight degradation midpoint
    stages = [(h_idx,  "Healthy (Cycle 1)"),
              (sl_idx, "Slight Degradation"),
              (m_idx,  "Moderate Degradation"),
              (s_idx,  "Severe Degradation")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Fig 6 – FFT Amplitude Spectra at Four Degradation Stages",
                 fontsize=13, fontweight="bold")
    axes = axes.ravel()

    for i, (idx, label) in enumerate(stages):
        comp  = np.sqrt(cycles_h[idx]**2 + cycles_v[idx]**2)
        N     = len(comp)
        freqs = np.fft.rfftfreq(N, d=1/25600)
        amp   = np.abs(np.fft.rfft(comp))
        pk    = np.max(amp)
        axes[i].semilogy(freqs, amp + 1e-10, lw=0.8, color=COLORS[i])
        axes[i].set_title(f"{label}\nPeak: {pk:.2f}", fontsize=9)
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Amplitude (log)")
        axes[i].set_xlim(0, freqs[-1])

    plt.tight_layout()
    _save(fig, save_dir, "fig06_fft_spectra.png")


# ── Figure 7 ──────────────────────────────────────────────────────────────────

def plot_fig7(feat, irrms, fpt, stp1, stp2, rul, save_dir):
    """
    Fig 7 — All 12 extracted features over bearing lifetime (top panel)
            + IRRMS health index with FPT, STP1, STP2, piecewise RUL (bottom).
    """
    T  = np.arange(len(feat))
    n  = feat.shape[1]                          # should be 12

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Fig 7 – Feature Evolution and Health Index", fontsize=13,
                 fontweight="bold")
    gs = GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.35)

    # ── Top: 12 feature time-series ──
    for i in range(n):
        row, col = divmod(i, 5)
        ax = fig.add_subplot(gs[row, col]) if row < 2 else \
             fig.add_subplot(gs[2, i - 10]) if i < 15 else None
        if ax is None:
            continue
        ax.plot(T, feat[:, i], lw=0.8, color=COLORS[i % len(COLORS)])
        ax.set_title(FEATURE_NAMES[i], fontsize=8)
        ax.set_xlabel("Cycle", fontsize=7)
        ax.tick_params(labelsize=7)

    # ── Bottom: IRRMS + FPT + STP + RUL ──
    ax_b = fig.add_subplot(gs[2, 2:])
    ax_b.plot(T, irrms, "b-", lw=1.5, label="IRRMS")
    ax_b.axvline(fpt,  color="blue",   ls="--", lw=1.2, label=f"FPT={fpt}")
    ax_b.axvline(stp1, color="orange", ls="--", lw=1.2, label=f"STP₁={stp1}")
    ax_b.axvline(stp2, color="red",    ls="--", lw=1.2, label=f"STP₂={stp2}")

    ax2 = ax_b.twinx()
    ax2.plot(T, rul, "g--", lw=1.5, label="RUL label")
    ax2.set_ylabel("RUL", color="green", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="green")

    ax_b.set_xlabel("Cycle"); ax_b.set_ylabel("IRRMS", fontsize=8)
    ax_b.set_title("IRRMS Health Index + Piecewise RUL Label", fontsize=9)
    handles1, l1 = ax_b.get_legend_handles_labels()
    handles2, l2 = ax2.get_legend_handles_labels()
    ax_b.legend(handles1 + handles2, l1 + l2, fontsize=7, loc="upper left")

    _save(fig, save_dir, "fig07_feature_evolution.png")


# ── Figure 8 ──────────────────────────────────────────────────────────────────

def plot_fig8(feat, irrms, fpt, stp1, stp2, save_dir, threshold=0.5):
    """
    Fig 8 — Adaptive sensor-correlation graph at healthy / moderate / severe.
    Shown as heatmap + degree bar (simple, no networkx dependency).
    """
    T   = len(feat)
    h_i = max(0, fpt // 2)
    m_i = stp1
    s_i = min(T - 1, stp2 + (T - stp2) // 2)

    stages = [(h_i, "Healthy"), (m_i, "Moderate"), (s_i, "Severe")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Fig 8 – Adaptive Sensor-Correlation Graph at Three Stages",
                 fontsize=12, fontweight="bold")

    for ax, (idx, label) in zip(axes, stages):
        window = feat[max(0, idx - 2): idx + 3]        # ±2 cycles window
        C      = np.abs(window.T @ window) / max(len(window), 1)
        np.fill_diagonal(C, 0)
        A      = (C > threshold).astype(float)
        np.fill_diagonal(A, 1)
        density = A[np.triu_indices_from(A, k=1)].mean()

        im = ax.imshow(C, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"{label}\n(edge density={density:.2f})", fontsize=9)
        ax.set_xticks(range(12)); ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=6)
        ax.set_yticks(range(12)); ax.set_yticklabels(FEATURE_NAMES, fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    _save(fig, save_dir, "fig08_correlation_graph.png")


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def generate_eda_figures(cycles_h, cycles_v, feat, irrms, fpt, stp1, stp2, rul,
                          save_dir="./figures"):
    """Run all 4 EDA figures."""
    print("\n[EDA Figures 5–8]")
    plot_fig5(cycles_h, cycles_v, irrms, fpt, stp1, stp2, save_dir)
    plot_fig6(cycles_h, cycles_v, irrms, fpt, stp1, stp2, save_dir)
    plot_fig7(feat, irrms, fpt, stp1, stp2, rul, save_dir)
    plot_fig8(feat, irrms, fpt, stp1, stp2, save_dir)
