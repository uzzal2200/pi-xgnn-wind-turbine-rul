"""Plotting utilities for RUL prediction results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # headless-safe backend
import matplotlib.pyplot as plt


def plot_rul(results: dict,
             title: str = "PI-XGNN – RUL Prediction",
             save_path: str | None = None) -> plt.Figure:
    """Two-panel RUL plot: prediction + CI (top) and absolute error (bottom)."""
    y   = results["y_true"]
    mu  = results["mu"]
    lo  = results["lo"]
    hi  = results["hi"]
    t   = np.arange(len(y))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(t, y,  "r-",  lw=2.0, label="True RUL")
    ax1.plot(t, mu, "g-",  lw=1.5,
             label=f"Predicted (RMSE={results['rmse']:.4f})")
    ax1.fill_between(t, lo, hi, alpha=0.25, color="green",
                     label=f"95% CI  (PICP={results['picp']:.3f})")
    ax1.set_ylabel("Normalised RUL")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(title)

    err = np.abs(mu - y)
    ax2.bar(t, err, color="steelblue", alpha=0.6, width=1.0)
    ax2.set_ylabel("|Error|")
    ax2.set_xlabel("Sampling cycle")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    return fig


def plot_feature_importance(imp: np.ndarray,
                             feature_names: list | None = None,
                             save_path: str | None = None) -> plt.Figure:
    """Horizontal bar chart of gradient-based node importance."""
    if feature_names is None:
        feature_names = ["RMS", "SRA", "Peak", "P2P", "Crest",
                         "Kurtosis", "Skewness", "Impulse",
                         "Waveform", "Mean Freq", "Std Freq", "IRRMS"]
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[idx], imp[idx], color="steelblue")
    ax.axvline(1 / len(imp), color="red", ls="--", lw=1.2, label="Uniform")
    ax.set_xlabel("Importance score (normalised)")
    ax.set_title("Gradient-based feature importance")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
