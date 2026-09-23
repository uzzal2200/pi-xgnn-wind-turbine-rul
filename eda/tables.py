"""
All Tables from the paper (Tables 1–14).

Static tables: printed + saved as CSV.
Dynamic tables: computed from experiment results + saved.
"""
import os
import csv
import numpy as np
from scipy import stats as sp_stats


def _save_csv(rows, header, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Saved → {path}")


def _hline(widths):
    return "─" * (sum(widths) + len(widths) * 3 + 1)


def _print_table(title, header, rows, widths=None):
    if widths is None:
        widths = [max(len(str(r[i])) for r in [header] + rows) + 2
                  for i in range(len(header))]
    hl = _hline(widths)
    fmt = " │ ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n{'─'*5} {title} {'─'*5}")
    print(hl)
    print(fmt.format(*header))
    print(hl)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))
    print(hl)


# ── Table 1 ───────────────────────────────────────────────────────────────────

def table1(save_dir):
    """Comparison of representative RUL methods from literature."""
    header = ["Method", "Year", "Venue", "Architecture",
              "Physics", "Graph", "Prob.", "Expl.", "WT"]
    rows = [
        ["PGLSTM",    "2021", "AEI",   "PI-LSTM",           "✓","✗","✗","✗","✗"],
        ["AttnPINN",  "2023", "AEI",   "PINN+Attention",    "✓","✗","✗","✗","✗"],
        ["STA-HPINN", "2025", "AEI",   "ST-PINN",           "✓","✗","✗","✗","✗"],
        ["PI-TENN",   "2026", "TIM",   "LSTM+Attn+PDE",     "✓","✗","✗","✗","✗"],
        ["ST-GNN",    "2025", "RESS",  "Spatio-temporal GNN","✗","✓","✗","✗","✗"],
        ["GAT-RUL",   "2025", "RESS",  "Graph attention",   "✗","✓","✗","✗","✗"],
        ["Dual-GNN",  "2025", "TIM",   "Dual-channel GNN",  "✗","✓","✗","✗","✗"],
        ["GCN-ST",    "2024", "TIM",   "Spatio-temporal GNN","✗","✓","✗","✗","✗"],
        ["MC-Dropout","2021", "Meas.", "CNN+MC-Dropout",    "✗","✗","✓","✗","✗"],
        ["SHAP-RUL",  "2024", "Access","CNN+SHAP",          "✗","✗","✗","✓","✗"],
        ["Attn-RUL",  "2025", "Res.Eng","Attention+Expl.",  "✗","✗","✗","✓","✗"],
        ["GNN-Expl",  "2023", "arXiv†","GNN+Expl.",         "✗","✓","✗","✓","✗"],
        ["Mog-LSTM",  "2024", "YAC",   "Mogrifier-LSTM",    "✗","✗","✗","✗","✗"],
        ["PI-XGNN",   "2026", "—",     "PI-GNN+PDE",        "✓","✓","✓","✓","✓"],
    ]
    _print_table("Table 1 – Literature Comparison", header, rows)
    _save_csv(rows, header, save_dir, "table01_literature.csv")


# ── Table 2 ───────────────────────────────────────────────────────────────────

def table2(data_summary: dict | None = None, save_dir="./tables"):
    """Dataset summary — computed from loaded data or defaults to paper values."""
    header = ["Dataset", "Type", "Bearings", "Conditions", "Fs (kHz)"]
    defaults = {
        "XJTU-SY":  ["Lab run-to-failure", 15, 3, 25.6],
        "PHM 2012": ["Lab run-to-failure", 17, 3, 25.6],
        "NREL GRC": ["Real WT gearbox",     2, 1, "Var."],
    }
    rows = []
    for name, vals in defaults.items():
        if data_summary and name in data_summary:
            vals = data_summary[name]
        rows.append([name] + [str(v) for v in vals])

    _print_table("Table 2 – Dataset Summary", header, rows)
    _save_csv(rows, header, save_dir, "table02_datasets.csv")


# ── Table 3 ───────────────────────────────────────────────────────────────────

def table3(save_dir):
    header = ["#", "Feature", "Physical meaning"]
    rows = [
        [1, "RMS",         "Vibration energy level"],
        [2, "SRA",         "Amplitude growth sensitivity"],
        [3, "Peak value",  "Instantaneous maximum amplitude"],
        [4, "Peak-to-peak","Full excursion range"],
        [5, "Crest factor","Peak-to-RMS ratio"],
        [6, "Kurtosis",    "Impulsiveness; early fault indicator"],
        [7, "Skewness",    "Signal asymmetry"],
        [8, "Impulse factor","Impact severity"],
        [9, "Waveform factor","Signal shape descriptor"],
        [10,"Mean frequency","Spectral centroid"],
        [11,"Std frequency","Spectral spread"],
        [12,"IRRMS",       "Smoothed relative health index"],
    ]
    _print_table("Table 3 – Extracted Vibration Features", header, rows)
    _save_csv(rows, header, save_dir, "table03_features.csv")


# ── Table 4 ───────────────────────────────────────────────────────────────────

def table4(save_dir):
    header = ["Component", "Physical Role", "Weight"]
    rows = [
        ["L_data (MSE)",   "Fits true RUL labels",                    "1.00"],
        ["L_PDE (residual)","Enforces PDE-based decay-rate consistency","0.15"],
        ["L_mono",          "Enforces physically irreversible degradation","0.15"],
    ]
    _print_table("Table 4 – PI-XGNN Loss Function Components", header, rows)
    _save_csv(rows, header, save_dir, "table04_loss.csv")


# ── Table 5 ───────────────────────────────────────────────────────────────────

def table5(save_dir):
    header = ["Module", "Input → Output", "Activation", "Role"]
    rows = [
        ["Adaptive graph encoding","(B,L,d)→(B,L,d)","GELU","Sensor relation graph"],
        ["LSTM layer 1","(B,L,12)→(B,L,32)","tanh","Short-range temporal"],
        ["LSTM layer 2","(B,L,32)→(B,L,64)","tanh","Mid-range temporal"],
        ["LSTM layer 3","(B,L,64)→(B,L,64)","tanh","Long-range temporal"],
        ["Multi-head attn (4 heads)","(B,L,64)×3→(B,L,64)","softmax","Noise suppression"],
        ["Feed-forward+LayerNorm","(B,L,64)→(B,L,128)→(B,L,64)","GELU","Feature enhancement"],
        ["Flat MLP","(B,L·64)→(B,128)→(B,32)→(B,8)","GELU","Degradation code h"],
        ["Solution net F","(B,9)→(B,32)→(B,16)→(B,1)","Tanh","RUL estimate û"],
        ["Dynamics net G","(B,12)→(B,64)→(B,64)→(B,32)→(B,1)","ReLU","Decay-rate"],
        ["MC-Dropout","p=0.20, Flat & FF layers","Bernoulli","Epistemic uncertainty"],
    ]
    _print_table("Table 5 – PI-XGNN Architecture Summary", header, rows)
    _save_csv(rows, header, save_dir, "table05_architecture.csv")


# ── Table 6 ───────────────────────────────────────────────────────────────────

def table6(config=None, save_dir="./tables"):
    header = ["Hyperparameter", "PI-XGNN", "All baselines"]
    defaults = [
        ["Window length L",  "5",       "5 (same)"],
        ["Batch size",        "64",      "64 (same)"],
        ["Optimizer",         "AdamW",   "AdamW (same)"],
        ["Learning rate",     "1e-3",    "1e-3 (same)"],
        ["LR scheduler",      "Cosine",  "Cosine (same)"],
        ["Weight decay",      "1e-4",    "1e-4 (same)"],
        ["Training epochs",   "200×3 seeds","200"],
        ["Gradient clip norm","1.0",     "1.0"],
        ["Seeds (best-of-N)", "3",       "3"],
        ["MC samples T",      "100",     "—"],
    ]
    if config:
        cfg_map = {
            "Window length L":   config.window_length,
            "Batch size":        config.batch_size,
            "Learning rate":     config.lr,
            "Weight decay":      config.weight_decay,
            "Training epochs":   f"{config.epochs}×{config.n_seeds} seeds",
            "MC samples T":      config.mc_samples,
        }
        for row in defaults:
            if row[0] in cfg_map:
                row[1] = str(cfg_map[row[0]])

    _print_table("Table 6 – Training Hyperparameters", header, defaults)
    _save_csv(defaults, header, save_dir, "table06_hyperparams.csv")


# ── Table 7 ───────────────────────────────────────────────────────────────────

def table7(save_dir):
    """Model complexity comparison (from paper; params and timings hardcoded)."""
    header = ["Method", "Params", "Params (K)", "Physics", "Graph", "Prob.", "Infer (ms/batch)"]
    rows = [
        ["MLP",         "16,129",  "16.1",  "✗","✗","✗","1.00"],
        ["Transformer", "25,857",  "25.9",  "✗","✗","✗","1.00"],
        ["AttnPINN",    "70,081",  "70.1",  "✓","✗","✗","1.00"],
        ["GNN",         "70,382",  "70.4",  "✗","✓","✗","1.00"],
        ["LSTM",        "86,593",  "86.6",  "✗","✗","✗","1.00"],
        ["CNN-BiLSTM",  "150,817", "150.8", "✗","✗","✗","1.00"],
        ["PI-TENN",     "151,178", "151.2", "✓","✗","✗","2.18"],
        ["PI-XGNN",     "151,479", "151.5", "✓","✓","✓","2.51"],
    ]
    _print_table("Table 7 – Model Complexity Comparison", header, rows)
    _save_csv(rows, header, save_dir, "table07_complexity.csv")


# ── Table 8 ───────────────────────────────────────────────────────────────────

def table8(results_per_model_per_cond: dict, save_dir="./tables"):
    """
    Comprehensive performance comparison — computed from experiment results.
    Accepts either:
      {"C1": {model: result}, "C2": ..., "C3": ...}  OR
      {model: result}  (single condition, label as C1)
    """
    header = ["Condition","Bearing","T","Method","RMSE↓","MAE↓","R²↑","PICP↑","MPIW↓"]
    rows   = []
    bearing_map = {"C1": ("Bearing1_5", 52), "C2": ("Bearing2_5", 339),
                   "C3": ("Bearing3_5", 114)}

    # Detect flat dict {model: result} vs nested {cond: {model: result}}
    first_val = next(iter(results_per_model_per_cond.values()))
    if isinstance(first_val, dict) and "rmse" in first_val:
        # flat: single condition
        data = {"C1": results_per_model_per_cond}
    else:
        data = results_per_model_per_cond

    for cond, model_res in data.items():
        if not model_res: continue
        bear, T = bearing_map.get(cond, ("—", "—"))
        for m_name, res in model_res.items():
            rows.append([cond, bear, T, m_name,
                         f"{res['rmse']:.4f}", f"{res['mae']:.4f}",
                         f"{res['r2']:.4f}",   f"{res['picp']:.3f}",
                         f"{res['mpiw']:.4f}"])

    _print_table("Table 8 – Comprehensive Performance Comparison", header, rows)
    _save_csv(rows, header, save_dir, "table08_performance.csv")


# ── Table 9 ───────────────────────────────────────────────────────────────────

def table9(results_per_cond: dict, pixgnn_key="pi_xgnn", save_dir="./tables"):
    """
    Two-sided Wilcoxon signed-rank test: PI-XGNN vs each baseline.
    Accepts flat {model: result} or nested {"C1": {model: result}, ...}
    """
    # Normalise to nested format
    first_val = next(iter(results_per_cond.values()))
    if isinstance(first_val, dict) and "rmse" in first_val:
        results_per_cond = {"C1": results_per_cond}
    header = ["Condition", "Method", "T", "W", "p-value", "|Z|", "r"]
    rows = []
    alpha_star = 0.05 / 7

    for cond, model_res in results_per_cond.items():
        if pixgnn_key not in model_res:
            continue
        pi_err = np.abs(model_res[pixgnn_key]["mu"] - model_res[pixgnn_key]["y_true"])

        for m_name, res in model_res.items():
            if m_name == pixgnn_key:
                continue
            err  = np.abs(res["mu"] - res["y_true"])
            T_n  = len(pi_err)
            try:
                stat, pval = sp_stats.wilcoxon(pi_err, err, alternative="two-sided")
                Z    = float(sp_stats.norm.ppf(pval / 2))  # approx
                r    = abs(Z) / np.sqrt(T_n)
            except Exception:
                stat, pval, Z, r = float("nan"), float("nan"), float("nan"), float("nan")

            sig = "✓" if pval < alpha_star else "✗"
            rows.append([cond, m_name, T_n,
                         f"{stat:.0f}", f"{pval:.2e}", f"{abs(Z):.2f}",
                         f"{r:.3f} {sig}"])

    _print_table(
        f"Table 9 – Wilcoxon Test  (α*={alpha_star:.4f})", header, rows)
    _save_csv(rows, header, save_dir, "table09_wilcoxon.csv")


# ── Table 10 ──────────────────────────────────────────────────────────────────

def table10(phm_results: dict, save_dir="./tables"):
    """
    Cross-dataset generalisation on PHM 2012.
    phm_results : {"PHM-C1": {"zero_shot": {model: result}, "fine_tuned": {model: result}}, ...}
    """
    header = ["Method", "Setup", "PHM-C1 RMSE", "PHM-C2 RMSE", "PHM-C3 RMSE",
              "PHM-C1 R²", "PHM-C1 PICP"]
    rows = []
    conds = ["PHM-C1", "PHM-C2", "PHM-C3"]

    models = set()
    for c in conds:
        if c in phm_results:
            for setup in phm_results[c]:
                models.update(phm_results[c][setup].keys())

    for m in models:
        for setup in ("zero_shot", "fine_tuned"):
            row = [m, setup]
            for c in conds:
                try:
                    r = phm_results[c][setup][m]
                    row.append(f"{r['rmse']:.4f}")
                except (KeyError, TypeError):
                    row.append("—")
            # R² and PICP for C1
            try:
                r1 = phm_results["PHM-C1"][setup][m]
                row += [f"{r1['r2']:.4f}", f"{r1['picp']:.3f}"]
            except (KeyError, TypeError):
                row += ["—", "—"]
            rows.append(row)

    if rows:
        _print_table("Table 10 – PHM 2012 Cross-Dataset Generalisation", header, rows)
        _save_csv(rows, header, save_dir, "table10_phm_generalisation.csv")
    else:
        print("\n  Table 10: No PHM 2012 results available.")


# ── Table 11 ──────────────────────────────────────────────────────────────────

def table11(silhouette_scores: dict, save_dir="./tables"):
    """Latent-space separation on NREL GRC."""
    header = ["Variant", "t-SNE Silhouette↑", "UMAP Silhouette↑"]
    rows = [[k, f"{v[0]:.3f}", f"{v[1]:.3f}"]
            for k, v in silhouette_scores.items()]
    _print_table("Table 11 – NREL GRC Latent-Space Separation", header, rows)
    _save_csv(rows, header, save_dir, "table11_grc_silhouette.csv")


# ── Table 12 ──────────────────────────────────────────────────────────────────

def table12(ablation_results: dict, save_dir="./tables"):
    """Ablation study results."""
    baseline = ablation_results.get("PI-XGNN", {}).get("rmse", float("nan"))
    header   = ["Variant", "RMSE↓", "MAE↓", "R²↑", "RMSE Δ"]
    rows = []
    for name, res in ablation_results.items():
        pct = f"+{100*(res['rmse']-baseline)/baseline:.1f}%" \
              if name != "PI-XGNN" else "—"
        rows.append([name,
                     f"{res['rmse']:.4f}", f"{res['mae']:.4f}",
                     f"{res['r2']:.4f}", pct])
    _print_table("Table 12 – Ablation Study (Condition 1)", header, rows)
    _save_csv(rows, header, save_dir, "table12_ablation.csv")


# ── Table 13 ──────────────────────────────────────────────────────────────────

def table13(window_results: dict, save_dir="./tables"):
    """Sensitivity to sliding-window length L."""
    header = ["L", "RMSE↓", "MAE↓", "R²↑", "PICP"]
    rows   = [[L, f"{r['rmse']:.4f}", f"{r['mae']:.4f}",
               f"{r['r2']:.4f}",  f"{r['picp']:.3f}"]
              for L, r in sorted(window_results.items())]
    _print_table("Table 13 – Sensitivity to Window Length L", header, rows)
    _save_csv(rows, header, save_dir, "table13_window_sensitivity.csv")


# ── Table 14 ──────────────────────────────────────────────────────────────────

def table14(save_dir):
    """Computational cost comparison (from paper; hardcoded)."""
    header = ["Method", "Params", "FLOPs", "Train (min/seed)", "Infer (ms)", "Size (MB)"]
    rows = [
        ["MLP",        "16.1K", "0.03M", "~12", "1.00", "0.06"],
        ["Transformer","25.9K", "0.09M", "~15", "1.00", "0.10"],
        ["AttnPINN",   "70.1K", "0.27M", "~25", "1.00", "0.27"],
        ["GNN",        "70.4K", "0.11M", "~20", "1.00", "0.27"],
        ["LSTM",       "86.6K", "0.41M", "~28", "1.00", "0.33"],
        ["CNN-BiLSTM", "150.8K","0.89M", "~41", "1.00", "0.58"],
        ["PI-TENN",    "151.2K","1.12M", "~53", "2.18", "0.58"],
        ["PI-XGNN",    "151.5K","1.19M", "~58", "2.51", "0.61"],
    ]
    _print_table("Table 14 – Computational Cost Comparison", header, rows)
    _save_csv(rows, header, save_dir, "table14_compute.csv")


# ── Master table runner ────────────────────────────────────────────────────────

def generate_static_tables(config=None, save_dir="./tables"):
    """Tables that don't require experiment results (1–7, 14)."""
    print("\n[Static Tables 1, 2, 3, 4, 5, 6, 7, 14]")
    table1(save_dir)
    table2(save_dir=save_dir)
    table3(save_dir)
    table4(save_dir)
    table5(save_dir)
    table6(config, save_dir)
    table7(save_dir)
    table14(save_dir)


def generate_result_tables(results_c1, results_c2, results_c3,
                            ablation_res=None, window_res=None,
                            phm_results=None, silhouette_scores=None,
                            save_dir="./tables"):
    """Tables that require experiment results (8–13)."""
    print("\n[Dynamic Tables 8–13]")
    table8({"C1": results_c1, "C2": results_c2, "C3": results_c3}, save_dir)
    table9({"C1": results_c1, "C2": results_c2, "C3": results_c3},
           save_dir=save_dir)
    if ablation_res:  table12(ablation_res, save_dir)
    if window_res:    table13(window_res, save_dir)
    if phm_results:   table10(phm_results, save_dir)
    if silhouette_scores: table11(silhouette_scores, save_dir)
