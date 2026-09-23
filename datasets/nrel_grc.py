"""NREL Gearbox Reliability Collaborative (GRC) dataset loader.

Download
--------
https://www.nrel.gov/wind/grc.html
Reference: Rezamand et al., IEEE/ASME Trans. Mechatronics, 2020.

Dataset description
-------------------
750 kW, three-stage wind turbine gearbox tested on a dynamometer.
Two configurations: Healthy and Damaged (artificially seeded fault).
Accelerometers at three locations:
  - Planet Carrier (PC)  — low-speed stage
  - Intermediate-speed stage (IMS)
  - High-speed shaft     (HSS)  ← primary focus in the paper
Each location has Horizontal (H) and Vertical (V) channels.

Expected directory layout (most common download format)
--------------------------------------------------------
data/NREL_GRC/
  Healthy/
    *.mat   OR   *.csv
  Damaged/
    *.mat   OR   *.csv

OR flat layout:
  data/NREL_GRC/
    GRC_Healthy_*.mat
    GRC_Damaged_*.mat

The loader auto-detects .mat or .csv and tries common key/column names.

Channel mapping (paper uses HSS H+V for zero-shot evaluation)
-------------------------------------------------------------
  hss_h  — HSS horizontal acceleration
  hss_v  — HSS vertical acceleration
  ims_h  — IMS horizontal  (available but not used in primary eval)
  pc_h   — Planet carrier  (available but not used in primary eval)
"""
import os
import numpy as np

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── Key-name candidates for common GRC .mat structures ───────────────────────

_HSS_H_KEYS = ["hss_h", "HSS_H", "ch7", "channel7", "acc_hss_h",
               "AN6",  "AN7",  "AN8"]   # vary by version of GRC download
_HSS_V_KEYS = ["hss_v", "HSS_V", "ch8", "channel8", "acc_hss_v",
               "AN9",  "AN10", "AN11"]
_FS_KEYS    = ["fs", "Fs", "sample_rate", "SampleRate", "samplerate"]


def _extract_hss_from_mat(mat: dict):
    """Try known key-name patterns to extract HSS H and V channels."""
    keys = [k for k in mat if not k.startswith("__")]

    def _find(candidates, fallback_idx):
        for c in candidates:
            if c in mat:
                arr = np.array(mat[c]).ravel().astype(np.float32)
                if arr.size > 0:
                    return arr
        # Fallback: pick by column index if data is a 2-D array
        for k in keys:
            v = mat[k]
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= fallback_idx + 1:
                return v[:, fallback_idx].astype(np.float32)
        return None

    h_sig = _find(_HSS_H_KEYS, 0)
    v_sig = _find(_HSS_V_KEYS, 1)
    fs    = None
    for k in _FS_KEYS:
        if k in mat:
            fs = float(np.array(mat[k]).ravel()[0])
            break
    return h_sig, v_sig, fs


def _load_mat_file(path: str):
    """Load one GRC .mat file → (h_signal, v_signal, fs)."""
    if not HAS_SCIPY:
        raise ImportError("scipy required for .mat files: pip install scipy")
    mat = sio.loadmat(path)
    h, v, fs = _extract_hss_from_mat(mat)
    return h, v, (fs or 20000.0)


def _load_csv_file(path: str):
    """
    Load one GRC .csv file.
    Expected: columns include HSS_H and HSS_V (or similar),
    or plain two-column [h, v] format.
    """
    if HAS_PANDAS:
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        # Try named columns first
        for hc in [c for c in df.columns if any(k in c.lower() for k in ["hss_h","ch7","h_acc"])]:
            for vc in [c for c in df.columns if any(k in c.lower() for k in ["hss_v","ch8","v_acc"])]:
                return (df[hc].to_numpy(np.float32),
                        df[vc].to_numpy(np.float32),
                        20000.0)
        # Fallback: first two numeric columns
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            return (df[num_cols[0]].to_numpy(np.float32),
                    df[num_cols[1]].to_numpy(np.float32),
                    20000.0)
    else:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 0], data[:, 1], 20000.0
    raise ValueError(f"Cannot parse GRC file: {path}")


def _load_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        return _load_mat_file(path)
    elif ext == ".csv":
        return _load_csv_file(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Configuration loader ──────────────────────────────────────────────────────

def _find_files(folder: str):
    """Return sorted list of .mat or .csv files in folder."""
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(".mat") or f.lower().endswith(".csv")])
    return files


def load_grc_config(data_dir: str, config: str, segment_len: int = 20000):
    """
    Load one GRC configuration (Healthy or Damaged).

    Parameters
    ----------
    data_dir    : root GRC directory
    config      : "healthy" | "damaged"
    segment_len : samples per cycle (default 20000 ≈ 1 s at 20 kHz)
                  Set to match XJTU-SY's feature extraction granularity.

    Returns
    -------
    cycles_h : np.ndarray  (T, segment_len)
    cycles_v : np.ndarray  (T, segment_len)
    fs       : float  sampling frequency
    """
    # Try sub-folder first, then root-level file naming
    sub = os.path.join(data_dir, config.capitalize())
    if not os.path.isdir(sub):
        sub = data_dir   # files directly in root

    if not os.path.isdir(sub):
        raise FileNotFoundError(
            f"GRC folder not found: {sub}\n"
            f"Download from https://www.nrel.gov/wind/grc.html")

    files = _find_files(sub)
    # Filter by config name if files are in the root
    if sub == data_dir:
        files = [f for f in files if config.lower() in f.lower()]

    if not files:
        raise FileNotFoundError(f"No data files in {sub} for config={config}")

    all_h, all_v, fs_val = [], [], 20000.0
    for fname in files:
        try:
            h, v, fs = _load_file(os.path.join(sub, fname))
            fs_val = fs
            # Segment the long signal into cycles of segment_len samples
            if h is None or v is None:
                continue
            min_len = min(len(h), len(v))
            n_seg   = min_len // segment_len
            for i in range(n_seg):
                s = i * segment_len
                all_h.append(h[s: s + segment_len])
                all_v.append(v[s: s + segment_len])
        except Exception as e:
            print(f"  Warning: skipping {fname} — {e}")

    if not all_h:
        raise RuntimeError(f"No usable segments loaded for GRC {config}")

    return np.array(all_h), np.array(all_v), fs_val


def load_all_grc(data_dir: str, segment_len: int = 20000) -> dict:
    """
    Load both GRC configurations.

    Returns
    -------
    {
      "healthy": (cycles_h, cycles_v, fs),
      "damaged": (cycles_h, cycles_v, fs),
    }
    """
    data = {}
    for cfg in ["healthy", "damaged"]:
        try:
            h, v, fs = load_grc_config(data_dir, cfg, segment_len)
            data[cfg] = (h, v, fs)
            print(f"  [GRC] {cfg.capitalize()}: {len(h)} segments  fs={fs:.0f} Hz")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
    return data
