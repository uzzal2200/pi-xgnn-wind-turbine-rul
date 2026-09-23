"""XJTU-SY bearing dataset loader.

Dataset structure expected
--------------------------
data_dir/
  Bearing{condition}_{id}/
    acc_00001.mat   ...   acc_0XXXX.mat
Each .mat: {'bearing': {'gs': array(25600, 2)}}  [horizontal, vertical]

Download: https://biaowang.tech/xjtu-sy-bearing-datasets/
"""
import os
import numpy as np

try:
    import scipy.io as sio
except ImportError:
    sio = None


def load_bearing(data_dir: str, condition: int, bearing_id: int):
    """Load all acquisition cycles for one bearing.

    Returns
    -------
    cycles_h : np.ndarray  shape (T, N_samples)
    cycles_v : np.ndarray  shape (T, N_samples)
    """
    if sio is None:
        raise ImportError("scipy is required to load .mat files: pip install scipy")

    folder = os.path.join(data_dir, f"Bearing{condition}_{bearing_id}")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Bearing folder not found: {folder}\n"
                                f"Download XJTU-SY and set --data_dir accordingly.")

    files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])
    if not files:
        raise FileNotFoundError(f"No .mat files found in {folder}")

    cycles_h, cycles_v = [], []
    for fname in files:
        mat = sio.loadmat(os.path.join(folder, fname))
        # Handle both struct-style and direct array storage
        if "bearing" in mat:
            gs = mat["bearing"]["gs"][0][0]      # shape (25600, 2)
        else:
            # fallback: last numeric key
            gs = [v for v in mat.values()
                  if isinstance(v, np.ndarray) and v.ndim == 2][-1]
        cycles_h.append(gs[:, 0].astype(np.float32))
        cycles_v.append(gs[:, 1].astype(np.float32))

    return np.array(cycles_h), np.array(cycles_v)


def load_condition(data_dir: str, condition: int, n_bearings: int = 5) -> dict:
    """Load all bearings for one operating condition.

    Returns {bearing_id: (cycles_h, cycles_v)}
    """
    data = {}
    for bid in range(1, n_bearings + 1):
        try:
            h, v = load_bearing(data_dir, condition, bid)
            data[bid] = (h, v)
            print(f"  Loaded Bearing{condition}_{bid}: {len(h)} cycles")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
    return data
