"""PHM 2012 / PRONOSTIA (FEMTO-ST) bearing dataset loader.

Download
--------
https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/
IEEE-PHM-2012-Data-challenge

Expected directory layout
--------------------------
data/PHM2012/
  Learning_set/
    Bearing1_1/  acc_00001.csv  acc_00002.csv  ...
    Bearing1_2/  ...
    Bearing2_1/  ...
    Bearing2_2/  ...
    Bearing3_1/  ...
    Bearing3_2/  ...
  Full_Test_Set/
    Bearing1_3/  acc_00001.csv  ...
    Bearing1_4/  ...
    ...  (11 test bearings total)

CSV row format
--------------
  hour, minute, second, microsecond, acc_horizontal(g), acc_vertical(g)
  2560 rows per file  (0.1 s at 25.6 kHz)
  One file every 10 seconds

Operating conditions
--------------------
  Condition 1 : 1800 rpm / 4 kN
  Condition 2 : 1800 rpm / 4 kN  (different bearings)
  Condition 3 : 1500 rpm / 5 kN
"""
import os
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── Per-bearing loader ────────────────────────────────────────────────────────

def _read_acc_csv(path: str):
    """Read one acc_XXXXX.csv → (h_signal, v_signal) each shape (2560,)."""
    if HAS_PANDAS:
        df = pd.read_csv(path, header=None)
        h  = df.iloc[:, 4].to_numpy(dtype=np.float32)
        v  = df.iloc[:, 5].to_numpy(dtype=np.float32)
    else:
        data = np.loadtxt(path, delimiter=",")
        h    = data[:, 4].astype(np.float32)
        v    = data[:, 5].astype(np.float32)
    return h, v


def load_bearing_phm(data_dir: str, subset: str, condition: int, bearing_id: int):
    """
    Load one bearing from PHM 2012.

    Parameters
    ----------
    data_dir   : root dir (contains Learning_set/ and Full_Test_Set/)
    subset     : "learning" | "test"
    condition  : 1 | 2 | 3
    bearing_id : 1, 2 (learning) or 3-5 (test, condition-dependent)

    Returns
    -------
    cycles_h : np.ndarray  (T, 2560)
    cycles_v : np.ndarray  (T, 2560)
    """
    sub_map = {"learning": "Learning_set", "test": "Full_Test_Set"}
    folder  = os.path.join(data_dir,
                           sub_map.get(subset.lower(), subset),
                           f"Bearing{condition}_{bearing_id}")

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Bearing folder not found: {folder}\n"
            f"Download PHM 2012 from FEMTO-ST and set data_dir accordingly.")

    files = sorted([f for f in os.listdir(folder)
                    if f.startswith("acc") and f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No acc_*.csv files found in {folder}")

    cycles_h, cycles_v = [], []
    for fname in files:
        h, v = _read_acc_csv(os.path.join(folder, fname))
        cycles_h.append(h)
        cycles_v.append(v)

    return np.array(cycles_h), np.array(cycles_v)


# ── Condition-level loader ────────────────────────────────────────────────────

# Official PHM 2012 challenge split
_LEARNING_BEARINGS = {1: [1, 2], 2: [1, 2], 3: [1, 2]}
_TEST_BEARINGS     = {1: [3, 4, 5, 6, 7], 2: [3, 4, 5, 6, 7], 3: [3]}


def load_condition_phm(data_dir: str, condition: int) -> dict:
    """
    Load all available bearings for one PHM 2012 condition.

    Returns
    -------
    {
      "learning": {bearing_id: (cycles_h, cycles_v)},
      "test":     {bearing_id: (cycles_h, cycles_v)},
    }
    """
    result = {"learning": {}, "test": {}}

    for bid in _LEARNING_BEARINGS.get(condition, []):
        try:
            h, v = load_bearing_phm(data_dir, "learning", condition, bid)
            result["learning"][bid] = (h, v)
            print(f"  [PHM-C{condition}] Learning Bearing{condition}_{bid}: {len(h)} cycles")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    for bid in _TEST_BEARINGS.get(condition, []):
        try:
            h, v = load_bearing_phm(data_dir, "test", condition, bid)
            result["test"][bid] = (h, v)
            print(f"  [PHM-C{condition}] Test    Bearing{condition}_{bid}: {len(h)} cycles")
        except FileNotFoundError:
            pass   # some test bearings may be absent in partial downloads

    return result


def load_all_phm(data_dir: str) -> dict:
    """
    Load all three PHM 2012 conditions.

    Returns
    -------
    {"C1": {...}, "C2": {...}, "C3": {...}}
    each value = {"learning": {bid: (h, v)}, "test": {bid: (h, v)}}
    """
    data = {}
    for cond in [1, 2, 3]:
        data[f"C{cond}"] = load_condition_phm(data_dir, cond)
    return data
