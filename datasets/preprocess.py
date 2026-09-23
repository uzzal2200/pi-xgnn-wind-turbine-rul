"""Feature extraction and RUL label computation."""
import numpy as np
from scipy import stats


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(signal_h: np.ndarray, signal_v: np.ndarray) -> np.ndarray:
    """Extract 12 time/frequency-domain features from one acquisition window."""
    x = np.sqrt(signal_h ** 2 + signal_v ** 2)
    eps = 1e-8

    rms       = np.sqrt(np.mean(x ** 2))
    sra       = np.mean(np.sqrt(np.abs(x) + eps)) ** 2
    peak      = np.max(np.abs(x))
    p2p       = x.max() - x.min()
    crest     = peak / (rms + eps)
    kurt      = float(stats.kurtosis(x, fisher=True))
    skew      = float(stats.skew(x))
    impulse   = peak / (np.mean(np.abs(x)) + eps)
    waveform  = rms / (np.mean(np.abs(x)) + eps)

    N         = len(x)
    freqs     = np.fft.rfftfreq(N)
    amp       = np.abs(np.fft.rfft(x))
    amp_norm  = amp / (amp.sum() + eps)
    mf        = float(np.sum(freqs * amp_norm))
    sf        = float(np.sqrt(np.sum((freqs - mf) ** 2 * amp_norm)))

    # Index 11 = IRRMS placeholder; updated in preprocess_bearing()
    return np.array([rms, sra, peak, p2p, crest, kurt, skew,
                     impulse, waveform, mf, sf, 1.0], dtype=np.float32)


# ── Health index ──────────────────────────────────────────────────────────────

def compute_irrms(rms_series: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    T     = len(rms_series)
    irrms = np.ones(T, dtype=np.float32)
    for t in range(1, T):
        irrms[t] = ((1 - alpha) * irrms[t - 1]
                    + alpha * rms_series[t] / (rms_series[0] + 1e-8))
    return irrms


def find_fpt(irrms: np.ndarray, eta: float = 1.1) -> int:
    idx = np.where(irrms > eta)[0]
    return int(idx[0]) if len(idx) > 0 else len(irrms) - 1


# ── Changepoint detection ─────────────────────────────────────────────────────

def pelt_changepoints(signal: np.ndarray,
                       penalty_mult: float = 3.0,
                       min_size: int = 5) -> list:
    try:
        import ruptures as rpt
        penalty = penalty_mult * np.log(max(len(signal), 2))
        algo    = rpt.Pelt(model="l2", min_size=min_size).fit(signal.reshape(-1, 1))
        bps     = algo.predict(pen=penalty)
        return bps[:-1]                       # exclude trailing end-marker
    except Exception:
        mid = len(signal) // 2
        return [mid] if mid > min_size else []


# ── RUL labels ────────────────────────────────────────────────────────────────

def compute_rul_labels(irrms: np.ndarray, fpt: int,
                        penalty_mult: float = 3.0,
                        min_size: int = 5) -> np.ndarray:
    """Piecewise multi-stage linear RUL labels (normalised to [0, 1])."""
    T       = len(irrms)
    bps_rel = pelt_changepoints(irrms[fpt:], penalty_mult, min_size)
    bps_abs = sorted([fpt + bp for bp in bps_rel])

    stages  = [0] + bps_abs + [T]
    rul     = np.zeros(T, dtype=np.float32)

    for i in range(len(stages) - 1):
        s, e = stages[i], stages[i + 1]
        rul[s:e] = np.linspace(1.0 - s / T, 1.0 - (e - 1) / T, e - s)

    return np.clip(rul, 0.0, 1.0)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_bearing(cycles_h: np.ndarray,
                        cycles_v: np.ndarray,
                        alpha: float = 0.1,
                        eta: float = 1.1,
                        penalty_mult: float = 3.0,
                        min_size: int = 5):
    """
    Returns
    -------
    features : np.ndarray  shape (T, 12)
    rul      : np.ndarray  shape (T,)
    fpt      : int
    irrms    : np.ndarray  shape (T,)
    """
    T    = len(cycles_h)
    feat = np.array([extract_features(cycles_h[t], cycles_v[t])
                     for t in range(T)])

    # IRRMS (feature index 11)
    irrms      = compute_irrms(feat[:, 0], alpha)
    feat[:, 11] = irrms

    # Per-bearing baseline normalisation (divide by median of first T/10 cycles)
    baseline_end = max(1, T // 10)
    base_med     = np.median(feat[:baseline_end], axis=0)
    feat         = feat / (base_med + 1e-8)

    fpt = find_fpt(irrms, eta)
    rul = compute_rul_labels(irrms, fpt, penalty_mult, min_size)

    return feat, rul, fpt, irrms
