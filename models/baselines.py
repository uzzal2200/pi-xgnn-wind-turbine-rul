"""
Baseline models compared against PI-XGNN in the paper.

┌─────────────┬────────────┬───────────┬────────────┐
│ Model       │ Physics    │ Graph     │ Prob.      │
├─────────────┼────────────┼───────────┼────────────┤
│ MLP         │ ✗          │ ✗         │ ✗          │
│ LSTM        │ ✗          │ ✗         │ ✗          │
│ CNN-BiLSTM  │ ✗          │ ✗         │ ✗          │
│ Transformer │ ✗          │ ✗         │ ✗          │
│ AttnPINN    │ ✓ (mono)   │ ✗         │ ✗          │
│ GNN         │ ✗          │ ✓ (fixed) │ ✗          │
│ PI-TENN     │ ✓ (PDE)    │ ✗         │ ✗          │
└─────────────┴────────────┴───────────┴────────────┘

All baselines share the same input format as PI-XGNN:
  x : (B, L, d)   feature window
  t : (B, 1)      normalised time
and expose forward(x, t) -> (pred,)  or  forward(x, t) -> (pred, aux...)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. MLP
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Flat MLP: flatten window → three hidden layers → RUL."""

    def __init__(self, config):
        super().__init__()
        in_dim = config.window_length * config.n_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 32),      nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, t):
        return self.net(x.reshape(x.size(0), -1)), None, None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Stacked LSTM
# ──────────────────────────────────────────────────────────────────────────────

class StackedLSTM(nn.Module):
    """Three-layer stacked LSTM → linear head."""

    def __init__(self, config):
        super().__init__()
        d = config.n_features
        self.lstm1 = nn.LSTM(d,  64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.head  = nn.Linear(64, 1)

    def forward(self, x, t):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        h, _ = self.lstm3(h)
        return self.head(h[:, -1, :]), None, None


# ──────────────────────────────────────────────────────────────────────────────
# 3. CNN-BiLSTM
# ──────────────────────────────────────────────────────────────────────────────

class CNNBiLSTM(nn.Module):
    """1-D CNN feature extractor + Bidirectional LSTM + linear head."""

    def __init__(self, config):
        super().__init__()
        d = config.n_features
        self.cnn = nn.Sequential(
            nn.Conv1d(d, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.bilstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head   = nn.Linear(128, 1)

    def forward(self, x, t):
        # x: (B, L, d) → CNN expects (B, d, L)
        c        = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, L, 128)
        h, _     = self.bilstm(c)                               # (B, L, 128)
        return self.head(h[:, -1, :]), None, None


# ──────────────────────────────────────────────────────────────────────────────
# 4. Transformer encoder
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """Positional-encoded Transformer encoder → linear head."""

    def __init__(self, config):
        super().__init__()
        d   = config.n_features
        self.proj = nn.Linear(d, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head    = nn.Linear(64, 1)

    def forward(self, x, t):
        h = self.encoder(self.proj(x))   # (B, L, 64)
        return self.head(h[:, -1, :]), None, None


# ──────────────────────────────────────────────────────────────────────────────
# 5. AttnPINN  (Liao et al., 2023  —  b15 in the paper)
# Physics: soft monotonicity penalty via softplus
# ──────────────────────────────────────────────────────────────────────────────

class AttnPINN(nn.Module):
    """Self-attention encoder + physics monotonicity loss (softplus)."""

    def __init__(self, config):
        super().__init__()
        d = config.n_features
        self.proj = nn.Linear(d, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, t):
        h    = self.attn(self.proj(x))
        pred = self.head(h[:, -1, :])

        # Physics: penalise non-monotonic segments (softplus)
        if self.training and pred.size(0) > 1:
            mono_loss = F.softplus(pred[1:] - pred[:-1]).mean()
        else:
            mono_loss = pred.new_zeros(1).squeeze()

        return pred, mono_loss, None


# ──────────────────────────────────────────────────────────────────────────────
# 6. GNN baseline  (graph only, no physics)
# Single-layer graph convolution on the correlation adjacency matrix
# ──────────────────────────────────────────────────────────────────────────────

class GNNBaseline(nn.Module):
    """Fixed-threshold graph convolution + LSTM + linear head (no physics)."""

    def __init__(self, config, threshold: float = 0.5):
        super().__init__()
        d = config.n_features
        self.threshold = threshold
        self.gc_proj   = nn.Linear(d, d)          # graph convolution weight
        self.lstm      = nn.LSTM(d, 64, batch_first=True)
        self.head      = nn.Linear(64, 1)

    def _graph_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Single-layer GCN with fixed correlation adjacency."""
        B, L, d = x.shape
        C       = (x.transpose(1, 2) @ x).abs() / L          # (B, d, d)
        I       = torch.eye(d, device=x.device).unsqueeze(0)
        A       = (C > self.threshold).float() * (1 - I) + I  # hard threshold
        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        deg     = A.sum(-1, keepdim=True).clamp(min=1.0)
        A_norm  = A / deg.sqrt() / deg.sqrt().transpose(1, 2)
        # Per-timestep graph convolution
        x_conv  = torch.einsum("bld,bde->ble", x, A_norm)     # (B, L, d)
        return F.relu(self.gc_proj(x_conv))

    def forward(self, x, t):
        h_g      = self._graph_conv(x)
        h, _     = self.lstm(h_g)
        return self.head(h[:, -1, :]), None, None


# ──────────────────────────────────────────────────────────────────────────────
# 7. PI-TENN  (Fang et al., 2026  —  b16 in the paper)
# LSTM + multi-head attention + PDE dual network  (no graph)
# ──────────────────────────────────────────────────────────────────────────────

class PITENN(nn.Module):
    """
    Physics-Informed Temporal-Enhancing Neural Network.
    Flat-vector input; no adaptive graph.
    Enforces the same PDE residual loss as PI-XGNN.
    """

    def __init__(self, config):
        super().__init__()
        d  = config.n_features
        ld = config.latent_dim

        # Temporal encoder (mirrors PI-XGNN encoder, no graph)
        self.lstm1 = nn.LSTM(d,  32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True)

        self.attn  = nn.MultiheadAttention(64, config.n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(64)
        self.ff    = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
        self.norm2 = nn.LayerNorm(64)

        flat_in  = config.window_length * 64
        self.flat = nn.Sequential(
            nn.Linear(flat_in, 128), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 32),      nn.GELU(),
            nn.Linear(32, ld),
        )

        # Physics dual networks (identical to PI-XGNN)
        self.F_net = nn.Sequential(
            nn.Linear(1 + ld, 32), nn.Tanh(),
            nn.Linear(32, 16),     nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.G_net = nn.Sequential(
            nn.Linear(ld + 4, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _encode(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        h, _ = self.lstm3(h)
        a, _ = self.attn(h, h, h)
        h    = self.norm1(h + a)
        h    = self.norm2(h + self.ff(h))
        return self.flat(h.reshape(h.size(0), -1))   # (B, latent_dim)

    def _pde_terms(self, t, h):
        t_v = t.detach().float().requires_grad_(True)
        h_v = h.detach().float().requires_grad_(True)

        u      = self.F_net(torch.cat([t_v, h_v], dim=-1))
        du_dt  = torch.autograd.grad(
            u.sum(), t_v, create_graph=True, retain_graph=True)[0]
        grad_h = torch.autograd.grad(
            u.sum(), h_v, create_graph=True, retain_graph=True)[0]
        s1     = grad_h.sum(-1, keepdim=True)
        grad2  = torch.autograd.grad(grad_h.sum(), h_v, create_graph=True)[0]
        s2     = grad2.sum(-1, keepdim=True)
        g_pred = self.G_net(torch.cat([t_v, h_v, u, s1, s2], dim=-1))
        return u, du_dt, g_pred

    def forward(self, x, t):
        h = self._encode(x)
        u = self.F_net(torch.cat([t, h], dim=-1))
        _, du_dt, g_pred = self._pde_terms(t, h)
        return u, du_dt, g_pred

    @torch.no_grad()
    def predict(self, x, t):
        self.eval()
        h = self._encode(x)
        return self.F_net(torch.cat([t, h], dim=-1))

    def predict_mc(self, x, t, n_samples=100):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                h = self._encode(x)
                preds.append(self.F_net(torch.cat([t, h], dim=-1)))
        stacked = torch.stack(preds)
        return stacked.mean(0), stacked.std(0)


# ──────────────────────────────────────────────────────────────────────────────
# Registry — maps CLI name → (class, uses_pde_loss)
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "mlp":         (MLP,             False),
    "lstm":        (StackedLSTM,     False),
    "cnn_bilstm":  (CNNBiLSTM,       False),
    "transformer": (TransformerEncoder, False),
    "attn_pinn":   (AttnPINN,        True),   # has its own mono loss in forward
    "gnn":         (GNNBaseline,     False),
    "pi_tenn":     (PITENN,          True),
}
