"""Stacked LSTM + Multi-Head Attention temporal encoder."""
import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Three-layer stacked LSTM → 4-head self-attention → nonlinear flat MLP
    that compresses the full sequence to an 8-dimensional degradation state h.

    Parameters
    ----------
    d_feat    : int  Input feature dimension (12).
    latent_dim: int  Output code dimension   (8).
    n_heads   : int  Attention heads         (4).
    dropout   : float Dropout probability    (0.20).
    window    : int  Sliding-window length   (5).
    """

    def __init__(self, d_feat: int = 12, latent_dim: int = 8,
                 n_heads: int = 4, dropout: float = 0.20, window: int = 5):
        super().__init__()

        # ── LSTM stack ─────────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(d_feat, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32,     64, batch_first=True)
        self.lstm3 = nn.LSTM(64,     64, batch_first=True)

        # ── Multi-head attention ───────────────────────────────────────────
        self.attn  = nn.MultiheadAttention(64, n_heads, batch_first=True,
                                            dropout=0.0)
        self.norm1 = nn.LayerNorm(64)
        self.ff    = nn.Sequential(nn.Linear(64, 128), nn.GELU(),
                                    nn.Linear(128, 64))
        self.norm2 = nn.LayerNorm(64)

        # ── Flat MLP → latent code ─────────────────────────────────────────
        flat_in = window * 64
        self.flat = nn.Sequential(
            nn.Linear(flat_in, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),     nn.GELU(),
            nn.Linear(32, latent_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d)

        Returns
        -------
        h : (B, latent_dim)  degradation state code
        """
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        h, _ = self.lstm3(h)                             # (B, L, 64)

        a, _ = self.attn(h, h, h)
        h    = self.norm1(h + a)
        h    = self.norm2(h + self.ff(h))

        h_flat = self.drop(h.reshape(h.size(0), -1))    # (B, L*64)
        return self.flat(h_flat)                         # (B, latent_dim)
