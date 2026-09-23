"""Adaptive sensor-correlation graph encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphEncoder(nn.Module):
    """
    Encodes inter-sensor relational structure via an adaptive binary adjacency
    matrix whose threshold τ is learned end-to-end (STE back-prop).

    Parameters
    ----------
    d_feat : int  Number of input features (= 12).
    """

    def __init__(self, d_feat: int = 12):
        super().__init__()
        self.d    = d_feat
        # τ initialised so σ(τ) ≈ 0.70 → moderately sparse healthy graph
        self.tau  = nn.Parameter(torch.tensor(0.85))
        self.proj = nn.Linear(d_feat * 2, d_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d)

        Returns
        -------
        out : (B, L, d)  graph-enriched feature sequence
        """
        B, L, d = x.shape

        # ── Unsigned cross-correlation proxy ──────────────────────────────
        C = (x.transpose(1, 2) @ x).abs() / L           # (B, d, d)

        # ── Adaptive binary adjacency with STE ────────────────────────────
        thresh  = torch.sigmoid(self.tau)
        I       = torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0)
        mask_off = 1.0 - I                               # off-diagonal only

        C_off   = C * mask_off
        diff    = C_off - thresh                         # >0 → edge exists

        # Forward: hard indicator; backward: identity (STE)
        A_hard  = (diff > 0).float()
        A_bin   = A_hard.detach() + diff - diff.detach()  # STE trick
        A       = A_bin * mask_off + I                   # add self-loops (B,d,d)

        # ── Normalised degree vector ───────────────────────────────────────
        deg     = A.sum(-1) / d                          # (B, d)
        deg_exp = deg.unsqueeze(1).expand(-1, L, -1)     # (B, L, d)

        # ── Projection ────────────────────────────────────────────────────
        x_aug = torch.cat([x, deg_exp], dim=-1)          # (B, L, 2d)
        return F.gelu(self.proj(x_aug))                  # (B, L, d)
