"""PI-XGNN: Adaptive Physics-Informed Explainable Graph Neural Network."""
import torch
import torch.nn as nn

from .graph_encoder    import AdaptiveGraphEncoder
from .temporal_encoder import TemporalEncoder
from .physics_net      import SolutionNet, DynamicsNet


class PIXGNN(nn.Module):
    """
    Full PI-XGNN model.

    Forward pass (training) returns (û, ∂û/∂t, G_pred) for the
    physics-informed loss computation.

    MC-Dropout inference: call predict_mc() with dropout active.
    """

    def __init__(self, config):
        super().__init__()
        cfg = config
        self.graph   = AdaptiveGraphEncoder(cfg.n_features)
        self.encoder = TemporalEncoder(cfg.n_features, cfg.latent_dim,
                                        cfg.n_heads, cfg.dropout,
                                        cfg.window_length)
        self.F_net   = SolutionNet(cfg.latent_dim)
        self.G_net   = DynamicsNet(cfg.latent_dim)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.graph(x))               # (B, latent_dim)

    def _pde_terms(self, t: torch.Tensor, h: torch.Tensor):
        """Compute û, ∂û/∂t, G̃ on a graph-detached branch (for PDE loss)."""
        t_v = t.detach().float().requires_grad_(True)
        h_v = h.detach().float().requires_grad_(True)

        u   = self.F_net(t_v, h_v)

        du_dt  = torch.autograd.grad(
            u.sum(), t_v, create_graph=True, retain_graph=True)[0]

        grad_h = torch.autograd.grad(
            u.sum(), h_v, create_graph=True, retain_graph=True)[0]
        s1     = grad_h.sum(-1, keepdim=True)

        grad2_h = torch.autograd.grad(
            grad_h.sum(), h_v, create_graph=True)[0]
        s2      = grad2_h.sum(-1, keepdim=True)

        g_pred  = self.G_net(t_v, h_v, u, s1, s2)

        return u, du_dt, g_pred

    # ── Public API ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """Training forward.

        Returns
        -------
        u      : (B, 1) RUL prediction (trains encoder + F via data loss)
        du_dt  : (B, 1) time derivative from F
        g_pred : (B, 1) dynamics network output (trains F + G via PDE loss)
        """
        h     = self._encode(x)
        u     = self.F_net(t, h)             # main prediction path
        _, du_dt, g_pred = self._pde_terms(t, h)
        return u, du_dt, g_pred

    @torch.no_grad()
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Deterministic single-pass prediction (dropout OFF)."""
        self.eval()
        h = self._encode(x)
        return self.F_net(t, h)

    def predict_mc(self, x: torch.Tensor, t: torch.Tensor,
                   n_samples: int = 100):
        """Monte Carlo Dropout inference.

        Returns
        -------
        mu       : (B, 1) predictive mean
        sigma_ep : (B, 1) epistemic standard deviation
        """
        self.train()                         # activate dropout masks
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                h = self._encode(x)
                preds.append(self.F_net(t, h))

        stacked  = torch.stack(preds, dim=0) # (S, B, 1)
        mu       = stacked.mean(0)
        sigma_ep = stacked.std(0)
        return mu, sigma_ep

    def node_importance(self, x: torch.Tensor, t: torch.Tensor,
                        n_samples: int = 100) -> torch.Tensor:
        """Gradient-based node (feature) importance attribution.

        Returns
        -------
        imp : (d,) importance scores normalised to sum 1
        """
        self.train()
        x    = x.requires_grad_(True)
        grads = []
        for _ in range(n_samples):
            h  = self._encode(x)
            u  = self.F_net(t, h)
            g  = torch.autograd.grad(u.sum(), x, retain_graph=False)[0]
            grads.append(g.abs().mean(dim=(0, 1)))   # (d,)

        imp = torch.stack(grads).mean(0)
        return imp / (imp.sum() + 1e-8)
