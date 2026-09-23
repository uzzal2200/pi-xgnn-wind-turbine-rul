"""Physics-informed dual network: solution net F and dynamics net G."""
import torch
import torch.nn as nn


class SolutionNet(nn.Module):
    """F(t, h) → û  — predicts the RUL from normalised time and latent code."""

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + latent_dim, 32), nn.Tanh(),
            nn.Linear(32, 16),             nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # t: (B, 1), h: (B, latent_dim)
        return self.net(torch.cat([t, h], dim=-1))       # (B, 1)


class DynamicsNet(nn.Module):
    """G(t, h, û, s₁, s₂) → ∂û/∂t  — predicts the decay rate.

    Input dimension = 1 + latent_dim + 1 + 1 + 1 = latent_dim + 4  (= 12 for d=8)
    where s₁ = Σ_i ∇_h û  and  s₂ = Σ_i ∇²_h û  (scalar reductions).
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        in_dim = latent_dim + 4          # t + h + û + s₁ + s₂
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, t, h, u, s1, s2) -> torch.Tensor:
        return self.net(torch.cat([t, h, u, s1, s2], dim=-1))
