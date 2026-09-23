"""Physics-informed composite loss."""
import torch
import torch.nn.functional as F


def pi_loss(u_pred: torch.Tensor,
            u_true: torch.Tensor,
            du_dt:  torch.Tensor,
            g_pred: torch.Tensor,
            lambda_pde:  float = 0.15,
            lambda_mono: float = 0.15):
    """
    L = L_data + λ_pde · L_pde + λ_mono · L_mono

    Returns
    -------
    total : scalar Tensor
    parts : dict  {'data', 'pde', 'mono'}
    """
    loss_data = F.mse_loss(u_pred, u_true)
    loss_pde  = F.mse_loss(du_dt,  g_pred)

    if u_pred.size(0) > 1:
        loss_mono = F.relu(u_pred[1:] - u_pred[:-1]).mean()
    else:
        loss_mono = u_pred.new_zeros(1).squeeze()

    total = loss_data + lambda_pde * loss_pde + lambda_mono * loss_mono

    return total, {
        "data": loss_data.item(),
        "pde":  loss_pde.item(),
        "mono": loss_mono.item(),
    }
