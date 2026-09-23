"""Training loop with multi-seed best-of-N strategy."""
import copy
import torch
import torch.optim as optim

from .losses import pi_loss


def train_one_seed(model, train_loader, config, device, verbose: bool = True):
    """Train for config.epochs. Returns (model, final_avg_loss)."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),
                             lr=config.lr,
                             weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    for epoch in range(config.epochs):
        model.train()
        total, n_batch = 0.0, 0

        for x, y, t in train_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            optimizer.zero_grad()

            u, du_dt, g_pred = model(x, t)
            loss, _ = pi_loss(u, y, du_dt, g_pred,
                               config.lambda_pde, config.lambda_mono)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total  += loss.item()
            n_batch += 1

        scheduler.step()
        avg = total / max(n_batch, 1)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1:3d}/{config.epochs}  loss={avg:.4f}")

    return model, avg


def train_with_seeds(model_class, model_kwargs, train_loader, config, device):
    """Train n_seeds times; return model with lowest final training loss."""
    best_loss  = float("inf")
    best_state = None

    for seed in range(config.n_seeds):
        torch.manual_seed(seed)
        model = model_class(**model_kwargs)
        print(f"  [Seed {seed + 1}/{config.n_seeds}]")
        model, loss = train_one_seed(model, train_loader, config, device)

        if loss < best_loss:
            best_loss  = loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"    → new best  loss={best_loss:.4f}")

    best = model_class(**model_kwargs).to(device)
    best.load_state_dict(best_state)
    return best
