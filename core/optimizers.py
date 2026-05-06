"""Optimizer and learning rate scheduling utilities."""

import math
import torch


def make_optimizer(param, name: str, lr: float):
    """
    Create optimizer for a parameter.

    Args:
        param: Parameter to optimize
        name: Optimizer name ("adam" or "sgd")
        lr: Learning rate

    Returns:
        Optimizer instance
    """
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD([param], lr=lr, momentum=0.9)
    return torch.optim.Adam([param], lr=lr)


def set_lr(opt, lr: float):
    """Set learning rate for all parameter groups in optimizer."""
    for g in opt.param_groups:
        g["lr"] = lr


def compute_scheduled_lr(step_idx: int, total_steps: int, base_lr: float,
                         final_lr: float, warmup_ratio: float) -> float:
    """
    Compute learning rate with warmup (linear) + cosine decay.

    Args:
        step_idx: Current step (1-indexed)
        total_steps: Total number of steps
        base_lr: Peak learning rate
        final_lr: Final learning rate after decay
        warmup_ratio: Fraction of steps for warmup

    Returns:
        Current learning rate
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, min(int(total_steps * warmup_ratio),
                              max(1, total_steps - 1)))

    if step_idx <= warmup_steps:
        # Linear warmup: 0 -> base_lr
        return base_lr * (step_idx / float(warmup_steps))

    # Cosine decay
    t = (step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * t))
