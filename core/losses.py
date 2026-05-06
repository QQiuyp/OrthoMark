"""Loss functions for watermark training."""

import math
import torch
import torch.nn.functional as F


def periodic_loss_from_t(t: torch.Tensor, bits: torch.Tensor, Delta: float) -> torch.Tensor:
    """
    QIM-style MSE to nearest coset.

    0's coset: {...,-4,-2,0,2,4,...}
    1's coset: {...,-3,-1,1,3,5,...}

    Args:
        t: Projected values [B, msg_len]
        bits: Target bits [B, msg_len] ∈ {0,1}
        Delta: QIM step size

    Returns:
        loss: MSE loss to nearest coset point
    """
    cst = bits * (0.5 * Delta)     # 0 -> 0, 1 -> Δ/2
    z = (t - cst) / Delta
    k = torch.round(z)
    qim_t = cst + Delta * k
    return F.mse_loss(t, qim_t)


def cos_periodic_loss_from_t(t: torch.Tensor, bits: torch.Tensor, Delta: float) -> torch.Tensor:
    """
    Deep QIM periodic loss (no dither).

    L = mean(1 - cos(2π*((t - c_b)/Δ))), c_b∈{0, Δ/2}, bits∈{0,1}

    Args:
        t: Projected values [B, msg_len]
        bits: Target bits [B, msg_len] ∈ {0,1}
        Delta: QIM step size

    Returns:
        loss: Cosine periodic loss
    """
    two_pi = 2.0 * math.pi
    bits = bits.to(device=t.device, dtype=torch.float32)
    cst = bits * (0.5 * Delta)
    z = (t - cst) / Delta
    return (1.0 - torch.cos(two_pi * z)).mean()


def compute_embed_loss(noise_tvec: torch.Tensor,
                       clean_tvec: torch.Tensor,
                       bits: torch.Tensor,
                       message_pm1: torch.Tensor,
                       mode: str,
                       Delta: float,
                       epoch: int = None,
                       mix_epochs: int = None) -> torch.Tensor:
    """
    Unified embedding loss wrapper.

    Args:
        noise_tvec: Projected values after noise [B, msg_len]
        clean_tvec: Projected values without noise [B, msg_len]
        bits: Target bits [B, msg_len] ∈ {0,1}
        message_pm1: Message in ±1 format [B, msg_len]
        mode: Loss mode ("qim", "mse", or "mix")
        Delta: QIM step size
        epoch: Current epoch (for "mix" mode)
        mix_epochs: Number of epochs for mixing transition (for "mix" mode)

    Returns:
        loss: Computed embedding loss
    """
    mode = mode.lower()
    if mode == "qim":
        return cos_periodic_loss_from_t(noise_tvec, bits, Delta)

    elif mode == "mse":
        return F.mse_loss(noise_tvec, message_pm1.to(device=noise_tvec.device, dtype=noise_tvec.dtype))

    elif mode == "mix":
        # Mix from MSE to QIM over mix_epochs
        q_loss = cos_periodic_loss_from_t(noise_tvec, bits, Delta)
        m_loss = F.mse_loss(noise_tvec, message_pm1)

        if epoch is None or mix_epochs is None or mix_epochs <= 0:
            return q_loss + m_loss

        # Linear schedule: epoch = 1..mix_epochs
        # epoch=1: w_mse ~ 1, w_qim ~ 0
        # epoch=mix_epochs: w_mse ~ 0, w_qim ~ 1
        alpha = max(0.0, min(1.0, epoch / float(mix_epochs)))
        w_qim = alpha
        w_mse = 1.0 - alpha
        return w_mse * m_loss + w_qim * q_loss

    else:
        raise ValueError(f"Unknown loss_mode: {mode}")
