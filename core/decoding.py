"""Watermark decoding functions."""

import torch


@torch.no_grad()
def decode_bits_from_t_qim(t: torch.Tensor, Delta: float) -> torch.Tensor:
    """
    Hard decision to bits {0,1} using nearest coset (QIM).

    0-bit coset: 0, ±2Δ, ±4Δ, ...
    1-bit coset: Δ/2, ±(3Δ/2), ...

    Args:
        t: Projected values
        Delta: QIM step size

    Returns:
        bits: Decoded bits ∈ {0,1} float
    """
    d0 = torch.abs(t - Delta * torch.round(t / Delta))
    d1 = torch.abs(t - (Delta * (torch.round((t - 0.5 * Delta) / Delta) + 0.5)))
    return (d1 < d0).to(torch.float32)


@torch.no_grad()
def decode_bits_from_t_sign(t: torch.Tensor) -> torch.Tensor:
    """
    Hard decision to bits {0,1} using single boundary at 0.

    Args:
        t: Projected values

    Returns:
        bits: Decoded bits ∈ {0,1} float
    """
    return (t > 0).to(torch.float32)


@torch.no_grad()
def decode_bits_from_t(tvec: torch.Tensor,
                       mode: str,
                       Delta: float) -> torch.Tensor:
    """
    Unified decoding wrapper.

    Args:
        tvec: Projected values
        mode: Decoding mode ("qim" or "mse")
        Delta: QIM step size (used for "qim" mode)

    Returns:
        bits_hat: Decoded bits ∈ {0,1} float
    """
    mode = mode.lower()
    if mode == "qim":
        return decode_bits_from_t_qim(tvec, Delta)
    elif mode == "mse":
        return decode_bits_from_t_sign(tvec)
    else:
        raise ValueError(f"Unknown loss_mode: {mode}")
