"""Orthogonal carrier generation for watermark embedding."""

import torch
import torch.nn.functional as F


def orthonormal_carriers(B: int, L: int, device, seed: int = 12345) -> torch.Tensor:
    """
    Return V: [B, L] with orthonormal rows (v_i · v_j = δ_ij).

    Args:
        B: Number of carriers (message length)
        L: Carrier dimension
        device: torch device
        seed: Random seed for reproducibility

    Returns:
        V: [B, L] orthonormal carrier matrix
    """
    if B > L:
        raise ValueError(f"B ({B}) must be <= L ({L}).")
    g = torch.randn(
        L, B,
        device=device,
        dtype=torch.float32,
        generator=torch.Generator(device=device).manual_seed(seed)
    )
    Q, _ = torch.linalg.qr(g, mode='reduced')  # [L,B]
    V = Q.transpose(0, 1).contiguous()         # [B,L]
    return V


def make_ortho(msg_len: int, L_dec: int, device, seed: int = 12345) -> torch.Tensor:
    """
    Generate orthonormal carrier matrix using QR decomposition.

    Args:
        msg_len: Message length (number of carriers)
        L_dec: Decoder output dimension
        device: torch device
        seed: Random seed

    Returns:
        V: [msg_len, L_dec] orthonormal carrier matrix
    """
    if msg_len > L_dec:
        raise ValueError(f"msg_len ({msg_len}) must be <= L_dec ({L_dec}).")
    g = torch.Generator(device=device).manual_seed(int(seed))
    A = torch.randn(L_dec, msg_len, device=device, dtype=torch.float32, generator=g)
    Q, _ = torch.linalg.qr(A, mode="reduced")   # [L_dec, msg_len]
    V = Q.transpose(0, 1).contiguous()          # [msg_len, L_dec]
    return V


def make_rand_unit(msg_len: int, L_dec: int, device, seed: int = 12345) -> torch.Tensor:
    """
    Generate random unit-norm carriers (not necessarily orthogonal).

    Args:
        msg_len: Message length
        L_dec: Decoder output dimension
        device: torch device
        seed: Random seed

    Returns:
        V: [msg_len, L_dec] unit-norm carrier matrix
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    V = torch.randn(msg_len, L_dec, device=device, dtype=torch.float32, generator=g)
    V = F.normalize(V, p=2, dim=1)
    return V


def make_rand_unit_corr(msg_len: int, L_dec: int, device,
                        seed: int = 12345, rho: float = 0.1) -> torch.Tensor:
    """
    Generate unit-norm carriers with controlled correlation.

    Args:
        msg_len: Message length
        L_dec: Decoder output dimension
        device: torch device
        seed: Random seed
        rho: Correlation coefficient [0, 1)

    Returns:
        V: [msg_len, L_dec] carrier matrix with controlled correlation
    """
    rho = float(rho)
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"V_rho must be in [0,1). got {rho}")

    g = torch.Generator(device=device).manual_seed(int(seed))

    u = torch.randn(L_dec, device=device, dtype=torch.float32, generator=g)
    u = u / (u.norm() + 1e-12)

    eps = make_ortho(msg_len, L_dec, device, seed=int(seed) + 1)  # already [msg_len, L_dec]

    # mix + renorm
    V = rho * u.unsqueeze(0) + (1.0 - rho) * eps
    V = V / (V.norm(dim=1, keepdim=True) + 1e-12)
    return V


def build_carriers(msg_len: int,
                   L_dec: int,
                   device,
                   seed: int = 12345,
                   mode: str = "ortho",
                   rho: float = 0.1) -> torch.Tensor:
    """
    Build carrier matrix based on specified mode.

    Args:
        msg_len: Message length
        L_dec: Decoder output dimension
        device: torch device
        seed: Random seed
        mode: Carrier mode ("ortho", "rand_unit", "rand_unit_corr")
        rho: Correlation coefficient (only for "rand_unit_corr")

    Returns:
        V: [msg_len, L_dec] carrier matrix
    """
    mode = (mode or "ortho").lower()
    if msg_len > L_dec:
        raise ValueError(f"msg_len ({msg_len}) must be <= L_dec ({L_dec}).")

    if mode == "ortho":
        return make_ortho(msg_len, L_dec, device, seed)
    elif mode == "rand_unit":
        return make_rand_unit(msg_len, L_dec, device, seed)
    elif mode == "rand_unit_corr":
        return make_rand_unit_corr(msg_len, L_dec, device, seed, rho=rho)
    else:
        raise ValueError(f"Unknown V_mode: {mode}. Use 'ortho'/'rand_unit'/'rand_unit_corr'.")
