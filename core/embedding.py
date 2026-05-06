"""QIM watermark embedding functions."""

import torch


def qim_embed(latent_z: torch.Tensor,
              message: torch.Tensor,
              V: torch.Tensor,
              Delta: float):
    """
    Embed watermark into latent_z using QIM.

    Args:
        latent_z: [B, ...] latent representation
        message: [B, msg_len] ∈ {0,1} message bits
        V: [msg_len, L_dec] orthogonal carriers
        Delta: QIM step size

    Returns:
        watermarked_latent_z: watermarked latent representation (same shape as latent_z)
    """
    B = latent_z.size(0)
    z_flat = latent_z.view(B, -1)          # [B, L_dec]
    L_dec = z_flat.size(1)
    msg_len = message.size(1)
    assert V.shape == (msg_len, L_dec), "V shape mismatch."

    # Project to orthogonal space: t = z * V^T
    t = z_flat @ V.t()                     # [B, msg_len]

    # QIM quantization (embed message)
    cst = message * (0.5 * Delta)          # [B, msg_len], 0 or Δ/2
    z = (t - cst) / Delta                  # normalized coordinate before quantization

    k = torch.round(z)
    qim_t = cst + Delta * k                # quantized t', i.e., qim_m

    delta_t = qim_t - t                    # [B, msg_len]
    # Pull back to latent space: Δz = delta_t * V
    z_flat_w = z_flat + delta_t @ V        # [B, L_dec]
    watermarked_latent_z = z_flat_w.view_as(latent_z)

    return watermarked_latent_z


def calculate_z(latent_z: torch.Tensor,
                bits: torch.Tensor,
                Delta: float,
                V: torch.Tensor) -> torch.Tensor:
    """
    QIM watermark embedding, returns watermarked latent_z.

    Args:
        latent_z: (B, 3, 128, 128) original latent representation
        bits: (B, msg_len) bits to embed (0 or 1)
        Delta: QIM quantization step size
        V: (msg_len, L) orthogonal carrier matrix, L = 3*128*128

    Returns:
        watermarked_z: (B, 3, 128, 128) watermarked latent representation
    """
    B = latent_z.shape[0]
    original_shape = latent_z.shape

    # 1. Flatten latent_z: (B, 3, 128, 128) -> (B, L)
    z_flat = latent_z.reshape(B, -1)  # (B, L)

    # 2. Project to carrier space: y = z @ V.T, (B, L) @ (L, msg_len) -> (B, msg_len)
    y = z_flat @ V.t()  # (B, msg_len)

    # 3. QIM embedding
    # bit=0: quantize to k*Delta (nearest integer multiple)
    # bit=1: quantize to (k+0.5)*Delta (nearest half-integer multiple)
    y_0 = Delta * torch.round(y / Delta)
    y_1 = Delta * (torch.round((y - 0.5 * Delta) / Delta) + 0.5)
    y_wm = torch.where(bits == 0, y_0, y_1)

    # 4. Compute change in y
    delta_y = y_wm - y  # (B, msg_len)

    # 5. Pull back change to latent_z: delta_z = delta_y @ V
    # Since V is orthogonal (V @ V.T = I):
    # (z + delta_y @ V) @ V.T = z @ V.T + delta_y @ V @ V.T = y + delta_y = y_wm
    delta_z = delta_y @ V  # (B, msg_len) @ (msg_len, L) -> (B, L)

    # 6. Get watermarked z
    z_wm_flat = z_flat + delta_z

    # 7. Reshape back to original shape
    watermarked_z = z_wm_flat.reshape(original_shape)

    return watermarked_z
