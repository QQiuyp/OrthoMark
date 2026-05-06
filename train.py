"""OrthoMark Training Script."""

import os
import re
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config as c
from utils.helpers import load, setup_logger
from utils.metrics import psnr
from utils.datasets import trainloader, testloader
from utils.eval_utils import save_final_visuals
from models.orthomark import OrthoMark
from core import (
    orthonormal_carriers,
    compute_embed_loss,
    decode_bits_from_t,
    build_noises,
    make_optimizer,
    set_lr,
    compute_scheduled_lr,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Helper Functions
# =============================================================================

def find_latest_ckpt(models_dir: str):
    """Find latest checkpoint in models_dir with integer filename (e.g., '10.pt')."""
    p = Path(models_dir)
    if not p.exists():
        return 0, None

    best_e, best_path = -1, None
    for f in p.glob("*.pt"):
        if re.fullmatch(r"\d+", f.stem):
            e = int(f.stem)
            if e > best_e:
                best_e, best_path = e, str(f)
    return best_e, best_path


def setup_logger_append(name: str, save_dir: str, log_subdir: str = "logging",
                        filename: str = "train.log", level=logging.INFO, screen=True):
    """Create a logger that appends to the same log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    while logger.handlers:
        logger.handlers.pop()

    os.makedirs(os.path.join(save_dir, log_subdir), exist_ok=True)
    log_path = os.path.join(save_dir, log_subdir, filename)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    if screen:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 80)
    logger.info(f"Logging to {log_path} (append)")
    return logger


def psnr_minus1_to1(x, y):
    """PSNR for images in [-1, 1] range."""
    return psnr(x, y, 2)


def msg_to_bits(m: torch.Tensor) -> torch.Tensor:
    """Convert message from {-1, 1} to bits {0, 1}."""
    return (m > 0).to(torch.float32)


def take_first_if_list(x):
    """Extract first element if input is a list/tuple."""
    return x[0] if isinstance(x, (list, tuple)) else x


def format_acc_log(acc, num_attacks=21):
    """Format accuracy list for logging."""
    labels = [
        'RC', 'ES', 'SH-55', 'SH+55', 'RO-45', 'RO+45',
        'ER', 'JP', 'MF', 'GF', 'DP', 'SP', 'GN',
        'BR0.2', 'BR2.0', 'CT0.2', 'CT2.0',
        'HU-', 'HU+', 'SA0.2', 'SA2.0'
    ]
    parts = [f'{labels[i]}: {np.mean(acc[i]):.2f}' for i in range(min(num_attacks, len(acc)))]
    avg = np.mean([np.mean(acc[i]) for i in range(num_attacks)])
    parts.append(f'AVG: {avg:.2f}')
    return ' | '.join(parts)


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    device = torch.device(f"cuda:{c.ndevice}" if torch.cuda.is_available() else "cpu")

    # Initialize model
    pmodel = OrthoMark().to(device)
    for p in pmodel.parameters():
        p.requires_grad_(True)

    # Setup result folder and resume
    if getattr(c, "tain_next", False):
        result_folder = os.path.join(c.MODEL_PATH, c.CONTINUE_PATH)
        os.makedirs(os.path.join(result_folder, "models"), exist_ok=True)
        logger = setup_logger_append("train", result_folder)

        latest_e, latest_ckpt = find_latest_ckpt(os.path.join(result_folder, "models"))
        if latest_ckpt is None:
            start_epoch = 1
            logger.info(f"[RESUME] No ckpt found, start from epoch=1")
        else:
            load(pmodel, latest_ckpt)
            start_epoch = latest_e + 1
            logger.info(f"[RESUME] Loaded {latest_ckpt}, start_epoch={start_epoch}")
    else:
        result_folder = os.path.join(
            c.MODEL_PATH, time.strftime(c.PROJECT_NAME + "__%H_%M_%S", time.localtime())
        )
        os.makedirs(os.path.join(result_folder, "models"), exist_ok=True)
        logger = setup_logger_append("train", result_folder)
        start_epoch = 1
        logger.info(f"[NEW] result_folder={result_folder}")

    # Build noise layers
    test_noise, train_noise, test_train_noise = build_noises(c.noise_type)

    # Decoder optimizer
    dec_params = list(filter(lambda p: p.requires_grad, pmodel.parameters()))
    opt_dec = torch.optim.Adam(dec_params, lr=c.dec_lr)

    # Carrier cache
    V_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    # ==========================================================================
    # Epoch Loop
    # ==========================================================================
    for epoch in range(start_epoch, c.epochs + 1):

        # ----------------------------------------------------------------------
        # TRAIN
        # ----------------------------------------------------------------------
        pmodel.train()
        train_psnrs = []
        acc = [[] for _ in range(21)]

        for cover in trainloader:
            if isinstance(cover, dict):
                cover = cover.get("image") or cover.get("img") or cover.get("x")
            cover = cover.to(device).float()
            B = cover.shape[0]

            # Generate message
            msg_len = int(c.message_length)
            message = torch.tensor(
                np.random.choice([-1, 1], (B, msg_len)), device=device, dtype=torch.float32
            )
            bits = msg_to_bits(message)

            # Build carrier matrix V
            with torch.no_grad():
                y0 = take_first_if_list(train_noise([cover, cover]))
                dec0 = pmodel(y0) if c.mode != 'PH' else pmodel(y0)[0]
            L_dec = int(np.prod(dec0.shape[1:])) if dec0.dim() == 4 else int(dec0.shape[1])

            key = (L_dec, msg_len)
            if key not in V_cache:
                V_cache[key] = orthonormal_carriers(msg_len, L_dec, device=device, seed=c.qim_seed)
            V = V_cache[key]

            # Initialize watermarked image x
            x = cover.clone().detach().requires_grad_(True)
            opt_x = make_optimizer(x, c.adv_opt_name, c.adv_lr)

            # Joint optimization loop
            for step_i in range(1, c.joint_steps + 1):
                lr_now = compute_scheduled_lr(
                    step_i, c.joint_steps, c.adv_lr, c.adv_lr_final, c.adv_warmup_ratio
                )
                set_lr(opt_x, lr_now)

                opt_x.zero_grad(set_to_none=True)
                opt_dec.zero_grad(set_to_none=True)

                # Forward through noise and decoder
                yk = take_first_if_list(train_noise([x, cover]))
                noise_dec_out = pmodel(yk) if c.mode != 'PH' else pmodel(yk.detach())[0]
                noise_tvec = noise_dec_out.reshape(B, -1) @ V.t()

                # Compute loss
                embed_loss = compute_embed_loss(
                    noise_tvec=noise_tvec, clean_tvec=noise_tvec, bits=bits,
                    message_pm1=message, mode=c.loss_mode, Delta=c.qim_Delta,
                    epoch=epoch, mix_epochs=c.mix_epochs
                )
                loss = c.acc_w * embed_loss
                if c.adv_mse_w > 0:
                    loss = loss + c.adv_mse_w * F.mse_loss(x, cover)

                loss.backward()

                # Gradient clipping
                if c.adv_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([x], max_norm=c.adv_grad_clip)
                if c.dec_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(dec_params, max_norm=c.dec_grad_clip)

                opt_x.step()
                opt_dec.step()

                with torch.no_grad():
                    x.clamp_(-1, 1)

            # Evaluate batch
            with torch.no_grad():
                train_psnrs.append(float(psnr_minus1_to1(cover, x)))

                for idx, noised in enumerate(test_noise([x, cover])):
                    dec_out = pmodel(noised)
                    tvec = dec_out.reshape(B, -1) @ V.t()
                    bits_hat = decode_bits_from_t(tvec, mode=c.test_loss_mode, Delta=c.qim_Delta)
                    error_rate = (bits_hat.round() != bits).float().mean().item()
                    acc[idx].append((1 - error_rate) * 100)

            print(f"TRAIN: PSNR={np.mean(train_psnrs):.2f} | {format_acc_log(acc)}")

        logger.info(f"[TRAIN] epoch={epoch} | PSNR={np.mean(train_psnrs):.2f} | {format_acc_log(acc)}")

        # ----------------------------------------------------------------------
        # TEST
        # ----------------------------------------------------------------------
        pmodel.eval()
        test_psnrs = []
        acc = [[] for _ in range(21)]

        for cover in testloader:
            if isinstance(cover, dict):
                cover = cover.get("image") or cover.get("img") or cover.get("x")
            cover = cover.to(device).float()
            B = cover.shape[0]

            msg_len = int(c.test_message_length)
            message = torch.tensor(
                np.random.choice([-1, 1], (B, msg_len)), device=device, dtype=torch.float32
            )
            bits = msg_to_bits(message)

            # Build V
            with torch.no_grad():
                y0 = take_first_if_list(train_noise([cover, cover]))
                dec0 = pmodel(y0) if c.mode != 'PH' else pmodel(y0)[0]
            L_dec = int(np.prod(dec0.shape[1:])) if dec0.dim() == 4 else int(dec0.shape[1])

            key = (L_dec, msg_len)
            if key not in V_cache:
                V_cache[key] = orthonormal_carriers(msg_len, L_dec, device=device, seed=c.qim_seed)
            V = V_cache[key]

            # Optimize x only (decoder frozen)
            x = cover.clone().detach().requires_grad_(True)
            opt_x = make_optimizer(x, c.adv_opt_name, c.adv_lr)

            for step_i in range(1, c.test_joint_steps + 1):
                lr_now = compute_scheduled_lr(
                    step_i, c.test_joint_steps, c.adv_lr, c.adv_lr_final, c.adv_warmup_ratio
                )
                set_lr(opt_x, lr_now)
                opt_x.zero_grad(set_to_none=True)

                yk = take_first_if_list(test_train_noise([x, cover]))
                noise_dec_out = pmodel(yk) if c.mode != 'PH' else pmodel(yk)[0]
                noise_tvec = noise_dec_out.reshape(B, -1) @ V.t()

                loss_x = compute_embed_loss(
                    noise_tvec=noise_tvec, clean_tvec=noise_tvec, bits=bits,
                    message_pm1=message, mode=c.test_loss_mode, Delta=c.qim_Delta,
                    epoch=epoch, mix_epochs=c.mix_epochs
                )
                if c.adv_mse_w > 0:
                    loss_x = c.acc_w * loss_x + c.adv_mse_w * F.mse_loss(x, cover)

                loss_x.backward()
                if c.adv_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([x], max_norm=c.adv_grad_clip)
                opt_x.step()

                with torch.no_grad():
                    x.clamp_(-1, 1)

            # Evaluate
            with torch.no_grad():
                test_psnrs.append(float(psnr_minus1_to1(cover, x)))

                for idx, noised in enumerate(test_noise([x, cover])):
                    dec_out = pmodel(noised)
                    tvec = dec_out.reshape(B, -1) @ V.t()
                    bits_hat = decode_bits_from_t(tvec, mode=c.test_loss_mode, Delta=c.qim_Delta)
                    error_rate = (bits_hat.round() != bits).float().mean().item()
                    acc[idx].append((1 - error_rate) * 100)

                # Save visualization
                if (epoch - 1) % 10 == 0:
                    save_final_visuals(result_folder, epoch, x, cover, prefix="epoch")

        logger.info(f"[TEST ] epoch={epoch} | PSNR={np.mean(test_psnrs):.2f} | {format_acc_log(acc)}")

        # Save checkpoint
        if epoch % c.SAVE_freq == 0:
            torch.save(pmodel.state_dict(), os.path.join(result_folder, "models", f"{epoch}.pt"))

    # Final save
    torch.save(pmodel.state_dict(), os.path.join(result_folder, "models", f"{c.epochs}.pt"))


if __name__ == "__main__":
    main()
