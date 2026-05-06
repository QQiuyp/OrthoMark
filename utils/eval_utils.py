"""Shared utilities for OrthoMark evaluation scripts."""

import argparse
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.utils import save_image

import config as c
from core import build_carriers, decode_bits_from_t
from utils.helpers import load
from utils.metrics import psnr


ATTACK_LABELS = [
    "RC", "ES", "SH-55", "SH+55", "RO-45", "RO+45",
    "ER", "JP", "MF", "GF", "DP", "SP", "GN",
    "BR0.2", "BR2.0", "CT0.2", "CT2.0",
    "HU-", "HU+", "SA0.2", "SA2.0",
]


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to load.")
    parser.add_argument("--runs", type=int, default=None, help="Number of evaluation runs.")
    parser.add_argument("--output-root", default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--device", type=int, default=None, help="CUDA device index.")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate optimization visualizations.",
    )


def resolve_positive_int(cli_value: Optional[int], default_value: int, name: str) -> int:
    value = default_value if cli_value is None else cli_value
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def resolve_device(device_override: Optional[int]) -> torch.device:
    device_index = c.ndevice if device_override is None else int(device_override)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def resolve_checkpoint(checkpoint_override: Optional[str]) -> Optional[str]:
    if checkpoint_override:
        return checkpoint_override
    if getattr(c, "test_checkpoint_path", ""):
        return c.test_checkpoint_path
    if getattr(c, "tain_next", False):
        return os.path.join(c.MODEL_PATH, c.CONTINUE_PATH, "models", f"{c.CONTINUE_EPOCH}.pt")
    return None


def load_eval_checkpoint(model: torch.nn.Module, checkpoint: Optional[str],
                         device: torch.device, logger: logging.Logger) -> None:
    if checkpoint is None:
        logger.info("[CKPT] No checkpoint configured; evaluating current model weights.")
        return
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    load(model, checkpoint, map_location=device)
    logger.info(f"[CKPT] Loaded {checkpoint}")


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def make_result_folder(script_name: str, output_root: Optional[str]) -> str:
    root = output_root or getattr(c, "test_output_root", "test_results")
    folder_name = time.strftime(f"{c.PROJECT_NAME}_{script_name}__%H_%M_%S", time.localtime())
    result_folder = os.path.join(root, folder_name)
    os.makedirs(result_folder, exist_ok=True)
    return result_folder


def setup_eval_logger(result_folder: str, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    while logger.handlers:
        logger.handlers.pop()

    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = os.path.join(result_folder, "eval.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"[OUT] result_folder={result_folder}")
    return logger


def maybe_seed_eval(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_minus1_to1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return psnr(x, y, 2)


def collect_all_covers(data_loader, device: torch.device) -> torch.Tensor:
    covers = []
    for cover in data_loader:
        if isinstance(cover, dict):
            for key in ("image", "img", "x"):
                if key in cover:
                    cover = cover[key]
                    break
            else:
                raise KeyError("Cover dict must contain one of: image, img, x.")
        covers.append(cover.to(device).float())
    if not covers:
        raise RuntimeError("testloader is empty.")
    return torch.cat(covers, dim=0)


def sample_message(batch_size: int, msg_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (batch_size, msg_len), device=device, dtype=torch.float32)
    message_pm1 = bits.mul(2.0).sub(1.0)
    return message_pm1, bits


def get_carriers(model: torch.nn.Module,
                 cover_all: torch.Tensor,
                 msg_len: int,
                 device: torch.device,
                 cache: Dict[Tuple, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        dec0 = model(cover_all)
    latent_dim = int(np.prod(dec0.shape[1:])) if dec0.dim() == 4 else int(dec0.shape[1])

    v_mode = getattr(c, "V_mode", "ortho")
    v_rho = getattr(c, "V_rho", 0.1)
    key = (latent_dim, msg_len, v_mode, v_rho if v_mode == "rand_unit_corr" else -1.0)
    if key not in cache:
        cache[key] = build_carriers(
            msg_len=msg_len,
            L_dec=latent_dim,
            device=device,
            seed=c.qim_seed,
            mode=v_mode,
            rho=v_rho,
        )
    return cache[key]


def evaluate_attacks(model: torch.nn.Module,
                     test_noise,
                     watermarked: torch.Tensor,
                     cover: torch.Tensor,
                     bits: torch.Tensor,
                     carriers: torch.Tensor) -> List[float]:
    batch_size = watermarked.shape[0]
    noised_outputs = test_noise([watermarked, cover])
    if not isinstance(noised_outputs, (list, tuple)):
        noised_outputs = [noised_outputs]

    acc_values = []
    for noised in noised_outputs:
        dec_out = model(noised)
        tvec = dec_out.reshape(batch_size, -1) @ carriers.t()
        bits_hat = decode_bits_from_t(tvec, mode=c.test_loss_mode, Delta=c.qim_Delta)
        error_rate = (bits_hat.round() != bits).float().mean().item()
        acc_values.append((1.0 - error_rate) * 100.0)
    return acc_values


def format_acc_log(acc_values: List[float]) -> str:
    parts = [
        f"{ATTACK_LABELS[i] if i < len(ATTACK_LABELS) else f'ATTACK{i}'}: {value:.2f}"
        for i, value in enumerate(acc_values)
    ]
    if acc_values:
        parts.append(f"AVG: {float(np.mean(acc_values)):.2f}")
    return " | ".join(parts)


def lm_latex_row(method_name: str, psnr_val: float, acc_values: List[float]) -> Optional[str]:
    if len(acc_values) < 21:
        return None

    cr = acc_values[0]
    elas = acc_values[1]
    shear = 0.5 * (acc_values[2] + acc_values[3])
    rot = 0.5 * (acc_values[4] + acc_values[5])
    er = acc_values[6]
    jpeg = acc_values[7]
    mf = acc_values[8]
    gf = acc_values[9]
    dp = acc_values[10]
    sp = acc_values[11]
    gn = acc_values[12]
    br = 0.5 * (acc_values[13] + acc_values[14])
    ct = 0.5 * (acc_values[15] + acc_values[16])
    hue = 0.5 * (acc_values[17] + acc_values[18])
    sa = 0.5 * (acc_values[19] + acc_values[20])

    vals = [jpeg, mf, gf, dp, sp, gn, er, cr, shear, rot, elas, hue, br, ct, sa]
    avg = float(np.mean(vals))

    return (
        f"{method_name} & {psnr_val:.2f} "
        f"& {jpeg:.2f} & {mf:.2f} & {gf:.2f} & {dp:.2f} & {sp:.2f} & {gn:.2f} "
        f"& {er:.2f} & {cr:.2f} & {shear:.2f} & {rot:.2f} & {elas:.2f} "
        f"& {hue:.2f} & {br:.2f} & {ct:.2f} & {sa:.2f} & {avg:.2f} \\\\"
    )


def log_and_print_result(logger: logging.Logger, run_idx: int, psnr_val: float,
                         acc_values: List[float]) -> None:
    logger.info(f"[TEST] run={run_idx} | PSNR={psnr_val:.2f} | {format_acc_log(acc_values)}")
    latex_row = lm_latex_row(c.method_name, psnr_val, acc_values)
    if latex_row is not None:
        print("\n%%%%%%%%%%%% LaTeX Row %%%%%%%%%%%%")
        print(latex_row)
        print("=" * 60 + "\n")


def _first_grid_batch(x: torch.Tensor, max_images: int = 10) -> Tuple[torch.Tensor, int]:
    count = min(int(max_images), int(x.shape[0]))
    nrow = min(5, count)
    return x[:count], nrow


def save_final_visuals(result_folder: str, run_idx: int,
                       watermarked: torch.Tensor, cover: torch.Tensor,
                       prefix: str = "run") -> None:
    out_dir = os.path.join(result_folder, "vis_test")
    os.makedirs(out_dir, exist_ok=True)
    residual, nrow = _first_grid_batch(((watermarked - cover) * 10.0 + 1.0) / 2.0)
    stego, _ = _first_grid_batch((watermarked.clamp(-1, 1) + 1.0) / 2.0)
    name = f"{prefix}{run_idx:03d}"
    save_image(residual.clamp(0, 1), os.path.join(out_dir, f"{name}_residual_grid.png"), nrow=nrow)
    save_image(stego.clamp(0, 1), os.path.join(out_dir, f"{name}_stego_grid.png"), nrow=nrow)


def save_intermediate_visuals(result_folder: str, run_idx: int, step_idx: int,
                              x: torch.Tensor, cover: torch.Tensor,
                              latent: torch.Tensor) -> None:
    base_dir = os.path.join(result_folder, "generate_test")
    for subdir in ("residual", "watermarked", "latent"):
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    prefix = f"run{run_idx:03d}_step{step_idx:04d}.png"

    residual, nrow = _first_grid_batch(((x - cover) * 10.0 + 1.0) / 2.0)
    watermarked, _ = _first_grid_batch((x + 1.0) / 2.0)
    latent_vis, _ = _first_grid_batch((latent + 1.0) / 2.0)

    save_image(residual.clamp(0, 1), os.path.join(base_dir, "residual", prefix), nrow=nrow)
    save_image(watermarked.clamp(0, 1), os.path.join(base_dir, "watermarked", prefix), nrow=nrow)
    save_image(latent_vis.clamp(0, 1), os.path.join(base_dir, "latent", prefix), nrow=nrow)
