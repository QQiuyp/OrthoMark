"""OrthoMark optimization-based evaluation script."""

import argparse
import warnings

import torch
import torch.nn.functional as F

import config as c
from core import build_test_noises, compute_embed_loss, compute_scheduled_lr, make_optimizer, set_lr
from models.orthomark import OrthoMark
from utils.datasets import testloader
from utils.eval_utils import (
    add_common_eval_args,
    collect_all_covers,
    evaluate_attacks,
    freeze_model,
    get_carriers,
    load_eval_checkpoint,
    log_and_print_result,
    make_result_folder,
    maybe_seed_eval,
    psnr_minus1_to1,
    resolve_checkpoint,
    resolve_device,
    resolve_positive_int,
    sample_message,
    save_final_visuals,
    save_intermediate_visuals,
    setup_eval_logger,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OrthoMark with test-time image optimization.")
    add_common_eval_args(parser)
    parser.add_argument("--steps", type=int, default=None, help="Optimization steps per run.")
    return parser.parse_args()


def main():
    args = parse_args()
    maybe_seed_eval(getattr(c, "test_seed", None))

    device = resolve_device(args.device)
    runs = resolve_positive_int(args.runs, getattr(c, "test_runs", 1), "runs")
    test_steps = resolve_positive_int(args.steps, c.test_joint_steps, "steps")
    save_intermediate = bool(getattr(c, "test_save_intermediate", False) or args.save_intermediate)
    vis_interval = resolve_positive_int(None, getattr(c, "test_vis_interval", 100), "test_vis_interval")

    result_folder = make_result_folder("test", args.output_root)
    logger = setup_eval_logger(result_folder, "orthomark.test")
    logger.info(f"[CFG] device={device} | runs={runs} | steps={test_steps}")

    model = OrthoMark().to(device)
    freeze_model(model)
    load_eval_checkpoint(model, resolve_checkpoint(args.checkpoint), device, logger)

    test_noise, test_train_noise = build_test_noises(c.noise_type)

    cover_all = collect_all_covers(testloader, device)
    batch_size = cover_all.shape[0]
    msg_len = int(c.test_message_length)
    carrier_cache = {}
    carriers = get_carriers(model, cover_all, msg_len, device, carrier_cache)
    logger.info(f"[DATA] covers={batch_size} | msg_len={msg_len} | carrier_shape={tuple(carriers.shape)}")

    for run_idx in range(1, runs + 1):
        message_pm1, bits = sample_message(batch_size, msg_len, device)
        x = cover_all.clone().detach().requires_grad_(True)
        opt_x = make_optimizer(x, c.adv_opt_name, c.adv_lr)

        for step_idx in range(1, test_steps + 1):
            lr_now = compute_scheduled_lr(
                step_idx, test_steps, c.adv_lr, c.adv_lr_final, c.adv_warmup_ratio
            )
            set_lr(opt_x, lr_now)
            opt_x.zero_grad(set_to_none=True)

            noised = test_train_noise([x, cover_all])
            if isinstance(noised, (list, tuple)):
                noised = noised[0]
            noise_dec_out = model(noised)
            noise_tvec = noise_dec_out.reshape(batch_size, -1) @ carriers.t()

            loss_x = compute_embed_loss(
                noise_tvec=noise_tvec,
                clean_tvec=noise_tvec,
                bits=bits,
                message_pm1=message_pm1,
                mode=c.test_loss_mode,
                Delta=c.qim_Delta,
            )
            if c.adv_mse_w > 0:
                loss_x = c.acc_w * loss_x + c.adv_mse_w * F.mse_loss(x, cover_all)

            loss_x.backward()
            if c.adv_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([x], max_norm=c.adv_grad_clip)
            opt_x.step()

            with torch.no_grad():
                x.clamp_(-1, 1)

            if save_intermediate and (step_idx == 1 or step_idx % vis_interval == 0 or step_idx == test_steps):
                save_intermediate_visuals(result_folder, run_idx, step_idx, x, cover_all, noise_dec_out)

        with torch.no_grad():
            watermarked = x.detach().clamp(-1, 1)
            psnr_x = float(psnr_minus1_to1(cover_all, watermarked))
            acc_values = evaluate_attacks(model, test_noise, watermarked, cover_all, bits, carriers)
            save_final_visuals(result_folder, run_idx, watermarked, cover_all)

        log_and_print_result(logger, run_idx, psnr_x, acc_values)


if __name__ == "__main__":
    main()
