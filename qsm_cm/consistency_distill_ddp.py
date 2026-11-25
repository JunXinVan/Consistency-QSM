"""
Consistency distillation training on a single machine with multiple GPUs.
Uses torch.distributed (NCCL) - MPI is not required.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from qsm_cm.dataset_clean import QSMCleanDataset
from qsm_cm.dataset_nifti import QSMNiftiDataset
from qsm_cm.student_model_3d import StudentUNetConfig, build_student_unet
from qsm_cm.teacher_diffusionqsm import DiffusionQSMTeacher


def setup_distributed():
    """Initialize the distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        print(f"[DIST] Rank {rank}/{world_size}, GPU {local_rank}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=tuple(range(1, x.ndim)))


def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while x.ndim < target_ndim:
        x = x[..., None]
    return x


def estimate_scale_from_loader(
    loader: torch.utils.data.DataLoader,
    num_batches: int,
    *,
    ignore_zero: bool,
    eps: float,
    target_std: float,
    target_p99: float,
    mode: str,
) -> tuple[float, float]:
    collected = []
    for idx, batch in enumerate(loader, start=1):
        chi = batch["chi"].detach().cpu().numpy().astype(np.float64).reshape(-1)
        if ignore_zero:
            chi = chi[np.abs(chi) > float(eps)]
        if chi.size:
            collected.append(chi)
        if idx >= num_batches:
            break

    if not collected:
        return 1.0, float(target_std)

    flat = np.concatenate(collected, axis=0)
    std = float(np.std(flat))
    p99 = float(np.percentile(np.abs(flat), 99.0))

    if mode == "p99" and p99 > 0:
        chi_scale = float(target_p99) / p99
    elif std > 0:
        chi_scale = float(target_std) / std
    else:
        chi_scale = 1.0

    sigma_data = std * chi_scale if std > 0 else float(target_std)
    return chi_scale, sigma_data


def get_weightings(weight_schedule: str, snrs: torch.Tensor, sigma_data: float) -> torch.Tensor:
    if weight_schedule == "snr":
        return snrs
    if weight_schedule == "snr+1":
        return snrs + 1.0
    if weight_schedule == "karras":
        return snrs + 1.0 / float(sigma_data) ** 2
    if weight_schedule == "truncated-snr":
        return torch.clamp(snrs, min=1.0)
    if weight_schedule == "uniform":
        return torch.ones_like(snrs)
    raise ValueError(f"Unknown weight_schedule: {weight_schedule}")


def parse_tuple3_i(s: str) -> Tuple[int, int, int]:
    parts = [int(x) for x in s.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 ints, got {s}")
    return parts[0], parts[1], parts[2]


def sample_karras_sigmas(
    batch: int,
    num_scales: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_scales < 2:
        raise ValueError("num_scales must be >= 2")
    indices = torch.randint(0, num_scales - 1, (batch,), device=device)
    inv_rho = 1.0 / float(rho)
    sigma_max_r = float(sigma_max) ** inv_rho
    sigma_min_r = float(sigma_min) ** inv_rho
    t = sigma_max_r + indices / float(num_scales - 1) * (sigma_min_r - sigma_max_r)
    t2 = sigma_max_r + (indices + 1) / float(num_scales - 1) * (sigma_min_r - sigma_max_r)
    return t**rho, t2**rho


def get_sigmas_karras(
    n: int, sigma_min: float, sigma_max: float, rho: float, device: torch.device
) -> torch.Tensor:
    if n < 2:
        raise ValueError("num_scales must be >= 2")
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = float(sigma_min) ** (1.0 / float(rho))
    max_inv_rho = float(sigma_max) ** (1.0 / float(rho))
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** float(rho)
    return sigmas.to(dtype=torch.float32)


def build_discrete_sigma_schedule(
    teacher: DiffusionQSMTeacher,
    *,
    num_scales: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas_raw = get_sigmas_karras(num_scales, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, device=device)
    with torch.no_grad():
        t_idx = teacher.t_from_sigma(sigmas_raw)
        for i in range(1, t_idx.numel()):
            if t_idx[i] >= t_idx[i - 1]:
                t_idx[i] = torch.clamp(t_idx[i - 1] - 1, min=0)
        sigmas = teacher.sigma_table[t_idx]
    return sigmas, t_idx


def get_scalings_for_boundary_condition(
    sigma: torch.Tensor, sigma_data: float, sigma_min: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_data_t = float(sigma_data)
    sigma_min_t = float(sigma_min)
    c_skip = sigma_data_t**2 / ((sigma - sigma_min_t) ** 2 + sigma_data_t**2)
    c_out = (sigma - sigma_min_t) * sigma_data_t / torch.sqrt(sigma**2 + sigma_data_t**2)
    c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data_t**2)
    return c_skip, c_out, c_in


def denoise_karras_uncond(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    *,
    sigma_data: float,
    sigma_min: float,
) -> torch.Tensor:
    """Karras preconditioned denoiser with numerical stability fixes.
    
    NOTE: Must be called outside of autocast context (or with autocast disabled)
    to prevent FP16 overflow in 3D UNet skip connections.
    FP16 max value is 65504, but Conv3d can overflow when intermediate values reach ~20000.
    """
    sigma_min_t = float(sigma_min)
    
    # Clamp sigma to safe range
    sigma = sigma.clamp(min=sigma_min_t + 1e-6, max=1e3)
    
    boundary = sigma <= (sigma_min_t + 1e-6)
    if boundary.ndim == 0:
        if bool(boundary.item()):
            return x_t
    elif bool(boundary.all().item()):
        return x_t

    c_skip, c_out, c_in = get_scalings_for_boundary_condition(sigma, sigma_data, sigma_min)
    c_skip = append_dims(c_skip, x_t.ndim)
    c_out = append_dims(c_out, x_t.ndim)
    c_in = append_dims(c_in, x_t.ndim)

    # Use float32 for time embedding and input preprocessing
    sigma_float32 = sigma.float()
    t = 250.0 * torch.log(sigma_float32 + 1e-10)
    
    base = c_skip * x_t
    
    if boundary.ndim > 0 and bool(boundary.any().item()):
        idx = (~boundary).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return x_t
        model_input = (x_t * c_in)[idx]
        model_out = model(model_input, t[idx])
        # FIXED: Model now outputs 2 channels (learn_sigma), take only first channel
        if model_out.shape[1] == 2:
            model_out = model_out[:, :1]
        out = base.clone()
        out[idx] = base[idx] + c_out[idx] * model_out
        return out

    model_input = x_t * c_in
    model_out = model(model_input, t)
    # FIXED: Model now outputs 2 channels (learn_sigma), take only first channel
    if model_out.shape[1] == 2:
        model_out = model_out[:, :1]
    return base + c_out * model_out


@torch.no_grad()
def heun_solver(
    teacher: DiffusionQSMTeacher,
    x: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    teacher_mode: str,
    chi_scale: float = 1.0,
) -> torch.Tensor:
    """Heun ODE solver with numerical stability checks"""
    dims = x.ndim

    # The training loop runs in the chi-scaled
    # space (x0 = chi_ppm * chi_scale, sigma taken numerically from the
    # teacher's ppm-native sigma table). The ppm-native teacher must be
    # queried in its own units: divide BOTH the state and sigma by
    # chi_scale, then multiply the denoised output back. Previously the
    # scaled state/sigma were fed directly, so the teacher saw ~29x
    # magnitudes and produced severely over-shrunk targets.
    cs = float(chi_scale)

    def query_teacher(x_q: torch.Tensor, s_q: torch.Tensor) -> torch.Tensor:
        return cs * teacher.denoise_sigma(x_q / cs, s_q / cs, mode=teacher_mode)

    # Add epsilon to prevent division by zero
    sigma_safe = sigma.clamp(min=1e-6)

    denoiser = query_teacher(x, sigma)
    
    # Check denoiser output
    if not torch.isfinite(denoiser).all():
        # If the teacher output is invalid, return the input as a fallback
        return x
    
    d = (x - denoiser) / append_dims(sigma_safe, dims)
    x_euler = x + d * append_dims(next_sigma - sigma, dims)
    
    # Limit the Euler step result to prevent explosion
    # Widened 100 -> 1000. With the unit-fixed teacher the
    # legitimate trajectory magnitude is ~sigma (~157 at the top bin), so the
    # old +-100 bound clipped 0.5%-47% of voxels on the top 10 of 40 bins and
    # corrupted those Heun targets. +-1000 still guards true explosions while
    # never clipping legitimate values (max legit |x_euler| ~ 800).
    x_euler = torch.clamp(x_euler, min=-1000.0, max=1000.0)

    if next_sigma.ndim == 0:
        if float(next_sigma.item()) <= 0.0:
            return x_euler
        denoiser_2 = query_teacher(x_euler, next_sigma)
        
        if not torch.isfinite(denoiser_2).all():
            return x_euler
            
        next_d = (x_euler - denoiser_2) / append_dims(next_sigma.clamp(min=1e-6), dims)
        return x + (d + next_d) * append_dims((next_sigma - sigma) / 2.0, dims)

    zero_mask = next_sigma <= 0.0
    if bool(zero_mask.any().item()):
        out = x_euler.clone()
        idx = (~zero_mask).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return out
        denoiser_2 = query_teacher(x_euler[idx], next_sigma[idx])
        
        if not torch.isfinite(denoiser_2).all():
            return out
            
        next_d = (x_euler[idx] - denoiser_2) / append_dims(next_sigma[idx].clamp(min=1e-6), dims)
        out[idx] = x[idx] + (d[idx] + next_d) * append_dims((next_sigma[idx] - sigma[idx]) / 2.0, dims)
        return out

    denoiser_2 = query_teacher(x_euler, next_sigma)
    
    if not torch.isfinite(denoiser_2).all():
        return x_euler
        
    next_d = (x_euler - denoiser_2) / append_dims(next_sigma.clamp(min=1e-6), dims)
    return x + (d + next_d) * append_dims((next_sigma - sigma) / 2.0, dims)


def update_ema_safe(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float, step: int = 0) -> bool:
    """Safe EMA update that skips parameters containing NaN/Inf."""
    with torch.no_grad():
        valid_update = True
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            if not torch.isfinite(param).all():
                # Parameter contains NaN/Inf; skip this parameter's update
                valid_update = False
                continue
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
        return valid_update

# Kept for backward compatibility
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    update_ema_safe(ema_model, model, decay)


def load_student_weights(model: torch.nn.Module, ckpt: Path) -> None:
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=True)
        return
    model.load_state_dict(state, strict=True)


def init_student_from_teacher(model: torch.nn.Module, teacher_ckpt: Path) -> None:
    try:
        raw = torch.load(teacher_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(teacher_ckpt, map_location="cpu")

    if isinstance(raw, dict):
        teacher_state = raw.get("model_state_dict") or raw.get("state_dict") or raw.get("model_state") or raw
    else:
        teacher_state = raw

    if not isinstance(teacher_state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: expected state_dict dict, got {type(teacher_state)}")
    if any(k.startswith("module.") for k in teacher_state.keys()):
        teacher_state = {k[len("module.") :]: v for k, v in teacher_state.items()}

    model_state = model.state_dict()
    updated = 0
    for k, v in teacher_state.items():
        if k not in model_state:
            continue
        dst = model_state[k]
        if tuple(v.shape) == tuple(dst.shape):
            model_state[k] = v
            updated += 1
            continue
        if v.ndim == dst.ndim and v.ndim >= 1 and tuple(v.shape[1:]) == tuple(dst.shape[1:]) and dst.shape[0] == 1:
            model_state[k] = v[:1].clone()
            updated += 1
            continue

    model.load_state_dict(model_state, strict=True)
    if is_main_process():
        print(f"[Init] Copied {updated} tensors from teacher into student (shape-compatible).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consistency distillation with Multi-GPU support (DDP)"
    )
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--teacher-ckpt", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=Path("runs/qsm_cd_ddp"))
    p.add_argument("--init-ckpt", type=Path, default=None)
    p.add_argument("--skip-teacher-init", action="store_true")

    p.add_argument("--patch-size", type=str, default="64,64,64")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--repeats-per-volume", type=int, default=64)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--chi-scale", type=float, default=1.0)
    p.add_argument("--nifti-pattern", type=str, default="cosmos*.nii*")
    p.add_argument("--auto-scale", action="store_true")
    p.add_argument("--auto-scale-mode", type=str, default="std", choices=["std", "p99"])
    p.add_argument("--auto-scale-batches", type=int, default=20)
    p.add_argument("--auto-scale-eps", type=float, default=1e-6)
    p.add_argument("--target-std", type=float, default=0.5)
    p.add_argument("--target-p99", type=float, default=1.0)
    p.add_argument("--ignore-zero", action="store_true")

    p.add_argument("--num-scales", type=int, default=40)
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma-data", type=float, default=0.5)
    p.add_argument("--teacher-mode", type=str, default="nearest", choices=["nearest", "interp"])
    p.add_argument("--weight-schedule", type=str, default="karras",
                   choices=["snr", "snr+1", "karras", "truncated-snr", "uniform"])
    p.add_argument("--loss-norm", type=str, default="l2", choices=["l1", "l2", "pseudo-huber"])
    p.add_argument("--pseudo-huber-c", type=float, default=0.28)
    p.add_argument("--keep-latest-ckpt-only", action="store_true",
                   help="delete the previous epoch checkpoint after each save")
    p.add_argument("--min-free-gb", type=float, default=0.0,
                   help="stop training cleanly if free disk drops below this")

    p.add_argument("--amp", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Initialize distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait for the main process to create the directory
    if dist.is_initialized():
        dist.barrier()

    teacher = DiffusionQSMTeacher.load(args.teacher_ckpt, device=device)
    teacher_sigma_min = float(teacher.sigma_table[0].item())
    teacher_sigma_max = float(teacher.sigma_table[-1].item())
    sigma_min = float(args.sigma_min) if args.sigma_min is not None else teacher_sigma_min
    sigma_max = float(args.sigma_max) if args.sigma_max is not None else teacher_sigma_max
    sigma_min = max(sigma_min, teacher_sigma_min)
    sigma_max = min(sigma_max, teacher_sigma_max)
    
    if is_main_process():
        print(f"[Teacher] sigma range: [{teacher_sigma_min:.6g}, {teacher_sigma_max:.6g}]")
        print(f"[Train] sigma range: [{sigma_min:.6g}, {sigma_max:.6g}]")

    if args.teacher_mode == "interp":
        sigmas_schedule = get_sigmas_karras(
            args.num_scales, sigma_min=sigma_min, sigma_max=sigma_max, rho=args.rho, device=device
        )
    else:
        sigmas_schedule, _ = build_discrete_sigma_schedule(
            teacher,
            num_scales=args.num_scales,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=args.rho,
            device=device,
        )

    patch_size = parse_tuple3_i(args.patch_size)
    case_files = sorted((args.data_dir / "cases").glob("*.npz"))
    if not case_files:
        case_files = sorted(args.data_dir.glob("*.npz"))

    if case_files:
        dataset = QSMCleanDataset(
            case_files,
            patch_size=patch_size,
            repeats_per_volume=args.repeats_per_volume,
            require_field=False,
        )
    else:
        nifti_files = sorted(args.data_dir.rglob(args.nifti_pattern))
        if not nifti_files:
            raise RuntimeError(f"No data found")
        dataset = QSMNiftiDataset(
            nifti_files,
            patch_size=patch_size,
            repeats_per_volume=args.repeats_per_volume,
            try_load_mask=False,
            try_load_field=False,
        )
    
    # Use DistributedSampler
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    if args.auto_scale and is_main_process():
        chi_scale, sigma_data_auto = estimate_scale_from_loader(
            loader,
            num_batches=int(args.auto_scale_batches),
            ignore_zero=bool(args.ignore_zero),
            eps=float(args.auto_scale_eps),
            target_std=float(args.target_std),
            target_p99=float(args.target_p99),
            mode=str(args.auto_scale_mode),
        )
        args.chi_scale = chi_scale
        args.sigma_data = sigma_data_auto
        print(f"[AutoScale] chi_scale={args.chi_scale:.6g} sigma_data={args.sigma_data:.6g}")
    
    # Broadcast auto-scale parameters to all processes
    if dist.is_initialized():
        params = torch.tensor([args.chi_scale, args.sigma_data], device=device)
        dist.broadcast(params, 0)
        args.chi_scale = float(params[0].item())
        args.sigma_data = float(params[1].item())

    # Create model
    # FIXED: Teacher uses learn_sigma=True -> out_channels=2
    model = build_student_unet(StudentUNetConfig(), in_channels=1, out_channels=2).to(device)
    
    # Track whether training state needs to be resumed from a checkpoint
    start_epoch = 1
    
    if not args.skip_teacher_init and args.init_ckpt is None:
        init_student_from_teacher(model, args.teacher_ckpt)
    if args.init_ckpt is not None:
        # Load model weights
        load_student_weights(model, args.init_ckpt)
        
        # If it is a full checkpoint, try to restore training state
        if is_main_process():
            print(f"[Resume] Loading checkpoint from {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location=device)
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1  # Start from the next epoch
            if is_main_process():
                print(f"[Resume] Will start from epoch {start_epoch}")

    # Wrap with DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ema_model = copy.deepcopy(model.module if dist.is_initialized() else model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # If resuming from a checkpoint, load optimizer state
    if args.init_ckpt is not None:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            # Resume the optimizer statistics, but keep the continuation run's
            # explicitly requested hyperparameters authoritative.
            for group in optimizer.param_groups:
                group["lr"] = float(args.lr)
                group["weight_decay"] = float(args.weight_decay)
            if is_main_process():
                print(
                    f"[Resume] Optimizer state loaded; "
                    f"override lr={float(args.lr):.6g}, weight_decay={float(args.weight_decay):.6g}"
                )
        if 'ema_state' in ckpt:
            ema_model.load_state_dict(ckpt['ema_state'])
            if is_main_process():
                print(f"[Resume] EMA state loaded")
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(start_epoch, args.epochs + 1):
        if dist.is_initialized():
            loader.sampler.set_epoch(epoch)
        
        model.train()
        ema_model.eval()
        running = 0.0
        steps = 0

        for step, batch in enumerate(loader, start=1):
            x0 = batch["chi"].to(device) * float(args.chi_scale)
            
            # Check input data
            if not torch.isfinite(x0).all():
                if is_main_process():
                    nan_count = torch.isnan(x0).sum().item()
                    inf_count = torch.isinf(x0).sum().item()
                    print(f"[WARNING] Epoch {epoch} step {step}: Input has {nan_count} NaN, {inf_count} Inf values, skipping")
                continue
            
            noise = torch.randn_like(x0)

            scale_idx = torch.randint(0, args.num_scales - 1, (x0.shape[0],), device=device)
            sigma_t = sigmas_schedule[scale_idx]
            sigma_t2 = sigmas_schedule[scale_idx + 1]
            x_t = x0 + noise * append_dims(sigma_t, x0.ndim)

            # Run model forward pass (in FP32 to avoid overflow in 3D UNet)
            model_to_call = model.module if dist.is_initialized() else model
            distiller = denoise_karras_uncond(
                model_to_call,
                x_t=x_t,
                sigma=sigma_t,
                sigma_data=args.sigma_data,
                sigma_min=sigma_min,
            )
            
            # Check student output
            if not torch.isfinite(distiller).all():
                if is_main_process():
                    print(f"[WARNING] Epoch {epoch} step {step}: Student distiller output has NaN/Inf, skipping")
                continue

            with torch.no_grad():
                x_t2 = heun_solver(teacher, x_t, sigma_t, sigma_t2, args.teacher_mode,
                                   chi_scale=float(args.chi_scale))
                
                # Check Heun solver output
                if not torch.isfinite(x_t2).all():
                    if is_main_process():
                        print(f"[WARNING] Epoch {epoch} step {step}: Heun solver output has NaN/Inf, skipping")
                    continue
                
                distiller_target = denoise_karras_uncond(
                    ema_model,
                    x_t=x_t2,
                    sigma=sigma_t2,
                    sigma_data=args.sigma_data,
                    sigma_min=sigma_min,
                )
                
                # Check target output
                if not torch.isfinite(distiller_target).all():
                    if is_main_process():
                        print(f"[WARNING] Epoch {epoch} step {step}: Target distiller output has NaN/Inf, skipping")
                    continue

            # Loss computation OUTSIDE no_grad (needs gradients)
            if args.loss_norm == "l1":
                diffs = torch.abs(distiller - distiller_target)
            elif args.loss_norm == "pseudo-huber":
                c_ph = float(args.pseudo_huber_c)
                d_sq = (distiller - distiller_target) ** 2
                diffs = c_ph * (torch.sqrt(1.0 + d_sq / (c_ph * c_ph)) - 1.0)
            else:
                diffs = (distiller - distiller_target) ** 2

            snrs = sigma_t.pow(-2)
            weights = get_weightings(args.weight_schedule, snrs, sigma_data=args.sigma_data)
            loss = (mean_flat(diffs) * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            # Check whether loss is NaN; if so, skip this step
            if not torch.isfinite(loss):
                if is_main_process():
                    print(f"[WARNING] Epoch {epoch} step {step}: loss is NaN/Inf, skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # FP32 backward (no scaler needed)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check whether gradients are finite
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                if is_main_process():
                    print(f"[WARNING] Epoch {epoch} step {step}: NaN gradient detected, skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            optimizer.step()
            
            # Update EMA (main process only) - use the safe version
            if is_main_process():
                ema_valid = update_ema_safe(
                    ema_model, 
                    model.module if dist.is_initialized() else model, 
                    decay=args.ema_decay,
                    step=step
                )
                if not ema_valid:
                    print(f"[WARNING] Epoch {epoch} step {step}: EMA update had invalid params, some skipped")

            running += float(loss.item())
            steps += 1
            
            if is_main_process() and args.log_every > 0 and step % args.log_every == 0:
                avg = running / max(steps, 1)
                print(f"[Epoch {epoch} step {step}] loss={loss.item():.6f} avg={avg:.6f}")

        avg_loss = running / max(steps, 1)
        if is_main_process():
            print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f}")
            if args.min_free_gb > 0:
                free_gb = shutil.disk_usage(args.save_dir).free / 1e9
                if free_gb < args.min_free_gb:
                    print(f"[DISK] free={free_gb:.2f}G < {args.min_free_gb}G, "
                          f"stopping cleanly BEFORE epoch {epoch} save")
                    break
            ckpt_path = args.save_dir / f"cd_ddp_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": (model.module if dist.is_initialized() else model).state_dict(),
                    "ema_state": ema_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(args),
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                },
                ckpt_path,
            )
            if args.keep_latest_ckpt_only and epoch > 1:
                prev = args.save_dir / f"cd_ddp_epoch_{epoch - 1:03d}.pth"
                if prev.exists():
                    prev.unlink()
                    print(f"[DISK] deleted previous ckpt {prev.name}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
