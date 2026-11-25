#!/usr/bin/env python3
"""
Inference on Diffusion-QSM-formatted test cases (`data_DiffusionQSM.mat`).

This script aligns the I/O contract with Diffusion-QSM:
  - input fields: phi_use / mask / D2 / pad_size / optional calci_use
  - patch size / stride style compatible with Diffusion-QSM recon.py
  - output cropped back by pad_size and optionally saved as NIfTI

Important:
  - The model itself remains our Stage 4 v3 patch-level recurrent CM + HardDC solver.
  - We align benchmark data handling to Diffusion-QSM, not the internal diffusion sampler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import nibabel as nib
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qsm_bank_utils import (
    ConditioningSpec,
    StepAdaptiveStudentModel,
    adjoint_field_to_chi,
    build_conditioning_spec,
    build_dc_rollout_policy,
    conditioning_channels_for_mode,
    create_dipole_kernel,
    expected_base_model_in_channels,
    geometric_sigma_schedule,
    load_student_components,
    maybe_expand_input_conv_weight,
    parse_triplet,
    run_refinement_rollout,
    run_residual_delta_refinement_rollout,
)
from qsm_morph import build_morph_qsm0


def parse_subject_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tokens: list[str] = []
    seen: set[str] = set()
    for raw_part in value.split(","):
        token = raw_part.strip().strip("/")
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens or None


def overlapping_grid_indices(shape: tuple[int, int, int], output_size: tuple[int, int, int], stride: tuple[int, int, int]):
    def axis_starts(length: int, patch: int, step: int) -> list[int]:
        if patch <= 0 or step <= 0:
            raise ValueError(f"Patch and stride must be positive, got patch={patch}, stride={step}")
        if length <= patch:
            return [0]
        starts = list(range(0, length - patch + 1, step))
        tail_start = length - patch
        if starts[-1] != tail_start:
            starts.append(tail_start)
        return starts

    d, h, w = shape
    return (
        axis_starts(d, output_size[0], stride[0]),
        axis_starts(h, output_size[1], stride[1]),
        axis_starts(w, output_size[2], stride[2]),
    )


def patch_process(patch1: np.ndarray, patch2: np.ndarray, overlap: int, direction: int) -> np.ndarray:
    result = patch1.copy()
    if patch1.shape != patch2.shape:
        raise ValueError(f"Overlapping patches must have the same shape, got {patch1.shape} vs {patch2.shape}")
    if overlap <= 1:
        return result

    weight = np.ones(overlap, dtype=np.float32)
    for idx in range(overlap):
        weight[idx] = 1.0 - idx / (overlap - 1)

    if direction == 0:
        block1 = patch1[0:overlap, :, :]
        block2 = patch2[int(patch1.shape[0] - overlap) : patch1.shape[0], :, :]
        for i in range(overlap):
            result[i, :, :] = weight[i] * block2[i, :, :] + (1.0 - weight[i]) * block1[i, :, :]
    elif direction == 1:
        block1 = patch1[:, 0:overlap, :]
        block2 = patch2[:, int(patch1.shape[1] - overlap) : patch1.shape[1], :]
        for j in range(overlap):
            result[:, j, :] = weight[j] * block2[:, j, :] + (1.0 - weight[j]) * block1[:, j, :]
    elif direction == 2:
        block1 = patch1[:, :, 0:overlap]
        block2 = patch2[:, :, int(patch1.shape[2] - overlap) : patch1.shape[2]]
        for k in range(overlap):
            result[:, :, k] = weight[k] * block2[:, :, k] + (1.0 - weight[k]) * block1[:, :, k]
    else:
        raise ValueError(f"direction must be 0, 1, or 2, got {direction}")
    return result


def patch_overlap_from_starts(patch_size: int, current_start: int, previous_start: int) -> int:
    """Return the true overlap when tail starts are not equal to the nominal stride."""
    offset = int(current_start) - int(previous_start)
    if offset <= 0 or offset >= int(patch_size):
        raise ValueError(
            f"Invalid neighboring patch starts: previous={previous_start}, current={current_start}, patch={patch_size}"
        )
    return int(patch_size) - offset


def detect_stage4_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(
        key.startswith("base_model.")
        or key.startswith("adapter.")
        or key.startswith("conditioning_stem.")
        for key in state_dict.keys()
    )


def resolve_saved_arg(saved_args: dict, key: str, fallback, provided_dests: set[str] | None = None):
    if provided_dests and key in provided_dests:
        return fallback
    value = saved_args.get(key, fallback)
    return fallback if value is None else value


def extract_provided_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    provided: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            provided.add(dest)
    return provided


def load_stage4_wrapper_state(
    model: StepAdaptiveStudentModel,
    state_dict: dict[str, torch.Tensor],
    allow_missing_conditioning_stem: bool,
) -> None:
    state_dict = maybe_expand_input_conv_weight(
        state_dict,
        weight_key="base_model.input_blocks.0.0.weight",
        target_in_channels=expected_base_model_in_channels(
            model.conditioning_mode,
            conditioning_channels=model.conditioning_channels,
        ),
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if allow_missing_conditioning_stem:
        missing = [key for key in missing if not key.startswith("conditioning_stem.")]
    if missing or unexpected:
        raise RuntimeError(
            "Incompatible Stage 4 checkpoint state. "
            f"missing={missing}, unexpected={unexpected}"
        )


def myfftnc(x: torch.Tensor, dims=(-3, -2, -1)) -> torch.Tensor:
    return torch.fft.ifftshift(
        torch.fft.fftn(torch.fft.fftshift(x, dim=dims), dim=dims, norm="ortho"),
        dim=dims,
    )


def myifftnc(x: torch.Tensor, dims=(-3, -2, -1)) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"),
        dim=dims,
    )


def full_volume_data_consistency_cg(
    chi_init: torch.Tensor,
    field_gt: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    calci: torch.Tensor | None = None,
    lam: float = 0.1,
    num_cg_steps: int = 8,
    tol: float = 1e-8,
) -> torch.Tensor:
    if chi_init.ndim != 5 or field_gt.ndim != 5 or mask.ndim != 5:
        raise ValueError("full_volume_data_consistency_cg expects 5D tensors [B,C,D,H,W].")

    if calci is None:
        calci = torch.ones_like(field_gt)
    if calci.ndim != 5:
        raise ValueError("calci must be a 5D tensor matching field_gt when provided.")

    kernel = dipole_kernel
    while kernel.ndim < 5:
        kernel = kernel.unsqueeze(0)

    lam_value = torch.tensor(float(lam), device=chi_init.device, dtype=torch.float32)
    lam_complex = torch.complex(lam_value, torch.zeros_like(lam_value)).view(1, 1, 1, 1, 1)

    def A(x: torch.Tensor) -> torch.Tensor:
        return calci * myifftnc(kernel * myfftnc(x)).real

    def AT(y: torch.Tensor) -> torch.Tensor:
        return myifftnc(kernel * myfftnc(calci * y))

    def AtA(x: torch.Tensor) -> torch.Tensor:
        return AT(A(x),) + lam_complex * x

    z_k = torch.complex(chi_init, torch.zeros_like(chi_init))
    tissue_phi = torch.complex(field_gt, torch.zeros_like(field_gt))
    x = torch.zeros_like(z_k)

    rhs = AT(tissue_phi) + lam_complex * z_k
    r = rhs.clone()
    p = rhs.clone()
    r_tr = torch.sum((r.conj() * r).real).to(torch.float32)

    for _ in range(max(int(num_cg_steps), 0)):
        if float(torch.real(r_tr).item()) <= tol:
            break
        ap = AtA(p)
        denom = torch.sum((p.conj() * ap).real).to(torch.float32).clamp_min(1e-12)
        alpha = torch.complex(r_tr / denom, torch.zeros((), device=chi_init.device, dtype=torch.float32))
        x = x + alpha * p
        r = r - alpha * ap
        r_tr_new = torch.sum((r.conj() * r).real).to(torch.float32)
        beta = torch.complex((r_tr_new / r_tr.clamp_min(1e-12)), torch.zeros((), device=chi_init.device, dtype=torch.float32))
        p = r + beta * p
        r_tr = r_tr_new

    return x.real


def build_cone_weight_map(d2: np.ndarray, threshold: float, power: float = 1.0) -> np.ndarray:
    threshold = max(float(threshold), 1e-6)
    power = max(float(power), 1e-6)
    weight = np.clip((threshold - np.abs(d2).astype(np.float32)) / threshold, 0.0, 1.0)
    if abs(power - 1.0) > 1e-6:
        weight = weight**power
    return weight.astype(np.float32)


def cone_frequency_blend(
    prior_volume: np.ndarray,
    dc_volume: np.ndarray,
    d2: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    power: float,
    device: torch.device,
) -> np.ndarray:
    prior_t = torch.from_numpy(prior_volume).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    dc_t = torch.from_numpy(dc_volume).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask_t = torch.from_numpy(mask).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cone_weight = build_cone_weight_map(d2, threshold=threshold, power=power)
    weight_t = torch.from_numpy(cone_weight).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    prior_f = myfftnc(prior_t)
    dc_f = myfftnc(dc_t)
    blended_f = weight_t * prior_f + (1.0 - weight_t) * dc_f
    blended = myifftnc(blended_f).real * mask_t
    return blended.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


class Stage4DiffusionQSMInference:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.setup_model()

    def _select_state_dict(self, ckpt: dict[str, torch.Tensor]) -> tuple[str, dict]:
        preferred = "ema_state" if self.args.infer_use_ema else "model_state"
        fallback = "model_state" if preferred == "ema_state" else "ema_state"
        for state_name in (preferred, fallback):
            state_dict = ckpt.get(state_name)
            if state_dict is not None:
                return state_name, state_dict
        return "raw", ckpt

    @staticmethod
    def normalize_registry_value(raw_value):
        if isinstance(raw_value, str):
            parts = [part.strip() for part in raw_value.split(",") if part.strip()]
            if not parts:
                return raw_value
            if len(parts) == 1:
                token = parts[0]
                for caster in (int, float):
                    try:
                        return caster(token)
                    except ValueError:
                        pass
                return token
            normalized = []
            for token in parts:
                parsed = token
                for caster in (int, float):
                    try:
                        parsed = caster(token)
                        break
                    except ValueError:
                        continue
                normalized.append(parsed)
            return normalized
        return raw_value

    def setup_model(self) -> None:
        build_student_unet, StudentUNetConfig, get_scalings_for_boundary_condition = load_student_components(
            project_root=self.args.project_root,
            diffusion_root=self.args.diffusion_root,
            script_path=__file__,
        )
        self.get_scalings_for_boundary_condition = get_scalings_for_boundary_condition

        ckpt = torch.load(self.args.ckpt, map_location="cpu", weights_only=False)
        saved_args = dict(ckpt.get("args", {}))
        state_name, state_dict = self._select_state_dict(ckpt)
        is_stage4 = detect_stage4_checkpoint(state_dict)

        step_adapter_mode = resolve_saved_arg(saved_args, "step_adapter_mode", self.args.step_adapter_mode)
        step_adapter_hidden_dim = int(resolve_saved_arg(saved_args, "step_adapter_hidden_dim", self.args.step_adapter_hidden_dim))
        conditioning_mode = (
            resolve_saved_arg(saved_args, "conditioning_mode", self.args.conditioning_mode)
            if is_stage4
            else self.args.conditioning_mode
        )
        conditioning_hidden_dim = int(
            resolve_saved_arg(saved_args, "conditioning_hidden_dim", self.args.conditioning_hidden_dim)
            if is_stage4
            else self.args.conditioning_hidden_dim
        )
        conditioning_norm_mode = (
            resolve_saved_arg(saved_args, "conditioning_norm_mode", self.args.conditioning_norm_mode)
            if is_stage4
            else self.args.conditioning_norm_mode
        )
        conditioning_cone_threshold = float(
            resolve_saved_arg(saved_args, "conditioning_cone_threshold", self.args.conditioning_cone_threshold)
            if is_stage4
            else self.args.conditioning_cone_threshold
        )
        init_error_proxy_sigma = float(
            resolve_saved_arg(saved_args, "init_error_proxy_sigma", self.args.init_error_proxy_sigma)
            if is_stage4
            else self.args.init_error_proxy_sigma
        )
        self.morph_prior_mode = str(resolve_saved_arg(saved_args, "morph_prior_mode", self.args.morph_prior_mode))
        self.morph_residual_scale = float(resolve_saved_arg(saved_args, "morph_residual_scale", self.args.morph_residual_scale))
        self.morph_tkd_low = float(resolve_saved_arg(saved_args, "morph_tkd_low", self.args.morph_tkd_low))
        self.morph_tkd_high = float(resolve_saved_arg(saved_args, "morph_tkd_high", self.args.morph_tkd_high))
        self.morph_cone_threshold = float(resolve_saved_arg(saved_args, "morph_cone_threshold", self.args.morph_cone_threshold))
        self.morph_cone_power = float(resolve_saved_arg(saved_args, "morph_cone_power", self.args.morph_cone_power))
        self.sigma_data = float(ckpt.get("sigma_data", ckpt.get("config", {}).get("sigma_data", 0.5)))
        self.sigma_min = float(ckpt.get("sigma_min", 0.01))
        self.sigma_max = float(
            ckpt.get(
                "sigma_max",
                max(self.sigma_data * self.args.student_sigma_max_factor, self.sigma_min * 1.01),
            )
        )
        if getattr(self.args, "sigma_max_override", None) is not None:
            self.sigma_max = float(self.args.sigma_max_override)

        conditioning_channels = conditioning_channels_for_mode(conditioning_mode)
        base_model_in_channels = expected_base_model_in_channels(
            conditioning_mode,
            conditioning_channels=conditioning_channels,
        )
        base_model = build_student_unet(StudentUNetConfig(), in_channels=base_model_in_channels, out_channels=2)
        if is_stage4:
            model = StepAdaptiveStudentModel(
                base_model,
                mode=step_adapter_mode,
                hidden_dim=step_adapter_hidden_dim,
                conditioning_mode=conditioning_mode,
                conditioning_channels=conditioning_channels,
                conditioning_hidden_dim=conditioning_hidden_dim,
            )
            has_conditioning_in_ckpt = any(key.startswith("conditioning_stem.") for key in state_dict.keys())
            load_stage4_wrapper_state(
                model,
                state_dict,
                allow_missing_conditioning_stem=bool(model.conditioning_stem is not None and not has_conditioning_in_ckpt),
            )
            self.checkpoint_kind = "stage4_v3"
        else:
            state_dict = maybe_expand_input_conv_weight(
                state_dict,
                weight_key="input_blocks.0.0.weight",
                target_in_channels=base_model_in_channels,
            )
            base_model.load_state_dict(state_dict, strict=True)
            model = StepAdaptiveStudentModel(
                base_model,
                mode=self.args.step_adapter_mode,
                hidden_dim=self.args.step_adapter_hidden_dim,
                conditioning_mode=conditioning_mode,
                conditioning_channels=conditioning_channels,
                conditioning_hidden_dim=conditioning_hidden_dim,
            )
            self.checkpoint_kind = "stage3_or_raw_student"

        self.model = model.to(self.device).eval()
        self.conditioning_mode = str(conditioning_mode)
        self.conditioning_hidden_dim = int(conditioning_hidden_dim)
        self.conditioning_norm_mode = str(conditioning_norm_mode)
        self.conditioning_cone_threshold = float(conditioning_cone_threshold)
        self.init_error_proxy_sigma = float(init_error_proxy_sigma)
        self.conditioning_channels = int(conditioning_channels)
        self.base_model_in_channels = int(base_model_in_channels)

        provided_cli_dests = set(getattr(self.args, "provided_cli_dests", []))
        self.dc_policy = str(resolve_saved_arg(saved_args, "dc_policy", self.args.dc_policy, provided_cli_dests))
        self.lambda_dc = resolve_saved_arg(saved_args, "lambda_dc", self.args.lambda_dc, provided_cli_dests)
        self.cg_steps_infer = resolve_saved_arg(saved_args, "cg_steps_infer", self.args.cg_steps_infer, provided_cli_dests)
        self.final_polish_infer = bool(
            resolve_saved_arg(saved_args, "final_polish_infer", self.args.final_polish_infer, provided_cli_dests)
        )
        self.final_polish_lam = resolve_saved_arg(
            saved_args, "final_polish_lam", self.args.final_polish_lam, provided_cli_dests
        )
        self.final_polish_steps_infer = resolve_saved_arg(
            saved_args, "final_polish_steps_infer", self.args.final_polish_steps_infer, provided_cli_dests
        )
        self.effective_config = {
            "entrypoint": str(Path(__file__).resolve()),
            "checkpoint": str(Path(self.args.ckpt).expanduser().resolve()),
            "checkpoint_state": state_name,
            "checkpoint_kind": self.checkpoint_kind,
            "effective_step_adapter_mode": step_adapter_mode,
            "effective_step_adapter_hidden_dim": int(step_adapter_hidden_dim),
            "effective_conditioning_mode": self.conditioning_mode,
            "effective_conditioning_hidden_dim": self.conditioning_hidden_dim,
            "effective_conditioning_norm_mode": self.conditioning_norm_mode,
            "effective_conditioning_cone_threshold": self.conditioning_cone_threshold,
            "effective_init_error_proxy_sigma": self.init_error_proxy_sigma,
            "effective_morph_prior_mode": self.morph_prior_mode,
            "effective_morph_residual_scale": self.morph_residual_scale,
            "conditioning_channels": self.conditioning_channels,
            "effective_base_model_in_channels": self.base_model_in_channels,
            "sigma_data": float(self.sigma_data),
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
            "effective_dc_policy": self.dc_policy,
            "effective_lambda_dc": self.normalize_registry_value(self.lambda_dc),
            "effective_cg_steps_infer": self.normalize_registry_value(self.cg_steps_infer),
            "effective_final_polish_infer": bool(self.final_polish_infer),
            "effective_final_polish_lam": self.normalize_registry_value(self.final_polish_lam),
            "effective_final_polish_steps_infer": self.normalize_registry_value(self.final_polish_steps_infer),
            "output_masking_policy": "stitched volume is multiplied by mask after patch stitching and again after full-volume DC",
            "cli_args": vars(self.args),
        }

        print("=" * 72)
        print("Stage 4 v3 Diffusion-QSM-format Inference")
        print("=" * 72)
        print(f"Checkpoint: {self.args.ckpt}")
        print(f"Checkpoint state: {state_name}")
        print(f"Checkpoint kind: {self.checkpoint_kind}")
        print(f"Device: {self.device}")
        print(
            f"DC policy={self.dc_policy}, lambda_dc={self.lambda_dc}, cg_steps_infer={self.cg_steps_infer}, "
            f"final_polish_infer={self.final_polish_infer}"
        )
        print(f"Patch calci-aware DC: {self.args.patch_calci_aware_dc}")
        print(
            "Effective config: "
            f"adapter={step_adapter_mode}/{step_adapter_hidden_dim}, "
            f"conditioning={self.conditioning_mode}/{self.conditioning_hidden_dim}, "
            f"base_in={self.base_model_in_channels}, "
            f"sigma=({self.sigma_data:.4f},{self.sigma_min:.4f},{self.sigma_max:.4f})"
        )
        print("Output masking: enabled (stitched and post-DC volumes are multiplied by brain mask)")

    def write_run_config(self, output_dir: Path) -> None:
        payload = dict(self.effective_config)
        payload["data_root"] = str(Path(self.args.data_root).expanduser().resolve()) if self.args.data_root else None
        payload["mat_path"] = str(Path(self.args.mat_path).expanduser().resolve()) if self.args.mat_path else None
        (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def adapt_schedule_arg(raw_value, num_steps: int):
        if isinstance(raw_value, str):
            parts = [part.strip() for part in raw_value.split(",") if part.strip()]
            if len(parts) <= 1:
                return raw_value
            if len(parts) < num_steps:
                raise ValueError(f"Schedule {raw_value!r} does not provide enough values for {num_steps} steps.")
            return ",".join(parts[:num_steps])
        if isinstance(raw_value, (list, tuple)):
            if len(raw_value) <= 1:
                return raw_value
            if len(raw_value) < num_steps:
                raise ValueError(f"Schedule {raw_value!r} does not provide enough values for {num_steps} steps.")
            return list(raw_value[:num_steps])
        return raw_value

    def build_sigma_schedule(self, batch_size: int, num_steps: int) -> list[torch.Tensor]:
        sigma_values = geometric_sigma_schedule(self.sigma_max, self.sigma_min, num_steps + 1)
        return [
            torch.full((batch_size,), float(value), device=self.device, dtype=torch.float32) for value in sigma_values
        ]

    def build_single_step_sigma_schedule(self, batch_size: int, sigma_start: float, sigma_end: float | None = None) -> list[torch.Tensor]:
        sigma_end = float(self.sigma_min if sigma_end is None else sigma_end)
        sigma_start = float(max(min(sigma_start, self.sigma_max), sigma_end))
        return [
            torch.full((batch_size,), sigma_start, device=self.device, dtype=torch.float32),
            torch.full((batch_size,), sigma_end, device=self.device, dtype=torch.float32),
        ]

    def build_dc_policy(self, num_steps: int):
        return build_dc_rollout_policy(
            num_steps=num_steps,
            mode=self.dc_policy,
            lam=self.adapt_schedule_arg(self.lambda_dc, num_steps=num_steps),
            num_cg_steps=self.adapt_schedule_arg(self.cg_steps_infer, num_steps=num_steps),
            final_polish=self.final_polish_infer,
            final_polish_lam=self.final_polish_lam,
            final_polish_steps=self.final_polish_steps_infer,
        )

    def build_conditioning_volume(
        self,
        x_init: torch.Tensor,
        field_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        patch_dipole_kernel: torch.Tensor,
        morph_qsm0: torch.Tensor | None = None,
    ) -> torch.Tensor | ConditioningSpec | None:
        if self.conditioning_mode == "none":
            return None
        return build_conditioning_spec(
            mode=self.conditioning_mode,
            x_init=x_init,
            field_input=field_patch,
            mask=mask_patch,
            dipole_kernel=patch_dipole_kernel,
            cone_threshold=self.conditioning_cone_threshold,
            cone_risk=None,
            morph_qsm0=morph_qsm0,
            init_error_proxy_sigma=self.init_error_proxy_sigma,
            norm_mode=self.conditioning_norm_mode,
        )

    def build_morph_prior(
        self,
        field_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        patch_dipole_kernel: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.morph_prior_mode == "none":
            return None
        return build_morph_qsm0(
            field=field_patch,
            dipole_kernel=patch_dipole_kernel,
            mask=mask_patch,
            tkd_low=self.morph_tkd_low,
            tkd_high=self.morph_tkd_high,
            cone_threshold=self.morph_cone_threshold,
            cone_power=self.morph_cone_power,
            centered_fft=True,
        ).to(device=field_patch.device, dtype=field_patch.dtype)

    def infer_patch(
        self,
        field_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        patch_dipole_kernel: torch.Tensor,
        infer_nfe: int,
        calci_patch: torch.Tensor | None = None,
        x_init_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            x_init = (
                x_init_override
                if x_init_override is not None
                else adjoint_field_to_chi(field_patch, patch_dipole_kernel, mask=mask_patch)
            )
            morph_qsm0 = self.build_morph_prior(field_patch, mask_patch, patch_dipole_kernel)
            if self.morph_prior_mode == "replace_xinit" and morph_qsm0 is not None:
                x_init = morph_qsm0
            if self.args.init_scale != 1.0:
                x_init = x_init * float(self.args.init_scale)
            start_noise_sigma = float(getattr(self.args, "start_noise_sigma", None) or 0.0)
            if start_noise_sigma > 0:
                gen = torch.Generator(device="cpu").manual_seed(20261026)
                eps = torch.randn(tuple(x_init.shape), generator=gen).to(
                    device=x_init.device, dtype=x_init.dtype)
                x_init = (x_init + start_noise_sigma * eps) * mask_patch
            conditioning_volume = self.build_conditioning_volume(
                x_init=x_init,
                field_patch=field_patch,
                mask_patch=mask_patch,
                patch_dipole_kernel=patch_dipole_kernel,
                morph_qsm0=morph_qsm0,
            )
            if self.morph_prior_mode == "residual_delta":
                if morph_qsm0 is None:
                    raise RuntimeError("residual_delta inference requires a morphology prior.")
                if conditioning_volume is None:
                    conditioning_volume = ConditioningSpec(mode="none", morph_qsm0=morph_qsm0)
                rollout = run_residual_delta_refinement_rollout(
                    model=self.model,
                    residual_base=morph_qsm0,
                    residual_scale=self.morph_residual_scale,
                    sigma_schedule=self.build_sigma_schedule(batch_size=1, num_steps=infer_nfe),
                    field_gt=field_patch,
                    mask=mask_patch,
                    dipole_kernel=patch_dipole_kernel,
                    dc_policy=self.build_dc_policy(num_steps=infer_nfe),
                    sigma_data=self.sigma_data,
                    sigma_min=self.sigma_min,
                    get_scalings_for_boundary_condition=self.get_scalings_for_boundary_condition,
                    use_amp=False,
                    calci=calci_patch,
                    conditioning_volume=conditioning_volume,
                    collect_states=False,
                    collect_priors=False,
                )
                return rollout.final_state
            rollout = run_refinement_rollout(
                model=self.model,
                x_start=x_init,
                sigma_schedule=self.build_sigma_schedule(batch_size=1, num_steps=infer_nfe),
                field_gt=field_patch,
                mask=mask_patch,
                dipole_kernel=patch_dipole_kernel,
                dc_policy=self.build_dc_policy(num_steps=infer_nfe),
                sigma_data=self.sigma_data,
                sigma_min=self.sigma_min,
                get_scalings_for_boundary_condition=self.get_scalings_for_boundary_condition,
                use_amp=False,
                calci=calci_patch,
                conditioning_volume=conditioning_volume,
                collect_states=False,
                collect_priors=False,
            )
        return rollout.final_state

    def late_refine_patch(
        self,
        x_start_patch: torch.Tensor,
        field_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        patch_dipole_kernel: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            conditioning_volume = self.build_conditioning_volume(
                x_init=adjoint_field_to_chi(field_patch, patch_dipole_kernel, mask=mask_patch),
                field_patch=field_patch,
                mask_patch=mask_patch,
                patch_dipole_kernel=patch_dipole_kernel,
            )
            rollout = run_refinement_rollout(
                model=self.model,
                x_start=x_start_patch,
                sigma_schedule=self.build_single_step_sigma_schedule(
                    batch_size=x_start_patch.shape[0],
                    sigma_start=self.args.late_refine_sigma,
                    sigma_end=self.sigma_min,
                ),
                field_gt=field_patch,
                mask=mask_patch,
                dipole_kernel=patch_dipole_kernel,
                dc_policy=build_dc_rollout_policy(
                    num_steps=1,
                    mode="fixed",
                    lam=self.args.late_refine_lam,
                    num_cg_steps=self.args.late_refine_cg_steps,
                    final_polish=False,
                ),
                sigma_data=self.sigma_data,
                sigma_min=self.sigma_min,
                get_scalings_for_boundary_condition=self.get_scalings_for_boundary_condition,
                use_amp=False,
                calci=None,
                conditioning_volume=conditioning_volume,
                collect_states=False,
                collect_priors=False,
            )
        return rollout.final_state

    def late_refine_volume(
        self,
        x_start_volume: np.ndarray,
        phi: np.ndarray,
        mask: np.ndarray,
        patch_dipole_kernel: torch.Tensor,
        corners: list[tuple[int, int, int]],
        num_x: int,
        num_y: int,
    ) -> np.ndarray:
        refined = np.zeros_like(x_start_volume, dtype=np.float32)
        output_patches: list[np.ndarray] = []
        d_list = sorted({corner[0] for corner in corners})
        h_list = sorted({corner[1] for corner in corners})
        w_list = sorted({corner[2] for corner in corners})

        for idx, (di, hi, wi) in enumerate(corners):
            x_start_patch = torch.from_numpy(
                x_start_volume[
                    di : di + self.args.patch_size[0],
                    hi : hi + self.args.patch_size[1],
                    wi : wi + self.args.patch_size[2],
                ]
            ).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            field_patch = torch.from_numpy(
                phi[
                    di : di + self.args.patch_size[0],
                    hi : hi + self.args.patch_size[1],
                    wi : wi + self.args.patch_size[2],
                ]
            ).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_patch = torch.from_numpy(
                mask[
                    di : di + self.args.patch_size[0],
                    hi : hi + self.args.patch_size[1],
                    wi : wi + self.args.patch_size[2],
                ]
            ).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            pred_patch = (
                self.late_refine_patch(
                    x_start_patch=x_start_patch,
                    field_patch=field_patch,
                    mask_patch=mask_patch,
                    patch_dipole_kernel=patch_dipole_kernel,
                )
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            if di != 0:
                prev_di = d_list[d_list.index(di) - 1]
                pred_patch = patch_process(
                    pred_patch,
                    output_patches[-1],
                    patch_overlap_from_starts(self.args.patch_size[0], di, prev_di),
                    0,
                )
            if hi != 0:
                prev_hi = h_list[h_list.index(hi) - 1]
                pred_patch = patch_process(
                    pred_patch,
                    output_patches[-num_x],
                    patch_overlap_from_starts(self.args.patch_size[1], hi, prev_hi),
                    1,
                )
            if wi != 0:
                prev_wi = w_list[w_list.index(wi) - 1]
                pred_patch = patch_process(
                    pred_patch,
                    output_patches[-num_y * num_x],
                    patch_overlap_from_starts(self.args.patch_size[2], wi, prev_wi),
                    2,
                )

            refined[
                di : di + self.args.patch_size[0],
                hi : hi + self.args.patch_size[1],
                wi : wi + self.args.patch_size[2],
            ] = pred_patch
            output_patches.append(pred_patch)

            if (idx + 1) % 20 == 0 or idx + 1 == len(corners):
                print(f"  late refined {idx + 1}/{len(corners)} patches", flush=True)

        return (refined * mask).astype(np.float32)

    def load_case(self, mat_path: Path):
        from scipy.io import loadmat

        data = loadmat(mat_path)
        phi = np.asarray(data["phi_use"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=np.float32)
        pad_size = np.asarray(data["pad_size"]).reshape(-1).astype(int).tolist()
        d2 = np.asarray(data["D2"], dtype=np.float32)
        calci = np.asarray(data["calci_use"], dtype=np.float32) if "calci_use" in data else None
        return {
            "phi": phi,
            "mask": mask,
            "pad_size": pad_size,
            "d2": d2,
            "calci": calci,
        }

    def crop_volume(self, volume: np.ndarray, pad_size: list[int]) -> np.ndarray:
        return volume[
            int(pad_size[0] / 2) : volume.shape[0] - int(pad_size[0] / 2),
            int(pad_size[1] / 2) : volume.shape[1] - int(pad_size[1] / 2),
            int(pad_size[2] / 2) : volume.shape[2] - int(pad_size[2] / 2),
        ]

    def save_output(self, volume: np.ndarray, out_prefix: Path) -> None:
        np.save(out_prefix.with_suffix(".npy"), volume.astype(np.float32))
        if not self.args.save_nifti:
            return
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("nibabel is required for --save-nifti.") from exc

        vox = self.args.vox
        affine = np.array(
            [
                [vox[0], 0, 0, 0],
                [0, vox[1], 0, 0],
                [0, 0, vox[2], 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(out_prefix) + ".nii.gz")

    def run_case(self, mat_path: Path) -> dict:
        case = self.load_case(mat_path)
        phi = case["phi"]
        mask = case["mask"]
        pad_size = case["pad_size"]
        d2 = case["d2"]
        calci = case["calci"]

        data_shape = tuple(int(v) for v in phi.shape)
        d_list, h_list, w_list = overlapping_grid_indices(data_shape, self.args.patch_size, self.args.stride)
        corners = [(i, j, k) for k in w_list for j in h_list for i in d_list]
        patch_dipole_kernel = create_dipole_kernel(
            self.args.patch_size,
            voxel_size=self.args.vox,
            b0_dir=self.args.b0_dir,
            device=self.device,
        )

        stitched = np.zeros(data_shape, dtype=np.float32)
        output_patches: list[np.ndarray] = []
        num_x = len(d_list)
        num_y = len(h_list)

        print(f"\nCase: {mat_path.parent.name}")
        print(f"Input volume shape: {data_shape}")
        print(f"Patch size: {self.args.patch_size}, stride: {self.args.stride}, patches: {len(corners)}")
        print(f"Has calci_use: {calci is not None}")
        if self.args.patch_calci_aware_dc and calci is None:
            print("Warning: --patch-calci-aware-dc was requested, but calci_use is absent. Falling back to standard patch DC.")
        if self.args.cone_polish and not self.args.full_volume_final_dc:
            print("Warning: --cone-polish requested without --full-volume-final-dc. Cone blend will be a near no-op.", flush=True)

        case_start_time = time.perf_counter()
        pre_init_seconds = 0.0
        warm_init_volume = None
        external_init_volume = None
        if self.args.pre_init_full_volume_dc:
            print(
                f"Applying pre-init full-volume DC warm start (lam={self.args.pre_init_lam}, cg_steps={self.args.pre_init_cg_steps})",
                flush=True,
            )
            pre_init_start = time.perf_counter()
            field_t = torch.from_numpy(phi).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            d2_t = torch.from_numpy(d2).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            calci_t = None
            if calci is not None:
                calci_t = torch.from_numpy(calci).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            x_init_t = adjoint_field_to_chi(field_t, d2_t, mask=mask_t)
            warm_init_volume = (
                full_volume_data_consistency_cg(
                    chi_init=x_init_t,
                    field_gt=field_t,
                    mask=mask_t,
                    dipole_kernel=d2_t,
                    calci=calci_t,
                    lam=self.args.pre_init_lam,
                    num_cg_steps=self.args.pre_init_cg_steps,
                )
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            warm_init_volume = warm_init_volume * mask
            pre_init_seconds = time.perf_counter() - pre_init_start

        if self.args.init_volume_path:
            external_init_start = time.perf_counter()
            external_init_volume = np.asarray(
                nib.load(self.args.init_volume_path).dataobj,
                dtype=np.float32,
            )
            if external_init_volume.shape != phi.shape:
                raise ValueError(
                    f"External init volume shape {external_init_volume.shape} does not match input field shape {phi.shape}."
                )
            external_init_volume = external_init_volume * mask
            pre_init_seconds += time.perf_counter() - external_init_start

        for idx, (di, hi, wi) in enumerate(corners):
            field_patch = torch.from_numpy(phi[di : di + self.args.patch_size[0], hi : hi + self.args.patch_size[1], wi : wi + self.args.patch_size[2]]).to(
                self.device, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)
            mask_patch = torch.from_numpy(mask[di : di + self.args.patch_size[0], hi : hi + self.args.patch_size[1], wi : wi + self.args.patch_size[2]]).to(
                self.device, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)
            calci_patch = None
            if self.args.patch_calci_aware_dc and calci is not None:
                calci_patch = torch.from_numpy(
                    calci[
                        di : di + self.args.patch_size[0],
                        hi : hi + self.args.patch_size[1],
                        wi : wi + self.args.patch_size[2],
                    ]
                ).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            x_init_patch = None
            if external_init_volume is not None:
                x_init_patch = torch.from_numpy(
                    external_init_volume[
                        di : di + self.args.patch_size[0],
                        hi : hi + self.args.patch_size[1],
                        wi : wi + self.args.patch_size[2],
                    ]
                ).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            elif warm_init_volume is not None:
                x_init_patch = torch.from_numpy(
                    warm_init_volume[
                        di : di + self.args.patch_size[0],
                        hi : hi + self.args.patch_size[1],
                        wi : wi + self.args.patch_size[2],
                    ]
                ).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            pred_patch = (
                self.infer_patch(
                    field_patch,
                    mask_patch,
                    patch_dipole_kernel,
                    infer_nfe=self.args.infer_nfe,
                    calci_patch=calci_patch,
                    x_init_override=x_init_patch,
                )
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            if di != 0:
                prev_di = d_list[d_list.index(di) - 1]
                pred_patch = patch_process(
                    pred_patch,
                    output_patches[-1],
                    patch_overlap_from_starts(self.args.patch_size[0], di, prev_di),
                    0,
                )
            if hi != 0:
                prev_hi = h_list[h_list.index(hi) - 1]
                pred_patch = patch_process(
                    pred_patch,
                    output_patches[-num_x],
                    patch_overlap_from_starts(self.args.patch_size[1], hi, prev_hi),
                    1,
                )
            if wi != 0:
                prev_wi = w_list[w_list.index(wi) - 1]
                pred_patch = patch_process(
                    pred_patch,
                    output_patches[-num_y * num_x],
                    patch_overlap_from_starts(self.args.patch_size[2], wi, prev_wi),
                    2,
                )

            stitched[
                di : di + self.args.patch_size[0],
                hi : hi + self.args.patch_size[1],
                wi : wi + self.args.patch_size[2],
            ] = pred_patch
            output_patches.append(pred_patch)

            if (idx + 1) % 20 == 0 or idx + 1 == len(corners):
                print(f"  processed {idx + 1}/{len(corners)} patches", flush=True)

        stitched = stitched * mask

        prepolish = stitched.copy()
        if self.args.full_volume_final_dc:
            print(
                f"Applying full-volume final DC polish (lam={self.args.full_volume_lam}, cg_steps={self.args.full_volume_cg_steps})",
                flush=True,
            )
            stitched_t = torch.from_numpy(stitched).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            field_t = torch.from_numpy(phi).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            d2_t = torch.from_numpy(d2).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            calci_t = None
            if calci is not None:
                calci_t = torch.from_numpy(calci).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            stitched = (
                full_volume_data_consistency_cg(
                    chi_init=stitched_t,
                    field_gt=field_t,
                    mask=mask_t,
                    dipole_kernel=d2_t,
                    calci=calci_t,
                    lam=self.args.full_volume_lam,
                    num_cg_steps=self.args.full_volume_cg_steps,
                )
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            stitched = stitched * mask

        pre_cone = stitched.copy()
        if self.args.cone_polish:
            print(
                f"Applying cone polish (threshold={self.args.cone_polish_threshold}, power={self.args.cone_polish_power})",
                flush=True,
            )
            stitched = cone_frequency_blend(
                prior_volume=prepolish,
                dc_volume=stitched,
                d2=d2,
                mask=mask,
                threshold=self.args.cone_polish_threshold,
                power=self.args.cone_polish_power,
                device=self.device,
            )

        if self.args.late_refine:
            print(
                f"Applying late refinement (sigma={self.args.late_refine_sigma}, lam={self.args.late_refine_lam}, cg_steps={self.args.late_refine_cg_steps})",
                flush=True,
            )
            stitched = self.late_refine_volume(
                x_start_volume=stitched,
                phi=phi,
                mask=mask,
                patch_dipole_kernel=patch_dipole_kernel,
                corners=corners,
                num_x=num_x,
                num_y=num_y,
            )

        cropped = self.crop_volume(stitched, pad_size=pad_size)
        cropped_prepolish = self.crop_volume(prepolish, pad_size=pad_size)
        cropped_precone = self.crop_volume(pre_cone, pad_size=pad_size)

        out_dir = Path(self.args.output_dir).expanduser().resolve() / mat_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = out_dir / f"sample_stage4_v3_nfe{self.args.infer_nfe}"
        if self.args.full_volume_final_dc:
            self.save_output(cropped_prepolish, out_dir / f"sample_stage4_v3_nfe{self.args.infer_nfe}_prepolish")
        if self.args.cone_polish:
            self.save_output(cropped_precone, out_dir / f"sample_stage4_v3_nfe{self.args.infer_nfe}_precone")
        self.save_output(cropped, out_prefix)

        meta = {
            "case": mat_path.parent.name,
            "mat_path": str(mat_path),
            "input_shape_padded": list(data_shape),
            "output_shape_cropped": list(cropped.shape),
            "pad_size": pad_size,
            "patch_size": list(self.args.patch_size),
            "stride": list(self.args.stride),
            "infer_nfe": int(self.args.infer_nfe),
            "patch_calci_aware_dc": bool(self.args.patch_calci_aware_dc),
            "full_volume_final_dc": bool(self.args.full_volume_final_dc),
            "full_volume_lam": float(self.args.full_volume_lam),
            "full_volume_cg_steps": int(self.args.full_volume_cg_steps),
            "cone_polish": bool(self.args.cone_polish),
            "cone_polish_threshold": float(self.args.cone_polish_threshold),
            "cone_polish_power": float(self.args.cone_polish_power),
            "late_refine": bool(self.args.late_refine),
            "late_refine_sigma": float(self.args.late_refine_sigma),
            "late_refine_lam": float(self.args.late_refine_lam),
            "late_refine_cg_steps": int(self.args.late_refine_cg_steps),
            "checkpoint": str(Path(self.args.ckpt).expanduser().resolve()),
            "checkpoint_kind": self.checkpoint_kind,
            "has_calci": bool(calci is not None),
            "output_masking_policy": "stitched volume is multiplied by mask after patch stitching and again after full-volume DC",
            "pre_init_full_volume_dc": bool(self.args.pre_init_full_volume_dc),
            "pre_init_lam": float(self.args.pre_init_lam),
            "pre_init_cg_steps": int(self.args.pre_init_cg_steps),
            "init_scale": float(self.args.init_scale),
            "init_volume_path": str(Path(self.args.init_volume_path).expanduser().resolve()) if self.args.init_volume_path else None,
            "pre_init_seconds": float(pre_init_seconds),
            "case_runtime_seconds": float(time.perf_counter() - case_start_time),
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def run(self) -> None:
        output_dir = Path(self.args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.write_run_config(output_dir)

        requested_subjects = parse_subject_list(self.args.subject_allowlist)
        denied_subjects = parse_subject_list(self.args.subject_denylist)
        if self.args.mat_path:
            mat_paths = [Path(self.args.mat_path).expanduser().resolve()]
        else:
            root = Path(self.args.data_root).expanduser().resolve()
            mat_paths = sorted(root.rglob("data_DiffusionQSM.mat"))
        if requested_subjects is not None:
            allowed = set(requested_subjects)
            mat_paths = [path for path in mat_paths if path.parent.name in allowed]
        if denied_subjects is not None:
            denied = set(denied_subjects)
            mat_paths = [path for path in mat_paths if path.parent.name not in denied]
        if not mat_paths:
            raise FileNotFoundError("No data_DiffusionQSM.mat files found.")

        all_meta = []
        for mat_path in mat_paths:
            all_meta.append(self.run_case(mat_path))

        (output_dir / "run_summary.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
        print("\nDone.")
        print(f"Output dir: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None, help="Root directory to recursively search for data_DiffusionQSM.mat")
    parser.add_argument("--mat-path", type=str, default=None, help="Single data_DiffusionQSM.mat path.")
    parser.add_argument("--subject-allowlist", type=str, default=None)
    parser.add_argument("--subject-denylist", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./runs/stage4_v3_diffusionqsm_infer")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--diffusion-root", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--infer-nfe", type=int, default=2)
    parser.add_argument("--infer-use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dc-policy", choices=["fixed", "stepwise_lambda", "stepwise_lambda+final_polish"], default="fixed")
    parser.add_argument("--lambda-dc", type=str, default="0.1")
    parser.add_argument("--cg-steps-infer", type=str, default="8")
    parser.add_argument("--final-polish-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-polish-lam", type=float, default=None)
    parser.add_argument("--final-polish-steps-infer", type=int, default=None)
    parser.add_argument(
        "--step-adapter-mode",
        choices=["shared_only", "sigma_step_embed", "light_film_adapter", "time_embed_residual"],
        default="shared_only",
    )
    parser.add_argument("--step-adapter-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--conditioning-mode",
        choices=[
            "none",
            "physics_aware_stem",
            "physics_aware_concat",
            "morph_static_stem",
            "residual_aware_stem",
            "residual_aware_concat",
            "residual_aware_heavy_gate",
            "morph_dynamic_stem",
        ],
        default="none",
    )
    parser.add_argument("--conditioning-hidden-dim", type=int, default=16)
    parser.add_argument("--conditioning-norm-mode", choices=["none", "masked_std"], default="masked_std")
    parser.add_argument("--conditioning-cone-threshold", type=float, default=0.05)
    parser.add_argument("--init-error-proxy-sigma", type=float, default=1.0)
    parser.add_argument(
        "--morph-prior-mode",
        choices=["none", "replace_xinit", "parallel_xinit", "residual_delta", "dynamic_measurement"],
        default="none",
    )
    parser.add_argument("--morph-residual-scale", type=float, default=0.5)
    parser.add_argument("--morph-tkd-low", type=float, default=0.1)
    parser.add_argument("--morph-tkd-high", type=float, default=0.2)
    parser.add_argument("--morph-cone-threshold", type=float, default=0.1)
    parser.add_argument("--morph-cone-power", type=float, default=0.2)
    parser.add_argument("--student-sigma-max-factor", type=float, default=1.0)
    parser.add_argument("--sigma-max-override", type=float, default=None,
                        help="Override ckpt sigma_max for the infer schedule (default: read ckpt).")
    parser.add_argument("--start-noise-sigma", type=float, default=None,
                        help="If >0, replace x_start with (init + sigma*eps)*mask, fixed seed 20261026.")
    parser.add_argument("--patch-size", type=str, default="64,64,64")
    parser.add_argument("--stride", type=str, default="48,48,48")
    parser.add_argument("--vox", type=str, default="1,1,1")
    parser.add_argument("--b0-dir", type=str, default="0,0,1")
    parser.add_argument("--patch-calci-aware-dc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-volume-final-dc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-volume-lam", type=float, default=0.1)
    parser.add_argument("--full-volume-cg-steps", type=int, default=8)
    parser.add_argument("--pre-init-full-volume-dc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pre-init-lam", type=float, default=0.08)
    parser.add_argument("--pre-init-cg-steps", type=int, default=4)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--init-volume-path", type=str, default=None)
    parser.add_argument("--cone-polish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cone-polish-threshold", type=float, default=0.08)
    parser.add_argument("--cone-polish-power", type=float, default=1.0)
    parser.add_argument("--late-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--late-refine-sigma", type=float, default=0.05)
    parser.add_argument("--late-refine-lam", type=float, default=0.1)
    parser.add_argument("--late-refine-cg-steps", type=int, default=4)
    parser.add_argument("--save-nifti", action=argparse.BooleanOptionalAction, default=True)
    return parser


def parse_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    if bool(args.mat_path) == bool(args.data_root):
        raise ValueError("Specify exactly one of --mat-path or --data-root")
    if args.infer_nfe < 1:
        raise ValueError("--infer-nfe must be >= 1")
    if args.student_sigma_max_factor <= 0:
        raise ValueError("--student-sigma-max-factor must be > 0")
    if args.sigma_max_override is not None and args.sigma_max_override <= 0:
        raise ValueError("--sigma-max-override must be > 0")
    if args.start_noise_sigma is not None and args.start_noise_sigma < 0:
        raise ValueError("--start-noise-sigma must be >= 0")
    if args.conditioning_hidden_dim < 1:
        raise ValueError("--conditioning-hidden-dim must be >= 1")
    if args.conditioning_cone_threshold <= 0:
        raise ValueError("--conditioning-cone-threshold must be > 0")
    if args.init_error_proxy_sigma <= 0:
        raise ValueError("--init-error-proxy-sigma must be > 0")
    if args.full_volume_lam <= 0:
        raise ValueError("--full-volume-lam must be > 0")
    if args.full_volume_cg_steps < 0:
        raise ValueError("--full-volume-cg-steps must be >= 0")
    if args.pre_init_lam <= 0:
        raise ValueError("--pre-init-lam must be > 0")
    if args.pre_init_cg_steps < 0:
        raise ValueError("--pre-init-cg-steps must be >= 0")
    if args.init_scale <= 0:
        raise ValueError("--init-scale must be > 0")
    if args.cone_polish_threshold <= 0:
        raise ValueError("--cone-polish-threshold must be > 0")
    if args.cone_polish_power <= 0:
        raise ValueError("--cone-polish-power must be > 0")
    if args.late_refine_sigma <= 0:
        raise ValueError("--late-refine-sigma must be > 0")
    if args.late_refine_lam <= 0:
        raise ValueError("--late-refine-lam must be > 0")
    if args.late_refine_cg_steps < 0:
        raise ValueError("--late-refine-cg-steps must be >= 0")
    args.patch_size = parse_triplet(args.patch_size, int)
    args.stride = parse_triplet(args.stride, int)
    args.vox = parse_triplet(args.vox, float)
    args.b0_dir = parse_triplet(args.b0_dir, float)
    return args


def main() -> None:
    raw_argv = sys.argv[1:]
    parser = build_parser()
    args = parse_runtime_args(parser.parse_args(raw_argv))
    args.provided_cli_dests = sorted(extract_provided_cli_dests(parser, raw_argv))
    runner = Stage4DiffusionQSMInference(args)
    runner.run()


if __name__ == "__main__":
    main()
