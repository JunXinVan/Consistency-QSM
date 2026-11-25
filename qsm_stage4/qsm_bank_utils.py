#!/usr/bin/env python3
"""
Shared utilities for QSM Stage 4 cache generation, specialization training, and inference.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DCStepConfig:
    lam: float
    num_cg_steps: int
    tol: float = 1e-6
    enabled: bool = True


@dataclass(frozen=True)
class DCRolloutPolicy:
    mode: str
    steps: tuple[DCStepConfig, ...]
    final_polish: Optional[DCStepConfig] = None

    def validate(self, num_steps: int) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if len(self.steps) != num_steps:
            raise ValueError(
                f"DC rollout policy expects {len(self.steps)} steps, but rollout requested {num_steps} steps."
            )

    def step_config(self, step_index: int) -> DCStepConfig:
        return self.steps[step_index]


@dataclass
class RefinementRolloutResult:
    final_state: torch.Tensor
    states: list[torch.Tensor]
    priors: list[torch.Tensor]
    sigma_schedule: list[torch.Tensor]


CONDITIONING_MODE_CHANNELS = {
    "none": 0,
    "physics_aware_stem": 5,
    "physics_aware_concat": 5,
    "morph_static_stem": 6,
    "residual_aware_stem": 8,
    "residual_aware_concat": 8,
    "residual_aware_heavy_gate": 8,
    "morph_dynamic_stem": 7,
}
STEM_CONDITIONING_MODES = {
    "physics_aware_stem",
    "morph_static_stem",
    "residual_aware_stem",
    "residual_aware_heavy_gate",
    "morph_dynamic_stem",
}
CONCAT_CONDITIONING_MODES = {"physics_aware_concat", "residual_aware_concat"}
DYNAMIC_CONDITIONING_MODES = {
    "residual_aware_stem",
    "residual_aware_concat",
    "residual_aware_heavy_gate",
    "morph_dynamic_stem",
}


def conditioning_channels_for_mode(mode: str) -> int:
    try:
        return int(CONDITIONING_MODE_CHANNELS[str(mode)])
    except KeyError as exc:
        expected = ", ".join(sorted(CONDITIONING_MODE_CHANNELS))
        raise ValueError(
            f"Unknown conditioning mode {mode!r}. Expected one of: {expected}."
        ) from exc


def conditioning_mode_uses_stem(mode: str) -> bool:
    return str(mode) in STEM_CONDITIONING_MODES


def conditioning_mode_uses_concat(mode: str) -> bool:
    return str(mode) in CONCAT_CONDITIONING_MODES


def conditioning_mode_is_dynamic(mode: str) -> bool:
    return str(mode) in DYNAMIC_CONDITIONING_MODES


@dataclass
class ConditioningSpec:
    mode: str
    static_volume: Optional[torch.Tensor] = None
    x_init: Optional[torch.Tensor] = None
    field_input: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    dipole_kernel: Optional[torch.Tensor] = None
    cone_risk: Optional[torch.Tensor] = None
    morph_qsm0: Optional[torch.Tensor] = None
    cone_threshold: float = 0.05
    init_error_proxy_sigma: float = 1.0
    norm_mode: str = "masked_std"

    def build(self, current_state: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.mode == "none":
            return None
        if self.static_volume is not None:
            return self.static_volume
        if not conditioning_mode_is_dynamic(self.mode):
            raise RuntimeError(
                f"Conditioning mode {self.mode!r} requires a precomputed static volume."
            )
        if current_state is None:
            raise RuntimeError(
                f"Dynamic conditioning mode {self.mode!r} requires the current rollout state."
            )
        if self.field_input is None or self.mask is None or self.dipole_kernel is None:
            raise RuntimeError(
                f"Dynamic conditioning mode {self.mode!r} is missing field/mask/dipole context."
            )
        if self.mode == "morph_dynamic_stem":
            if self.morph_qsm0 is None:
                raise RuntimeError("morph_dynamic_stem requires a morphology-aware qsm0 prior.")
            return build_morph_dynamic_conditioning_volume(
                current_state=current_state,
                morph_qsm0=self.morph_qsm0,
                field_input=self.field_input,
                mask=self.mask,
                dipole_kernel=self.dipole_kernel,
                cone_threshold=self.cone_threshold,
                cone_risk=self.cone_risk,
                norm_mode=self.norm_mode,
            )
        return build_residual_aware_conditioning_volume(
            current_state=current_state,
            field_input=self.field_input,
            mask=self.mask,
            dipole_kernel=self.dipole_kernel,
            cone_threshold=self.cone_threshold,
            cone_risk=self.cone_risk,
            init_error_proxy_sigma=self.init_error_proxy_sigma,
            norm_mode=self.norm_mode,
        )


class StepAdaptiveStudentModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        mode: str = "shared_only",
        hidden_dim: int = 32,
        conditioning_mode: str = "none",
        conditioning_channels: int = 0,
        conditioning_hidden_dim: int = 16,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if mode not in {"shared_only", "sigma_step_embed", "light_film_adapter", "time_embed_residual"}:
            raise ValueError(
                f"Unknown step adapter mode {mode!r}. Expected one of: "
                "shared_only, sigma_step_embed, light_film_adapter, time_embed_residual."
            )

        self.base_model = base_model
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        self.conditioning_mode = str(conditioning_mode)
        self.conditioning_channels = int(conditioning_channels)
        self.conditioning_hidden_dim = int(conditioning_hidden_dim)
        self._rollout_context: Optional[dict[str, object]] = None

        if mode == "shared_only":
            self.adapter = None
        elif mode == "sigma_step_embed":
            self.adapter = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2),
            )
        elif mode == "time_embed_residual":
            time_embed_dim = self._infer_time_embed_dim(base_model)
            self.adapter = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, time_embed_dim),
            )
            # Keep the wrapped model behavior identical to shared_only at init.
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        else:
            self.adapter = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 4),
            )

        if self.conditioning_mode == "none" or self.conditioning_channels <= 0:
            self.conditioning_stem = None
        elif conditioning_mode_uses_stem(self.conditioning_mode):
            if self.conditioning_mode == "residual_aware_heavy_gate":
                self.conditioning_stem = ResidualAwareGatedStem(
                    in_channels=1 + self.conditioning_channels,
                    hidden_dim=self.conditioning_hidden_dim,
                )
            else:
                self.conditioning_stem = PhysicsAwareInputStem(
                    in_channels=1 + self.conditioning_channels,
                    hidden_dim=self.conditioning_hidden_dim,
                )
        elif conditioning_mode_uses_concat(self.conditioning_mode):
            self.conditioning_stem = None
        else:
            expected = ", ".join(sorted(CONDITIONING_MODE_CHANNELS))
            raise ValueError(
                f"Unknown conditioning_mode {conditioning_mode!r}. "
                f"Expected one of: {expected}."
            )

        expected_in_channels = expected_base_model_in_channels(
            self.conditioning_mode,
            conditioning_channels=self.conditioning_channels,
        )
        actual_in_channels = infer_base_model_in_channels(self.base_model)
        if actual_in_channels != expected_in_channels:
            raise ValueError(
                f"Base model input channel mismatch for conditioning_mode={self.conditioning_mode!r}: "
                f"expected {expected_in_channels}, got {actual_in_channels}."
            )

    def adapter_parameter_count(self) -> int:
        if self.adapter is None:
            return 0
        return sum(parameter.numel() for parameter in self.adapter.parameters())

    @staticmethod
    def _infer_time_embed_dim(base_model: nn.Module) -> int:
        time_embed = getattr(base_model, "time_embed", None)
        if isinstance(time_embed, nn.Sequential) and len(time_embed) > 0:
            last_layer = time_embed[-1]
            if isinstance(last_layer, nn.Linear):
                return int(last_layer.out_features)
        raise ValueError(
            "time_embed_residual requires a base model with a `time_embed` sequential "
            "module ending in nn.Linear."
        )

    def set_rollout_context(
        self,
        step_index: int,
        total_steps: int,
        sigmas: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> None:
        self._rollout_context = {
            "step_index": int(step_index),
            "total_steps": int(total_steps),
            "sigmas": sigmas.detach(),
            "conditioning": conditioning,
        }

    def clear_rollout_context(self) -> None:
        self._rollout_context = None

    def _build_condition(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._rollout_context is None:
            step_ratio = torch.zeros(batch_size, device=device, dtype=dtype)
            log_sigma = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            total_steps = max(int(self._rollout_context["total_steps"]), 1)
            step_index = int(self._rollout_context["step_index"])
            step_ratio_value = 0.0 if total_steps <= 1 else step_index / max(total_steps - 1, 1)
            step_ratio = torch.full((batch_size,), float(step_ratio_value), device=device, dtype=dtype)

            sigmas = self._rollout_context["sigmas"]
            if not isinstance(sigmas, torch.Tensor):
                raise TypeError(f"Expected sigmas tensor in rollout context, got {type(sigmas)!r}")
            if sigmas.ndim != 1:
                raise ValueError(f"Expected 1D sigmas tensor, got shape {tuple(sigmas.shape)}")
            if sigmas.shape[0] != batch_size:
                raise ValueError(
                    f"Context sigma batch mismatch: expected {batch_size}, got {sigmas.shape[0]}"
                )
            log_sigma = torch.log(sigmas.to(device=device, dtype=dtype).clamp_min(1e-6))

        return torch.stack([step_ratio, log_sigma], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        conditioning = None
        if self._rollout_context is not None:
            conditioning = self._rollout_context.get("conditioning")
        if conditioning is not None:
            if not isinstance(conditioning, torch.Tensor):
                raise TypeError(
                    f"Expected conditioning tensor in rollout context, got {type(conditioning)!r}"
                )
            if conditioning.ndim != x.ndim:
                raise ValueError(
                    f"Conditioning ndim mismatch: expected {x.ndim}, got {conditioning.ndim}"
                )
            if conditioning.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Conditioning batch mismatch: expected {x.shape[0]}, got {conditioning.shape[0]}"
                )
            conditioning = conditioning.to(device=x.device, dtype=x.dtype)

        if conditioning_mode_uses_stem(self.conditioning_mode):
            if conditioning is None:
                raise RuntimeError(
                    f"{self.conditioning_mode} requires a conditioning tensor in rollout context."
                )
            x = self.conditioning_stem(x, conditioning)
        elif conditioning_mode_uses_concat(self.conditioning_mode):
            if conditioning is None:
                raise RuntimeError(
                    f"{self.conditioning_mode} requires a conditioning tensor in rollout context."
                )
            x = torch.cat([x, conditioning], dim=1)

        if self.mode == "shared_only" or self.adapter is None:
            return self.base_model(x, t)

        cond = self._build_condition(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        params = self.adapter(cond)

        if self.mode == "time_embed_residual":
            return self.base_model(x, t, extra_emb=params)

        if self.mode == "sigma_step_embed":
            out_scale, out_bias = params.chunk(2, dim=1)
            pred = self.base_model(x, t)
            return pred * (1.0 + out_scale.view(-1, 1, 1, 1, 1)) + out_bias.view(-1, 1, 1, 1, 1)

        in_scale, in_bias, out_scale, out_bias = params.chunk(4, dim=1)
        x_mod = x * (1.0 + in_scale.view(-1, 1, 1, 1, 1)) + in_bias.view(-1, 1, 1, 1, 1)
        pred = self.base_model(x_mod, t)
        return pred * (1.0 + out_scale.view(-1, 1, 1, 1, 1)) + out_bias.view(-1, 1, 1, 1, 1)


class PhysicsAwareInputStem(nn.Module):
    """Residual fusion stem that upgrades scalar x_t into a richer physics-aware representation."""

    def __init__(self, in_channels: int, hidden_dim: int = 16):
        super().__init__()
        if in_channels < 2:
            raise ValueError(f"PhysicsAwareInputStem expects at least 2 input channels, got {in_channels}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(4, self.hidden_dim), num_channels=self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, 1, kernel_size=3, padding=1),
        )
        # Start close to the old identity model. The residual branch is trained to
        # learn how much the extra physics-aware channels should perturb x_t.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, primary: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if primary.ndim != 5 or conditioning.ndim != 5:
            raise ValueError(
                f"PhysicsAwareInputStem expects 5D tensors, got primary={tuple(primary.shape)}, "
                f"conditioning={tuple(conditioning.shape)}"
            )
        if conditioning.shape[1] != self.in_channels - 1:
            raise ValueError(
                f"Conditioning channel mismatch: expected {self.in_channels - 1}, got {conditioning.shape[1]}"
            )
        fused = torch.cat([primary, conditioning], dim=1)
        return primary + self.net(fused)


class ResidualAwareGatedStem(nn.Module):
    """Aggressive residual-aware stem that strongly modulates x_t before the base model."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        residual_gain: float = 2.5,
        gate_gain: float = 2.0,
    ):
        super().__init__()
        if in_channels < 2:
            raise ValueError(f"ResidualAwareGatedStem expects at least 2 input channels, got {in_channels}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.residual_gain = float(residual_gain)
        self.gate_gain = float(gate_gain)

        groups = min(8, self.hidden_dim)
        self.trunk = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=self.hidden_dim),
            nn.SiLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=self.hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Conv3d(self.hidden_dim, 1, kernel_size=3, padding=1)
        self.gate_head = nn.Conv3d(self.hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, primary: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if primary.ndim != 5 or conditioning.ndim != 5:
            raise ValueError(
                f"ResidualAwareGatedStem expects 5D tensors, got primary={tuple(primary.shape)}, "
                f"conditioning={tuple(conditioning.shape)}"
            )
        if conditioning.shape[1] != self.in_channels - 1:
            raise ValueError(
                f"Conditioning channel mismatch: expected {self.in_channels - 1}, got {conditioning.shape[1]}"
            )
        fused = torch.cat([primary, conditioning], dim=1)
        features = self.trunk(fused)
        gate = torch.tanh(self.gate_head(features))
        delta = self.delta_head(features)
        return primary * (1.0 + self.gate_gain * gate) + self.residual_gain * delta


def infer_base_model_in_channels(base_model: nn.Module) -> int:
    first_block = getattr(base_model, "input_blocks", None)
    if first_block is None or len(first_block) == 0:
        raise ValueError("Base model does not expose input_blocks needed to infer input channels.")
    first_module = first_block[0]
    if not isinstance(first_module, nn.Sequential) or len(first_module) == 0:
        raise ValueError("Unexpected input_blocks[0] structure while inferring input channels.")
    first_conv = first_module[0]
    if not isinstance(first_conv, nn.Conv3d):
        raise ValueError("Expected input_blocks[0][0] to be nn.Conv3d while inferring input channels.")
    return int(first_conv.in_channels)


def expected_base_model_in_channels(conditioning_mode: str, conditioning_channels: int) -> int:
    if conditioning_mode_uses_concat(conditioning_mode):
        channels = int(conditioning_channels)
        if channels <= 0:
            channels = conditioning_channels_for_mode(conditioning_mode)
        return 1 + channels
    return 1


def maybe_expand_input_conv_weight(
    state_dict: dict[str, torch.Tensor],
    weight_key: str,
    target_in_channels: int,
) -> dict[str, torch.Tensor]:
    if weight_key not in state_dict:
        return state_dict

    weight = state_dict[weight_key]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 5:
        raise ValueError(
            f"Expected a 5D Conv3d weight tensor for {weight_key!r}, got {type(weight)!r} shape={getattr(weight, 'shape', None)}"
        )

    current_in_channels = int(weight.shape[1])
    target_in_channels = int(target_in_channels)
    if current_in_channels == target_in_channels:
        return state_dict
    if current_in_channels < 1 or target_in_channels < 1 or current_in_channels > target_in_channels:
        raise RuntimeError(
            f"Cannot adapt input conv {weight_key!r} from in_channels={current_in_channels} "
            f"to target_in_channels={target_in_channels}."
        )

    expanded = torch.zeros(
        (weight.shape[0], target_in_channels, *weight.shape[2:]),
        dtype=weight.dtype,
        device=weight.device,
    )
    expanded[:, :current_in_channels] = weight
    patched = OrderedDict(state_dict)
    patched[weight_key] = expanded
    return patched


def parse_triplet(value: str, cast=float) -> tuple:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError(f"Expected three comma-separated values, got: {value!r}")
    return tuple(cast(part) for part in parts)


def parse_int_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Expected a comma-separated integer list, got: {value!r}")
    return [int(part) for part in parts]


def parse_float_list(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Expected a comma-separated float list, got: {value!r}")
    return [float(part) for part in parts]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def centered_fftn(x: torch.Tensor, dims: Sequence[int] = (-3, -2, -1)) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"),
        dim=dims,
    )


def centered_ifftn(x: torch.Tensor, dims: Sequence[int] = (-3, -2, -1)) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"),
        dim=dims,
    )


def create_dipole_kernel(
    shape: Sequence[int],
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    b0_dir: Sequence[float] = (0.0, 0.0, 1.0),
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if len(shape) != 3:
        raise ValueError(f"Dipole kernel expects a 3D shape, got: {shape}")

    voxel_size = tuple(float(v) for v in voxel_size)
    b0 = torch.tensor(b0_dir, device=device, dtype=dtype)
    b0_norm = torch.linalg.norm(b0).item()
    if not b0_norm > 0:
        raise ValueError(f"b0_dir must be non-zero, got: {b0_dir}")
    b0 = b0 / max(b0_norm, 1e-12)

    kx = torch.fft.fftshift(torch.fft.fftfreq(shape[0], d=voxel_size[0], device=device, dtype=dtype))
    ky = torch.fft.fftshift(torch.fft.fftfreq(shape[1], d=voxel_size[1], device=device, dtype=dtype))
    kz = torch.fft.fftshift(torch.fft.fftfreq(shape[2], d=voxel_size[2], device=device, dtype=dtype))
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")

    k2 = kx.square() + ky.square() + kz.square()
    k_dot_b0 = kx * b0[0] + ky * b0[1] + kz * b0[2]
    kernel = torch.where(
        k2 > 0,
        1.0 / 3.0 - k_dot_b0.square() / k2.clamp_min(1e-12),
        torch.zeros_like(k2),
    )
    return kernel.to(dtype=dtype)


def create_cone_mask(
    dipole_kernel: torch.Tensor,
    cone_threshold: float = 0.05,
    exclude_dc: bool = True,
) -> torch.Tensor:
    if cone_threshold < 0:
        raise ValueError(f"cone_threshold must be >= 0, got {cone_threshold}")

    kernel = dipole_kernel
    while kernel.ndim > 3:
        kernel = kernel.squeeze(0)
    if kernel.ndim != 3:
        raise ValueError(f"Expected a 3D dipole kernel, got shape {tuple(dipole_kernel.shape)}")

    cone_mask = kernel.abs() <= float(cone_threshold)
    if exclude_dc:
        center = tuple(size // 2 for size in kernel.shape)
        cone_mask = cone_mask.clone()
        cone_mask[center] = False
    return cone_mask.to(dtype=kernel.dtype)


def build_spectral_weight_map(
    dipole_kernel: torch.Tensor,
    mode: str = "none",
    cone_threshold: float = 0.05,
    cone_weight: float = 4.0,
    background_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    kernel = dipole_kernel
    while kernel.ndim > 3:
        kernel = kernel.squeeze(0)
    if kernel.ndim != 3:
        raise ValueError(f"Expected a 3D dipole kernel, got shape {tuple(dipole_kernel.shape)}")

    if mode == "none":
        return torch.ones_like(kernel)

    if cone_weight < 0 or background_weight < 0:
        raise ValueError("cone_weight and background_weight must be non-negative")

    abs_kernel = kernel.abs()
    cone_mask = create_cone_mask(kernel, cone_threshold=cone_threshold, exclude_dc=True)
    weight = torch.full_like(kernel, float(background_weight))

    if mode == "cone_weighted":
        weight = torch.where(cone_mask > 0, torch.full_like(weight, float(cone_weight)), weight)
        return weight

    if mode in {"cone_weighted_residual", "hybrid_cone_residual_dipole"}:
        relative = (cone_threshold - abs_kernel).clamp_min(0.0) / max(cone_threshold, eps)
        boosted = background_weight + (cone_weight - background_weight) * relative
        weight = torch.where(cone_mask > 0, boosted, weight)
        return weight

    raise ValueError(
        f"Unknown spectral weighting mode {mode!r}. Expected one of: "
        "none, cone_weighted, cone_weighted_residual, hybrid_cone_residual_dipole."
    )


def _ensure_5d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    original_shape = tuple(x.shape)
    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        x = x.unsqueeze(1)
    elif x.ndim != 5:
        raise ValueError(f"Expected 3D/4D/5D tensor, got shape {original_shape}")
    return x, original_shape


def _restore_shape(x: torch.Tensor, original_shape: Sequence[int]) -> torch.Tensor:
    if len(original_shape) == 3:
        return x.squeeze(0).squeeze(0)
    if len(original_shape) == 4:
        return x.squeeze(1)
    return x


def _expand_kernel(kernel: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while kernel.ndim < target_ndim:
        kernel = kernel.unsqueeze(0)
    return kernel


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x, torch.zeros_like(x))


def masked_channel_rms(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    x_5d, _ = _ensure_5d(x)
    if mask is None:
        denom = torch.full((x_5d.shape[0],), x_5d[0].numel(), device=x_5d.device, dtype=x_5d.dtype)
        numer = x_5d.square().reshape(x_5d.shape[0], -1).sum(dim=1)
    else:
        mask_5d, _ = _ensure_5d(mask)
        numer = (x_5d.square() * mask_5d).reshape(x_5d.shape[0], -1).sum(dim=1)
        denom = mask_5d.reshape(mask_5d.shape[0], -1).sum(dim=1)
    rms = torch.sqrt(numer / denom.clamp_min(eps))
    return rms.view(-1, 1, 1, 1, 1)


def normalize_conditioning_channel(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: str = "masked_std",
    clamp_value: float = 8.0,
) -> torch.Tensor:
    if mode == "none":
        return x
    if mode != "masked_std":
        raise ValueError(f"Unknown conditioning normalization mode: {mode!r}")
    scale = masked_channel_rms(x, mask).clamp_min(1e-6)
    return (x / scale).clamp(min=-clamp_value, max=clamp_value)


def _build_gaussian_kernel1d(
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
    truncate: float = 2.0,
) -> torch.Tensor:
    sigma = float(max(sigma, 1e-3))
    radius = max(int(round(truncate * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def gaussian_blur3d(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    x_5d, original_shape = _ensure_5d(x)
    dtype = x_5d.dtype
    device = x_5d.device
    kernel_1d = _build_gaussian_kernel1d(sigma=sigma, dtype=dtype, device=device)
    channels = x_5d.shape[1]

    out = x_5d
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel_1d.numel()
        kernel = kernel_1d.view(*shape).repeat(channels, 1, 1, 1, 1)
        padding = [0, 0, 0]
        padding[axis] = kernel_1d.numel() // 2
        out = F.conv3d(out, kernel, padding=tuple(padding), groups=channels)

    return _restore_shape(out, original_shape)


def compute_init_error_proxy(
    x_init: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    smoothed = gaussian_blur3d(x_init, sigma=sigma)
    proxy = (x_init - smoothed).abs()
    if mask is not None:
        mask_5d, _ = _ensure_5d(mask)
        proxy = proxy * mask_5d
    return proxy


def compute_gradient_magnitude3d(
    volume: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    volume_5d, original_shape = _ensure_5d(volume)
    grad_mag = torch.zeros_like(volume_5d)
    num_axes = 0
    pad_by_axis = {
        2: (0, 0, 0, 0, 0, 1),
        3: (0, 0, 0, 1, 0, 0),
        4: (0, 1, 0, 0, 0, 0),
    }

    for axis in (2, 3, 4):
        if volume_5d.shape[axis] <= 1:
            continue
        diff = torch.diff(volume_5d, dim=axis).abs()
        grad_mag = grad_mag + F.pad(diff, pad_by_axis[axis])
        num_axes += 1

    if num_axes > 1:
        grad_mag = grad_mag / float(num_axes)

    if mask is not None:
        mask_5d, _ = _ensure_5d(mask)
        grad_mag = grad_mag * mask_5d
    return _restore_shape(grad_mag, original_shape)


def compute_cone_risk_map(
    volume: torch.Tensor,
    dipole_kernel: torch.Tensor,
    cone_threshold: float = 0.05,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    volume_5d, original_shape = _ensure_5d(volume)
    kernel = build_spectral_weight_map(
        dipole_kernel,
        mode="cone_weighted_residual",
        cone_threshold=cone_threshold,
        cone_weight=1.0,
        background_weight=0.0,
    )
    kernel = _expand_kernel(kernel, volume_5d.ndim)
    risk = centered_ifftn(centered_fftn(_as_complex(volume_5d)) * kernel).real.abs()
    if mask is not None:
        mask_5d, _ = _ensure_5d(mask)
        risk = risk * mask_5d
    return _restore_shape(risk, original_shape)


def build_physics_aware_conditioning_volume(
    x_init: torch.Tensor,
    field_input: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    cone_threshold: float = 0.05,
    cone_risk: Optional[torch.Tensor] = None,
    init_error_proxy_sigma: float = 1.0,
    norm_mode: str = "masked_std",
) -> torch.Tensor:
    if norm_mode not in {"none", "masked_std"}:
        raise ValueError(f"Unsupported norm_mode {norm_mode!r}; expected one of: none, masked_std")

    x_init_5d, _ = _ensure_5d(x_init)
    field_5d, _ = _ensure_5d(field_input)
    mask_5d, _ = _ensure_5d(mask)
    mask_5d = mask_5d.to(device=x_init_5d.device, dtype=x_init_5d.dtype)

    if cone_risk is None:
        cone_risk_5d = compute_cone_risk_map(
            x_init_5d,
            dipole_kernel=dipole_kernel,
            cone_threshold=cone_threshold,
            mask=mask_5d,
        )
    else:
        cone_risk_5d, _ = _ensure_5d(cone_risk)
        cone_risk_5d = cone_risk_5d.to(device=x_init_5d.device, dtype=x_init_5d.dtype) * mask_5d

    init_error_proxy = compute_init_error_proxy(x_init_5d, mask=mask_5d, sigma=init_error_proxy_sigma)

    if norm_mode == "none":
        norm_x_init = x_init_5d
        norm_field = field_5d
        norm_cone = cone_risk_5d
        norm_proxy = init_error_proxy
    else:
        norm_x_init = normalize_conditioning_channel(x_init_5d, mask_5d, mode="masked_std")
        norm_field = normalize_conditioning_channel(field_5d, mask_5d, mode="masked_std")
        norm_cone = normalize_conditioning_channel(cone_risk_5d, mask_5d, mode="masked_std")
        norm_proxy = normalize_conditioning_channel(init_error_proxy, mask_5d, mode="masked_std")

    return torch.cat([norm_x_init, norm_field, mask_5d, norm_cone, norm_proxy], dim=1)


def build_morph_static_conditioning_volume(
    x_init: torch.Tensor,
    morph_qsm0: torch.Tensor,
    field_input: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    cone_threshold: float = 0.05,
    cone_risk: Optional[torch.Tensor] = None,
    init_error_proxy_sigma: float = 1.0,
    norm_mode: str = "masked_std",
) -> torch.Tensor:
    if norm_mode not in {"none", "masked_std"}:
        raise ValueError(f"Unsupported norm_mode {norm_mode!r}; expected one of: none, masked_std")

    x_init_5d, _ = _ensure_5d(x_init)
    morph_5d, _ = _ensure_5d(morph_qsm0)
    field_5d, _ = _ensure_5d(field_input)
    mask_5d, _ = _ensure_5d(mask)
    mask_5d = mask_5d.to(device=x_init_5d.device, dtype=x_init_5d.dtype)
    morph_5d = morph_5d.to(device=x_init_5d.device, dtype=x_init_5d.dtype) * mask_5d
    field_5d = field_5d.to(device=x_init_5d.device, dtype=x_init_5d.dtype)

    if cone_risk is None:
        cone_risk_5d = compute_cone_risk_map(
            morph_5d,
            dipole_kernel=dipole_kernel,
            cone_threshold=cone_threshold,
            mask=mask_5d,
        )
    else:
        cone_risk_5d, _ = _ensure_5d(cone_risk)
        cone_risk_5d = cone_risk_5d.to(device=x_init_5d.device, dtype=x_init_5d.dtype) * mask_5d

    init_error_proxy = compute_init_error_proxy(morph_5d, mask=mask_5d, sigma=init_error_proxy_sigma)

    if norm_mode == "none":
        norm_x_init = x_init_5d
        norm_morph = morph_5d
        norm_field = field_5d
        norm_cone = cone_risk_5d
        norm_proxy = init_error_proxy
    else:
        norm_x_init = normalize_conditioning_channel(x_init_5d, mask_5d, mode="masked_std")
        norm_morph = normalize_conditioning_channel(morph_5d, mask_5d, mode="masked_std")
        norm_field = normalize_conditioning_channel(field_5d, mask_5d, mode="masked_std")
        norm_cone = normalize_conditioning_channel(cone_risk_5d, mask_5d, mode="masked_std")
        norm_proxy = normalize_conditioning_channel(init_error_proxy, mask_5d, mode="masked_std")

    return torch.cat([norm_x_init, norm_morph, norm_field, mask_5d, norm_cone, norm_proxy], dim=1)


def build_residual_aware_conditioning_volume(
    current_state: torch.Tensor,
    field_input: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    cone_threshold: float = 0.05,
    cone_risk: Optional[torch.Tensor] = None,
    init_error_proxy_sigma: float = 1.0,
    norm_mode: str = "masked_std",
) -> torch.Tensor:
    if norm_mode not in {"none", "masked_std"}:
        raise ValueError(f"Unsupported norm_mode {norm_mode!r}; expected one of: none, masked_std")

    current_5d, _ = _ensure_5d(current_state)
    field_5d, _ = _ensure_5d(field_input)
    mask_5d, _ = _ensure_5d(mask)
    mask_5d = mask_5d.to(device=current_5d.device, dtype=current_5d.dtype)
    field_5d = field_5d.to(device=current_5d.device, dtype=current_5d.dtype)

    predicted_field_5d, _ = _ensure_5d(forward_field_from_chi(current_5d, dipole_kernel, mask=mask_5d))
    residual_5d = (predicted_field_5d - field_5d) * mask_5d
    abs_residual_5d = residual_5d.abs()
    residual_grad_5d, _ = _ensure_5d(compute_gradient_magnitude3d(residual_5d, mask=mask_5d))

    if cone_risk is None:
        cone_risk_5d = compute_cone_risk_map(
            current_5d,
            dipole_kernel=dipole_kernel,
            cone_threshold=cone_threshold,
            mask=mask_5d,
        )
    else:
        cone_risk_5d, _ = _ensure_5d(cone_risk)
        cone_risk_5d = cone_risk_5d.to(device=current_5d.device, dtype=current_5d.dtype) * mask_5d

    current_proxy_5d = compute_init_error_proxy(
        current_5d,
        mask=mask_5d,
        sigma=init_error_proxy_sigma,
    )

    if norm_mode == "none":
        norm_current = current_5d
        norm_field = field_5d
        norm_residual = residual_5d
        norm_abs_residual = abs_residual_5d
        norm_cone = cone_risk_5d
        norm_residual_grad = residual_grad_5d
        norm_proxy = current_proxy_5d
    else:
        norm_current = normalize_conditioning_channel(current_5d, mask_5d, mode="masked_std")
        norm_field = normalize_conditioning_channel(field_5d, mask_5d, mode="masked_std")
        norm_residual = normalize_conditioning_channel(residual_5d, mask_5d, mode="masked_std")
        norm_abs_residual = normalize_conditioning_channel(abs_residual_5d, mask_5d, mode="masked_std")
        norm_cone = normalize_conditioning_channel(cone_risk_5d, mask_5d, mode="masked_std")
        norm_residual_grad = normalize_conditioning_channel(residual_grad_5d, mask_5d, mode="masked_std")
        norm_proxy = normalize_conditioning_channel(current_proxy_5d, mask_5d, mode="masked_std")

    return torch.cat(
        [
            norm_current,
            norm_field,
            norm_residual,
            norm_abs_residual,
            mask_5d,
            norm_cone,
            norm_residual_grad,
            norm_proxy,
        ],
        dim=1,
    )


def build_morph_dynamic_conditioning_volume(
    current_state: torch.Tensor,
    morph_qsm0: torch.Tensor,
    field_input: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    cone_threshold: float = 0.05,
    cone_risk: Optional[torch.Tensor] = None,
    norm_mode: str = "masked_std",
) -> torch.Tensor:
    if norm_mode not in {"none", "masked_std"}:
        raise ValueError(f"Unsupported norm_mode {norm_mode!r}; expected one of: none, masked_std")

    current_5d, _ = _ensure_5d(current_state)
    morph_5d, _ = _ensure_5d(morph_qsm0)
    field_5d, _ = _ensure_5d(field_input)
    mask_5d, _ = _ensure_5d(mask)
    mask_5d = mask_5d.to(device=current_5d.device, dtype=current_5d.dtype)
    field_5d = field_5d.to(device=current_5d.device, dtype=current_5d.dtype)
    morph_5d = morph_5d.to(device=current_5d.device, dtype=current_5d.dtype) * mask_5d

    predicted_field_5d, _ = _ensure_5d(forward_field_from_chi(current_5d, dipole_kernel, mask=mask_5d))
    residual_5d = (predicted_field_5d - field_5d) * mask_5d
    abs_residual_5d = residual_5d.abs()
    residual_grad_5d, _ = _ensure_5d(compute_gradient_magnitude3d(residual_5d, mask=mask_5d))

    if cone_risk is None:
        cone_risk_5d = compute_cone_risk_map(
            current_5d,
            dipole_kernel=dipole_kernel,
            cone_threshold=cone_threshold,
            mask=mask_5d,
        )
    else:
        cone_risk_5d, _ = _ensure_5d(cone_risk)
        cone_risk_5d = cone_risk_5d.to(device=current_5d.device, dtype=current_5d.dtype) * mask_5d

    if norm_mode == "none":
        norm_current = current_5d
        norm_morph = morph_5d
        norm_residual = residual_5d
        norm_abs_residual = abs_residual_5d
        norm_cone = cone_risk_5d
        norm_residual_grad = residual_grad_5d
    else:
        norm_current = normalize_conditioning_channel(current_5d, mask_5d, mode="masked_std")
        norm_morph = normalize_conditioning_channel(morph_5d, mask_5d, mode="masked_std")
        norm_residual = normalize_conditioning_channel(residual_5d, mask_5d, mode="masked_std")
        norm_abs_residual = normalize_conditioning_channel(abs_residual_5d, mask_5d, mode="masked_std")
        norm_cone = normalize_conditioning_channel(cone_risk_5d, mask_5d, mode="masked_std")
        norm_residual_grad = normalize_conditioning_channel(residual_grad_5d, mask_5d, mode="masked_std")

    return torch.cat(
        [
            norm_current,
            norm_morph,
            norm_residual,
            norm_abs_residual,
            mask_5d,
            norm_cone,
            norm_residual_grad,
        ],
        dim=1,
    )


def build_conditioning_spec(
    mode: str,
    x_init: torch.Tensor,
    field_input: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    cone_threshold: float = 0.05,
    cone_risk: Optional[torch.Tensor] = None,
    morph_qsm0: Optional[torch.Tensor] = None,
    init_error_proxy_sigma: float = 1.0,
    norm_mode: str = "masked_std",
) -> Optional[ConditioningSpec]:
    mode = str(mode)
    if mode == "none":
        return None

    resolved_cone_risk = cone_risk
    if resolved_cone_risk is None:
        resolved_cone_risk = compute_cone_risk_map(
            x_init,
            dipole_kernel=dipole_kernel,
            cone_threshold=cone_threshold,
            mask=mask,
        )

    if mode == "morph_static_stem":
        if morph_qsm0 is None:
            raise RuntimeError("morph_static_stem requires morph_qsm0.")
        return ConditioningSpec(
            mode=mode,
            static_volume=build_morph_static_conditioning_volume(
                x_init=x_init,
                morph_qsm0=morph_qsm0,
                field_input=field_input,
                mask=mask,
                dipole_kernel=dipole_kernel,
                cone_threshold=cone_threshold,
                cone_risk=resolved_cone_risk,
                init_error_proxy_sigma=init_error_proxy_sigma,
                norm_mode=norm_mode,
            ),
            cone_risk=resolved_cone_risk,
            morph_qsm0=morph_qsm0,
            cone_threshold=float(cone_threshold),
            init_error_proxy_sigma=float(init_error_proxy_sigma),
            norm_mode=str(norm_mode),
        )

    if conditioning_mode_is_dynamic(mode):
        return ConditioningSpec(
            mode=mode,
            x_init=x_init,
            field_input=field_input,
            mask=mask,
            dipole_kernel=dipole_kernel,
            cone_risk=resolved_cone_risk,
            morph_qsm0=morph_qsm0,
            cone_threshold=float(cone_threshold),
            init_error_proxy_sigma=float(init_error_proxy_sigma),
            norm_mode=str(norm_mode),
        )

    return ConditioningSpec(
        mode=mode,
        static_volume=build_physics_aware_conditioning_volume(
            x_init=x_init,
            field_input=field_input,
            mask=mask,
            dipole_kernel=dipole_kernel,
            cone_threshold=cone_threshold,
            cone_risk=resolved_cone_risk,
            init_error_proxy_sigma=init_error_proxy_sigma,
            norm_mode=norm_mode,
        ),
        cone_risk=resolved_cone_risk,
        cone_threshold=float(cone_threshold),
        init_error_proxy_sigma=float(init_error_proxy_sigma),
        norm_mode=str(norm_mode),
    )


def materialize_conditioning_volume(
    conditioning: Optional[torch.Tensor | ConditioningSpec],
    current_state: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if conditioning is None or isinstance(conditioning, torch.Tensor):
        return conditioning
    if not isinstance(conditioning, ConditioningSpec):
        raise TypeError(
            "Expected conditioning to be None, a tensor, or ConditioningSpec; "
            f"got {type(conditioning)!r}."
        )
    return conditioning.build(current_state=current_state)


def forward_field_from_chi(
    chi: torch.Tensor,
    dipole_kernel: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    chi_5d, original_shape = _ensure_5d(chi)
    kernel = _expand_kernel(dipole_kernel, chi_5d.ndim)

    field = centered_ifftn(centered_fftn(_as_complex(chi_5d)) * kernel).real
    if mask is not None:
        mask_5d, _ = _ensure_5d(mask)
        field = field * mask_5d
    return _restore_shape(field, original_shape)


def adjoint_field_to_chi(
    field: torch.Tensor,
    dipole_kernel: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    field_5d, original_shape = _ensure_5d(field)
    if mask is not None:
        mask_5d, _ = _ensure_5d(mask)
        field_5d = field_5d * mask_5d

    kernel = _expand_kernel(dipole_kernel, field_5d.ndim)
    chi = centered_ifftn(centered_fftn(_as_complex(field_5d)) * kernel).real
    return _restore_shape(chi, original_shape)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    return diff.square().sum() / mask.sum().clamp_min(1e-8)


def pseudo_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 0.01,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = (pred - target) * mask
    scaled = diff / max(delta, 1e-12)
    loss = (delta**2) * (torch.sqrt(1.0 + scaled.square()) - 1.0)

    if weight is None:
        return loss.sum() / mask.sum().clamp_min(1e-8)

    weight = weight.view(-1, *([1] * (loss.ndim - 1)))
    weighted_loss = loss * weight
    weighted_mask = mask * weight
    return weighted_loss.sum() / weighted_mask.sum().clamp_min(1e-8)


def masked_gradient_pseudo_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 0.01,
) -> torch.Tensor:
    pred_5d, _ = _ensure_5d(pred)
    target_5d, _ = _ensure_5d(target)
    mask_5d, _ = _ensure_5d(mask)

    losses = []
    for axis in (-3, -2, -1):
        if pred_5d.shape[axis] <= 1:
            continue
        pred_grad = torch.diff(pred_5d, dim=axis)
        target_grad = torch.diff(target_5d, dim=axis)

        left_mask = mask_5d.narrow(axis, 0, mask_5d.shape[axis] - 1)
        right_mask = mask_5d.narrow(axis, 1, mask_5d.shape[axis] - 1)
        grad_mask = left_mask * right_mask
        losses.append(pseudo_huber_loss(pred_grad, target_grad, grad_mask, delta=delta))

    if not losses:
        return pred_5d.new_zeros(())
    return torch.stack(losses).mean()


def masked_frequency_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_5d, _ = _ensure_5d(pred)
    target_5d, _ = _ensure_5d(target)
    diff = pred_5d - target_5d
    if mask is not None:
        mask_5d, _ = _ensure_5d(mask)
        diff = diff * mask_5d

    diff_fft = centered_fftn(_as_complex(diff))
    power = diff_fft.real.square() + diff_fft.imag.square()
    if weight_map is not None:
        weight_fft = _expand_kernel(weight_map, power.ndim)
        power = power * weight_fft

    return power.mean()


def _build_residual_frequency_weight(
    source_power: torch.Tensor,
    power: float = 1.0,
    eps: float = 1e-8,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> torch.Tensor:
    if power <= 0:
        return torch.ones_like(source_power)

    residual_mag = source_power.clamp_min(eps).sqrt()
    mean_mag = residual_mag.detach().mean(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
    normalized = (residual_mag / mean_mag).clamp(min=min_scale, max=max_scale)
    return normalized.detach().pow(power)


def masked_hybrid_frequency_dipole_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dipole_kernel: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight_map: Optional[torch.Tensor] = None,
    chi_weight: float = 1.0,
    dipole_weight: float = 1.0,
    residual_power: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_5d, _ = _ensure_5d(pred)
    target_5d, _ = _ensure_5d(target)
    diff = pred_5d - target_5d
    if mask is not None:
        mask_5d, _ = _ensure_5d(mask)
        diff = diff * mask_5d

    diff_fft = centered_fftn(_as_complex(diff))
    chi_power = diff_fft.real.square() + diff_fft.imag.square()

    kernel = _expand_kernel(dipole_kernel, diff_fft.ndim)
    dipole_fft = diff_fft * kernel
    dipole_power = dipole_fft.real.square() + dipole_fft.imag.square()

    total_weight = torch.ones_like(chi_power)
    if weight_map is not None:
        total_weight = total_weight * _expand_kernel(weight_map, chi_power.ndim)

    residual_source = chi_power + dipole_power
    total_weight = total_weight * _build_residual_frequency_weight(
        residual_source,
        power=float(residual_power),
        eps=eps,
    )

    loss = torch.zeros((), device=chi_power.device, dtype=chi_power.dtype)
    if chi_weight > 0:
        loss = loss + float(chi_weight) * (chi_power * total_weight).mean()
    if dipole_weight > 0:
        loss = loss + float(dipole_weight) * (dipole_power * total_weight).mean()
    return loss


def sigma_weight(sigma: torch.Tensor, power: float = 0.5, clamp_min: float = 1e-3) -> torch.Tensor:
    return sigma.clamp_min(clamp_min).pow(-power)


def log_lerp(start: torch.Tensor, end: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.log(start.clamp_min(1e-8)) * (1.0 - ratio) + torch.log(end.clamp_min(1e-8)) * ratio)


def geometric_sigma_schedule(sigma_max: float, sigma_min: float, steps: int) -> list[float]:
    if steps < 2:
        return [float(sigma_max)]
    if sigma_max <= 0 or sigma_min <= 0:
        raise ValueError("sigma_max and sigma_min must be positive for a geometric schedule.")
    values = torch.logspace(
        torch.log10(torch.tensor(float(sigma_max))),
        torch.log10(torch.tensor(float(sigma_min))),
        steps=steps,
    )
    return [float(value.item()) for value in values]


def interpolate_sigma_schedule(
    sigma_start: torch.Tensor,
    sigma_end: torch.Tensor,
    num_steps: int,
) -> list[torch.Tensor]:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if num_steps == 1:
        return [sigma_start, sigma_end]

    values = [sigma_start]
    for step in range(1, num_steps):
        ratio = torch.full_like(sigma_start, step / num_steps)
        values.append(log_lerp(sigma_start, sigma_end, ratio))
    values.append(sigma_end)
    return values


def _expand_step_schedule(value, num_steps: int, cast, name: str) -> list:
    if num_steps < 1:
        raise ValueError(f"{name} schedule requires num_steps >= 1")

    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
        if not raw_values:
            raise ValueError(f"{name} schedule cannot be empty")
        values = [cast(part) for part in raw_values]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = [cast(part) for part in value]
    else:
        values = [cast(value)]

    if len(values) == 1:
        return values * num_steps
    if len(values) != num_steps:
        raise ValueError(f"{name} schedule length must be 1 or {num_steps}, got {len(values)}")
    return values


def build_dc_rollout_policy(
    num_steps: int,
    mode: str = "fixed",
    lam=0.1,
    num_cg_steps=8,
    tol: float = 1e-6,
    final_polish: bool = False,
    final_polish_lam: Optional[float] = None,
    final_polish_steps: Optional[int] = None,
) -> DCRolloutPolicy:
    if mode not in {"fixed", "stepwise_lambda", "stepwise_lambda+final_polish"}:
        raise ValueError(
            f"Unknown dc rollout mode {mode!r}. Expected one of: fixed, stepwise_lambda, stepwise_lambda+final_polish."
        )

    lam_schedule = _expand_step_schedule(lam, num_steps=num_steps, cast=float, name="lambda")
    cg_schedule = _expand_step_schedule(num_cg_steps, num_steps=num_steps, cast=int, name="cg_steps")
    steps = tuple(
        DCStepConfig(lam=float(step_lam), num_cg_steps=max(int(step_cg), 0), tol=float(tol), enabled=int(step_cg) > 0)
        for step_lam, step_cg in zip(lam_schedule, cg_schedule)
    )

    use_final_polish = final_polish or mode.endswith("+final_polish")
    final_cfg = None
    if use_final_polish:
        polish_lam = float(final_polish_lam if final_polish_lam is not None else lam_schedule[-1])
        polish_steps = int(final_polish_steps if final_polish_steps is not None else cg_schedule[-1])
        final_cfg = DCStepConfig(
            lam=polish_lam,
            num_cg_steps=max(polish_steps, 0),
            tol=float(tol),
            enabled=polish_steps > 0,
        )

    return DCRolloutPolicy(mode=mode, steps=steps, final_polish=final_cfg)


def consistency_denoise_step(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    sigmas: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    get_scalings_for_boundary_condition,
    use_amp: bool,
    step_index: Optional[int] = None,
    total_steps: Optional[int] = None,
    conditioning_volume: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    c_skip, c_out, c_in = get_scalings_for_boundary_condition(sigmas, sigma_data, sigma_min)
    c_skip = c_skip.view(-1, 1, 1, 1, 1)
    c_out = c_out.view(-1, 1, 1, 1, 1)
    c_in = c_in.view(-1, 1, 1, 1, 1)
    t = 250.0 * torch.log(sigmas.clamp_min(1e-6))
    device_type = "cuda" if x_t.is_cuda else "cpu"
    amp_enabled = bool(use_amp and x_t.is_cuda)
    context_model = model.module if hasattr(model, "module") else model

    if step_index is not None and total_steps is not None and hasattr(context_model, "set_rollout_context"):
        context_model.set_rollout_context(
            step_index=step_index,
            total_steps=total_steps,
            sigmas=sigmas,
            conditioning=conditioning_volume,
        )

    try:
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            pred = model(x_t * c_in, t)
            if pred.shape[1] == 2:
                pred = pred[:, :1]
            return c_skip * x_t + c_out * pred
    finally:
        if hasattr(context_model, "clear_rollout_context"):
            context_model.clear_rollout_context()


def sum_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).sum(dim=1)


def _broadcast_lam(lam, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lam_tensor = torch.as_tensor(lam, device=device, dtype=dtype)
    if lam_tensor.ndim == 0:
        return lam_tensor.view(1, 1, 1, 1, 1)
    if lam_tensor.ndim == 1 and lam_tensor.shape[0] == batch_size:
        return lam_tensor.view(batch_size, 1, 1, 1, 1)
    raise ValueError(f"lam must be a scalar or a batch-sized 1D tensor, got shape {tuple(lam_tensor.shape)}")


def data_consistency_cg(
    chi_init: torch.Tensor,
    field_gt: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    calci: Optional[torch.Tensor] = None,
    lam: float | torch.Tensor = 0.1,
    num_cg_steps: int = 8,
    tol: float = 1e-6,
) -> torch.Tensor:
    chi_5d, original_shape = _ensure_5d(chi_init)
    field_5d, _ = _ensure_5d(field_gt)
    mask_5d, _ = _ensure_5d(mask)
    batch_size = chi_5d.shape[0]
    lam_view = _broadcast_lam(lam, batch_size=batch_size, device=chi_init.device, dtype=chi_init.dtype)

    if calci is None:
        def A(x: torch.Tensor) -> torch.Tensor:
            return forward_field_from_chi(x, dipole_kernel, mask=mask_5d)

        def AT(y: torch.Tensor) -> torch.Tensor:
            return adjoint_field_to_chi(y, dipole_kernel, mask=mask_5d)
    else:
        calci_5d, _ = _ensure_5d(calci)
        kernel = _expand_kernel(dipole_kernel, chi_5d.ndim)

        def A(x: torch.Tensor) -> torch.Tensor:
            return calci_5d * centered_ifftn(centered_fftn(_as_complex(x)) * kernel).real

        def AT(y: torch.Tensor) -> torch.Tensor:
            return centered_ifftn(centered_fftn(_as_complex(calci_5d * y)) * kernel).real

    def normal_matrix(x: torch.Tensor) -> torch.Tensor:
        return AT(A(x)) + lam_view * x

    x = chi_5d.clone()
    rhs = AT(field_5d) + lam_view * chi_5d
    r = rhs - normal_matrix(x)
    p = r.clone()
    r_tr = sum_per_sample(r * r)

    for _ in range(max(int(num_cg_steps), 0)):
        active = r_tr > (tol * tol)
        if not torch.any(active):
            break

        ap = normal_matrix(p)
        p_ap = sum_per_sample(p * ap).clamp_min(1e-12)
        alpha = torch.zeros_like(r_tr)
        alpha[active] = r_tr[active] / p_ap[active]
        alpha_view = alpha.view(-1, 1, 1, 1, 1)

        x = x + alpha_view * p
        r = r - alpha_view * ap
        r_tr_new = sum_per_sample(r * r)

        if not torch.isfinite(r_tr_new).all() or not torch.isfinite(x).all():
            return chi_init.clone()

        beta = torch.zeros_like(r_tr)
        valid = active & (r_tr > 1e-12)
        beta[valid] = r_tr_new[valid] / r_tr[valid].clamp_min(1e-12)
        p = r + beta.view(-1, 1, 1, 1, 1) * p
        r_tr = r_tr_new

    result = torch.where(torch.isfinite(x), x, chi_5d)
    return _restore_shape(result, original_shape)


def apply_dc_step(
    chi_prior: torch.Tensor,
    field_gt: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    config: DCStepConfig,
    calci: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not config.enabled or config.num_cg_steps <= 0:
        return chi_prior
    return data_consistency_cg(
        chi_init=chi_prior,
        field_gt=field_gt,
        mask=mask,
        dipole_kernel=dipole_kernel,
        calci=calci,
        lam=config.lam,
        num_cg_steps=config.num_cg_steps,
        tol=config.tol,
    )


def apply_final_dc_polish(
    chi: torch.Tensor,
    field_gt: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    policy: DCRolloutPolicy,
    calci: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if policy.final_polish is None:
        return chi
    return apply_dc_step(chi, field_gt, mask, dipole_kernel, policy.final_polish, calci=calci)


def run_refinement_rollout(
    model: torch.nn.Module,
    x_start: torch.Tensor,
    sigma_schedule: Sequence[torch.Tensor],
    field_gt: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    dc_policy: DCRolloutPolicy,
    sigma_data: float,
    sigma_min: float,
    get_scalings_for_boundary_condition,
    use_amp: bool,
    calci: Optional[torch.Tensor] = None,
    conditioning_volume: Optional[torch.Tensor | ConditioningSpec] = None,
    collect_states: bool = True,
    collect_priors: bool = False,
) -> RefinementRolloutResult:
    num_steps = len(sigma_schedule) - 1
    dc_policy.validate(num_steps)

    current = x_start
    states = [current] if collect_states else []
    priors: list[torch.Tensor] = []

    for step_index in range(num_steps):
        step_conditioning = materialize_conditioning_volume(conditioning_volume, current_state=current)
        prior = consistency_denoise_step(
            model=model,
            x_t=current,
            sigmas=sigma_schedule[step_index],
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            get_scalings_for_boundary_condition=get_scalings_for_boundary_condition,
            use_amp=use_amp,
            step_index=step_index,
            total_steps=num_steps,
            conditioning_volume=step_conditioning,
        )
        if collect_priors:
            priors.append(prior)

        current = apply_dc_step(
            chi_prior=prior,
            field_gt=field_gt,
            mask=mask,
            dipole_kernel=dipole_kernel,
            config=dc_policy.step_config(step_index),
            calci=calci,
        )
        if collect_states:
            states.append(current)

    if dc_policy.final_polish is not None:
        current = apply_final_dc_polish(current, field_gt, mask, dipole_kernel, dc_policy, calci=calci)
        if collect_states:
            states[-1] = current

    return RefinementRolloutResult(
        final_state=current,
        states=states,
        priors=priors,
        sigma_schedule=list(sigma_schedule),
    )


def run_residual_delta_refinement_rollout(
    model: torch.nn.Module,
    residual_base: torch.Tensor,
    residual_scale: float,
    sigma_schedule: Sequence[torch.Tensor],
    field_gt: torch.Tensor,
    mask: torch.Tensor,
    dipole_kernel: torch.Tensor,
    dc_policy: DCRolloutPolicy,
    sigma_data: float,
    sigma_min: float,
    get_scalings_for_boundary_condition,
    use_amp: bool,
    calci: Optional[torch.Tensor] = None,
    conditioning_volume: Optional[torch.Tensor | ConditioningSpec] = None,
    collect_states: bool = True,
    collect_priors: bool = False,
) -> RefinementRolloutResult:
    num_steps = len(sigma_schedule) - 1
    dc_policy.validate(num_steps)
    residual_scale = float(residual_scale)
    if residual_scale <= 0:
        raise ValueError(f"residual_scale must be positive, got {residual_scale}")

    base_5d, _ = _ensure_5d(residual_base)
    current_delta = torch.zeros_like(base_5d)
    current_chi = base_5d
    states = [current_chi] if collect_states else []
    priors: list[torch.Tensor] = []

    for step_index in range(num_steps):
        step_conditioning = materialize_conditioning_volume(conditioning_volume, current_state=current_chi)
        prior_delta = consistency_denoise_step(
            model=model,
            x_t=current_delta,
            sigmas=sigma_schedule[step_index],
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            get_scalings_for_boundary_condition=get_scalings_for_boundary_condition,
            use_amp=use_amp,
            step_index=step_index,
            total_steps=num_steps,
            conditioning_volume=step_conditioning,
        )
        prior_chi = base_5d + residual_scale * prior_delta
        if collect_priors:
            priors.append(prior_chi)

        current_chi = apply_dc_step(
            chi_prior=prior_chi,
            field_gt=field_gt,
            mask=mask,
            dipole_kernel=dipole_kernel,
            config=dc_policy.step_config(step_index),
            calci=calci,
        )
        current_delta = (current_chi - base_5d) / residual_scale
        if collect_states:
            states.append(current_chi)

    if dc_policy.final_polish is not None:
        current_chi = apply_final_dc_polish(current_chi, field_gt, mask, dipole_kernel, dc_policy, calci=calci)
        if collect_states:
            states[-1] = current_chi

    return RefinementRolloutResult(
        final_state=current_chi,
        states=states,
        priors=priors,
        sigma_schedule=list(sigma_schedule),
    )


def select_teacher_state_indices(num_available_states: int, num_requested_states: int) -> tuple[int, ...]:
    if num_available_states < 2:
        raise ValueError(f"Teacher state list must contain at least 2 states, got {num_available_states}")
    if num_requested_states < 1:
        raise ValueError(f"num_requested_states must be >= 1, got {num_requested_states}")

    max_selectable = num_available_states - 1
    num_requested_states = min(num_requested_states, max_selectable)

    if num_requested_states == 1:
        return (num_available_states - 1,)

    values = torch.linspace(1, num_available_states - 1, steps=num_requested_states)
    indices = values.round().to(dtype=torch.int64).tolist()
    indices = [max(1, min(num_available_states - 1, int(index))) for index in indices]
    indices[-1] = num_available_states - 1

    deduped: list[int] = []
    for index in indices:
        if not deduped or index != deduped[-1]:
            deduped.append(index)

    while len(deduped) < num_requested_states:
        for index in range(1, num_available_states):
            if index not in deduped:
                deduped.append(index)
            if len(deduped) == num_requested_states:
                break
        deduped.sort()
        if len(deduped) == max_selectable:
            break

    return tuple(deduped[:num_requested_states])


def select_teacher_indices_from_steps(
    teacher_steps: Sequence[int],
    num_requested_states: int,
) -> tuple[int, ...]:
    if len(teacher_steps) < 2:
        raise ValueError(f"Teacher step list must contain at least 2 entries, got {len(teacher_steps)}")
    if num_requested_states < 1:
        raise ValueError(f"num_requested_states must be >= 1, got {num_requested_states}")

    step_values = [int(step) for step in teacher_steps]
    if step_values[0] != 0:
        raise ValueError(f"Teacher step list must start at 0 for x_init alignment, got {step_values[0]}")

    for prev, cur in zip(step_values[:-1], step_values[1:]):
        if cur <= prev:
            raise ValueError(f"Teacher steps must be strictly increasing, got {step_values}")

    candidate_indices = list(range(1, len(step_values)))
    candidate_steps = step_values[1:]
    if num_requested_states >= len(candidate_indices):
        return tuple(candidate_indices)
    if num_requested_states == 1:
        return (candidate_indices[-1],)

    target_steps = torch.linspace(
        float(candidate_steps[0]),
        float(candidate_steps[-1]),
        steps=num_requested_states,
    ).tolist()

    selected: list[int] = []
    search_start = 0
    for target_pos, target_step in enumerate(target_steps):
        remaining_targets = num_requested_states - target_pos
        max_search_index = len(candidate_indices) - remaining_targets
        best_local_index = min(
            range(search_start, max_search_index + 1),
            key=lambda idx: abs(candidate_steps[idx] - target_step),
        )
        selected.append(candidate_indices[best_local_index])
        search_start = best_local_index + 1

    selected[-1] = candidate_indices[-1]
    return tuple(selected)


def extract_teacher_targets(
    teacher_states: Sequence[torch.Tensor],
    num_student_steps: int,
    mode: str = "trajectory",
    teacher_steps: Optional[Sequence[int]] = None,
) -> dict[str, object]:
    if mode not in {"endpoint", "trajectory", "trajectory+endpoint"}:
        raise ValueError(
            f"Unknown teacher target mode {mode!r}. Expected one of: endpoint, trajectory, trajectory+endpoint."
        )

    if not teacher_states:
        raise ValueError("teacher_states cannot be empty")

    endpoint = teacher_states[-1]
    if mode == "endpoint":
        return {"endpoint": endpoint, "trajectory": []}

    if teacher_steps is None:
        indices = select_teacher_state_indices(len(teacher_states), num_requested_states=num_student_steps)
    else:
        if len(teacher_states) != len(teacher_steps):
            raise ValueError(
                f"teacher_states/teacher_steps length mismatch: {len(teacher_states)} vs {len(teacher_steps)}"
            )
        indices = select_teacher_indices_from_steps(teacher_steps, num_requested_states=num_student_steps)
    trajectory = [teacher_states[index] for index in indices]
    return {"endpoint": endpoint, "trajectory": trajectory}


def _candidate_roots(script_path: Optional[Path], explicit_root: Optional[str | os.PathLike[str]], env_var: str):
    seen: set[Path] = set()
    for raw in (explicit_root, os.environ.get(env_var)):
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            yield path

    if script_path is not None:
        for parent in [script_path.parent] + list(script_path.parents):
            resolved = parent.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved

    cwd = Path.cwd().resolve()
    if cwd not in seen:
        seen.add(cwd)
        yield cwd
    for parent in cwd.parents:
        resolved = parent.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _find_root_with_child(
    child_name: str,
    script_path: Optional[Path] = None,
    explicit_root: Optional[str | os.PathLike[str]] = None,
    env_var: str = "",
) -> Optional[Path]:
    for candidate in _candidate_roots(script_path, explicit_root, env_var):
        if (candidate / child_name).exists():
            return candidate
    return None


def configure_import_paths(
    project_root: Optional[str | os.PathLike[str]] = None,
    diffusion_root: Optional[str | os.PathLike[str]] = None,
    script_path: Optional[str | os.PathLike[str]] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    script_path_obj = Path(script_path).resolve() if script_path else None
    resolved_project_root = _find_root_with_child(
        "qsm_cm",
        script_path=script_path_obj,
        explicit_root=project_root,
        env_var="QSM_PROJECT_ROOT",
    )
    resolved_diffusion_root = _find_root_with_child(
        "guided_diffusion",
        script_path=script_path_obj,
        explicit_root=diffusion_root,
        env_var="QSM_DIFFUSION_ROOT",
    )

    for root in (resolved_project_root, resolved_diffusion_root):
        if root is None:
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    return resolved_project_root, resolved_diffusion_root


def load_student_components(
    project_root: Optional[str | os.PathLike[str]] = None,
    diffusion_root: Optional[str | os.PathLike[str]] = None,
    script_path: Optional[str | os.PathLike[str]] = None,
):
    configure_import_paths(project_root=project_root, diffusion_root=diffusion_root, script_path=script_path)
    try:
        from qsm_cm.consistency_distill_ddp import get_scalings_for_boundary_condition
        from qsm_cm.student_model_3d import StudentUNetConfig, build_student_unet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Unable to import qsm_cm modules. Pass --project-root or set QSM_PROJECT_ROOT "
            "to the repository that contains the qsm_cm package."
        ) from exc

    return build_student_unet, StudentUNetConfig, get_scalings_for_boundary_condition
