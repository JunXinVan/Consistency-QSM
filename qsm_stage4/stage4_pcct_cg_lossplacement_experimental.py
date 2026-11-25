#!/usr/bin/env python3
"""
Stage 4 loss-placement migration experiment.

This entrypoint keeps the canonical Stage 4 engineering path intact while
moving the primary supervision targets to the locations discussed in the audit:

- GT loss -> final post-DC state
- CT loss -> all prior states
- Physics loss -> final prior
- Spectral loss -> final prior

The stable mainline script remains untouched; this file is an isolated
experiment entrypoint for migration validation.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import stage4_pcct_cg as base
from qsm_bank_utils import adjoint_field_to_chi, create_dipole_kernel, data_consistency_cg
from qsm_morph import build_morph_qsm0, tkd_inversion


LOSS_PLACEMENT_SUMMARY = {
    "gt": "final_post_dc",
    "gradient": "final_post_dc",
    "physics": "final_prior",
    "ct": "all_priors_sigma_aligned",
    "spectral": "final_prior",
    "teacher_aux": "final_post_dc_unchanged",
}


def current_loss_placement_summary(args: argparse.Namespace) -> dict:
    summary = dict(LOSS_PLACEMENT_SUMMARY)
    summary["gt"] = str(args.gt_loss_target)
    if not bool(args.use_teacher_aux) or float(args.teacher_aux_weight) <= 0:
        summary["teacher_aux"] = "off"
    return summary


class MacroDCVolumeSampler:
    def __init__(
        self,
        source_root: str | Path,
        macro_shape: tuple[int, int, int],
        mask_threshold: float = 1e-3,
        cache_size: int = 2,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.macro_shape = tuple(int(v) for v in macro_shape)
        self.mask_threshold = float(mask_threshold)
        self.cache_size = max(int(cache_size), 1)
        self.volume_paths = sorted(self.source_root.glob("*/cosmos*.nii*"))
        if not self.volume_paths:
            raise FileNotFoundError(f"No cosmos*.nii* volumes found under {self.source_root}")
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _load_volume(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(path.resolve())
        if key in self._cache:
            payload = self._cache.pop(key)
            self._cache[key] = payload
            return payload
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("nibabel is required for --macro-dc-align-weight > 0.") from exc
        chi = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        if chi.ndim != 3:
            raise ValueError(f"Expected 3D macro DC source volume, got {path} with shape={chi.shape}")
        mask = (np.abs(chi) > self.mask_threshold).astype(np.float32)
        payload = (chi, mask)
        self._cache[key] = payload
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def sample(self, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(64):
            path = self.volume_paths[rng.randrange(len(self.volume_paths))]
            chi, mask = self._load_volume(path)
            if any(dim < size for dim, size in zip(chi.shape, self.macro_shape)):
                continue
            origin = [
                rng.randrange(int(dim - size) + 1) if int(dim - size) > 0 else 0
                for dim, size in zip(chi.shape, self.macro_shape)
            ]
            d0, h0, w0 = origin
            md, mh, mw = self.macro_shape
            chi_macro = np.array(chi[d0 : d0 + md, h0 : h0 + mh, w0 : w0 + mw], copy=True)
            mask_macro = np.array(mask[d0 : d0 + md, h0 : h0 + mh, w0 : w0 + mw], copy=True)
            if float(mask_macro.sum()) > 1024.0:
                return chi_macro, mask_macro
        raise RuntimeError(f"Could not sample a valid macro DC crop from {self.source_root}")


class LossPlacementMigrationTrainer(base.Stage4SpecializationTrainer):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.extra_training_metric_keys = ("macro_dc_loss", "macro_dc_contrib", "macro_dc_applied")
        self.macro_dc_sampler = None
        self.macro_dc_rng = random.Random(int(args.seed) + 7919 * max(int(self.rank), 0))
        if float(self.args.macro_dc_align_weight) > 0:
            if self.args.morph_prior_mode != "none":
                raise RuntimeError("macro DC alignment currently expects morph_prior_mode=none.")
            self.macro_dc_sampler = MacroDCVolumeSampler(
                source_root=self.args.macro_dc_source_root,
                macro_shape=self.args.macro_dc_shape,
            )
            self.macro_dc_patch_kernel = create_dipole_kernel(
                self.args.macro_dc_patch_shape,
                voxel_size=self.args.voxel_size,
                b0_dir=self.args.b0_dir,
                device=self.device,
                dtype=torch.float32,
            )
            self.macro_dc_volume_kernel = create_dipole_kernel(
                self.args.macro_dc_shape,
                voxel_size=self.args.voxel_size,
                b0_dir=self.args.b0_dir,
                device=self.device,
                dtype=torch.float32,
            )

    def build_effective_config(self) -> dict:
        payload = super().build_effective_config()
        payload["entrypoint"] = str(Path(__file__).resolve())
        payload["loss_placement_experiment"] = current_loss_placement_summary(self.args)
        payload["loss_placement_experiment_name"] = "full_migration_v1"
        payload["morph_prior_mode"] = self.args.morph_prior_mode
        payload["morph_residual_scale"] = float(self.args.morph_residual_scale)
        payload["two_step_role_mode"] = self.args.two_step_role_mode
        payload["two_step_role_low_weight"] = float(self.args.two_step_role_low_weight)
        payload["two_step_role_detail_weight"] = float(self.args.two_step_role_detail_weight)
        payload["two_step_role_lowpass_kernel"] = int(self.args.two_step_role_lowpass_kernel)
        payload["loss_morph_anchor_weight"] = float(self.args.loss_morph_anchor_weight)
        payload["loss_field_morph_weight"] = float(self.args.loss_field_morph_weight)
        payload["morph_anchor_scale"] = float(self.args.morph_anchor_scale)
        payload["morph_anchor_power"] = float(self.args.morph_anchor_power)
        payload["macro_dc_align_weight"] = float(self.args.macro_dc_align_weight)
        payload["macro_dc_every"] = int(self.args.macro_dc_every)
        payload["macro_dc_shape"] = list(self.args.macro_dc_shape)
        payload["macro_dc_patch_shape"] = list(self.args.macro_dc_patch_shape)
        payload["macro_dc_stride"] = list(self.args.macro_dc_stride)
        payload["macro_dc_lam"] = float(self.args.macro_dc_lam)
        payload["macro_dc_cg_steps"] = int(self.args.macro_dc_cg_steps)
        payload["macro_dc_source_root"] = str(self.args.macro_dc_source_root)
        payload["x_init_override_mode"] = str(self.args.x_init_override_mode)
        payload["x_init_tkd_threshold"] = float(self.args.x_init_tkd_threshold)
        payload["x_init_qsm0_blend_weight"] = float(self.args.x_init_qsm0_blend_weight)
        payload["gt_loss_target"] = str(self.args.gt_loss_target)
        return payload

    def build_rollout_x_init(
        self,
        cache_x_init: torch.Tensor,
        field_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(self.args.x_init_override_mode)
        if mode == "none":
            return cache_x_init
        if mode == "pure_noise":
            with torch.no_grad():
                valid = mask.to(device=cache_x_init.device, dtype=cache_x_init.dtype)
                noise = torch.randn_like(cache_x_init) * float(self.sigma_max) * valid
            if int(self.rank) == 0 and not getattr(self, "_pure_noise_start_logged", False):
                self._pure_noise_start_logged = True
                masked_rms = noise.flatten(1).pow(2).sum(dim=1).div(
                    valid.flatten(1).sum(dim=1).clamp_min(1.0)
                ).sqrt()
                rms_text = ", ".join(f"{float(v):.4f}" for v in masked_rms.cpu())
                print(
                    f"[x_init_override=pure_noise] rollout start = masked N(0, sigma_max^2), "
                    f"sigma_max={float(self.sigma_max):.4f}, per-sample masked RMS=[{rms_text}]",
                    flush=True,
                )
            return noise
        if mode == "blend_projected_qsm0":
            with torch.no_grad():
                qsm0 = build_morph_qsm0(
                    field=field_gt,
                    dipole_kernel=self.dipole_kernel,
                    mask=mask,
                    tkd_low=0.10,
                    tkd_high=0.20,
                    cone_threshold=0.10,
                    cone_power=0.20,
                    centered_fft=False,
                ).to(device=cache_x_init.device, dtype=cache_x_init.dtype)
                valid = mask.to(device=cache_x_init.device, dtype=cache_x_init.dtype)
                numerator = (qsm0 * cache_x_init * valid).flatten(1).sum(dim=1)
                denominator = (qsm0.square() * valid).flatten(1).sum(dim=1).clamp_min(1e-12)
                scale = (numerator / denominator).view(-1, 1, 1, 1, 1)
                projected = qsm0 * scale * valid
                weight = float(self.args.x_init_qsm0_blend_weight)
                blended = ((1.0 - weight) * cache_x_init + weight * projected) * valid
            return blended
        if mode != "tkd020_scaled":
            raise RuntimeError(f"Unsupported x_init_override_mode={mode}")

        # Match TKD's scale to the adjoint cache initialization per sample.
        # This keeps the rollout in the model's expected magnitude range while
        # testing whether TKD's structure is a better starting state.
        with torch.no_grad():
            tkd = tkd_inversion(
                field_gt,
                self.dipole_kernel,
                threshold=float(self.args.x_init_tkd_threshold),
                mask=mask,
                centered_fft=True,
            ).to(device=cache_x_init.device, dtype=cache_x_init.dtype)
            valid = mask.to(device=cache_x_init.device, dtype=cache_x_init.dtype)
            numerator = (tkd * cache_x_init * valid).flatten(1).sum(dim=1)
            denominator = (tkd.square() * valid).flatten(1).sum(dim=1).clamp_min(1e-12)
            scale = (numerator / denominator).view(-1, 1, 1, 1, 1)
            scaled = tkd * scale * valid
        return scaled

    def build_morph_prior(self, field_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if self.args.morph_prior_mode == "none":
            return None
        with torch.no_grad():
            qsm0 = build_morph_qsm0(
                field=field_gt,
                dipole_kernel=self.dipole_kernel,
                mask=mask,
                tkd_low=self.args.morph_tkd_low,
                tkd_high=self.args.morph_tkd_high,
                cone_threshold=self.args.morph_cone_threshold,
                cone_power=self.args.morph_cone_power,
                centered_fft=True,
            )
        return qsm0.to(device=field_gt.device, dtype=field_gt.dtype)

    def build_conditioning_volume(
        self,
        x_init: torch.Tensor,
        field_gt: torch.Tensor,
        mask: torch.Tensor,
        cone_risk: torch.Tensor | None = None,
        morph_qsm0: torch.Tensor | None = None,
    ):
        if not self.conditioning_active:
            return None
        return base.build_conditioning_spec(
            mode=self.effective_conditioning_mode,
            x_init=x_init,
            field_input=field_gt,
            mask=mask,
            dipole_kernel=self.dipole_kernel,
            cone_threshold=self.effective_conditioning_cone_threshold,
            cone_risk=cone_risk,
            morph_qsm0=morph_qsm0,
            init_error_proxy_sigma=self.effective_init_error_proxy_sigma,
            norm_mode=self.effective_conditioning_norm_mode,
        )

    def two_step_role_enabled(self) -> bool:
        return bool(
            self.args.two_step_role_mode != "none"
            and (
                float(self.args.two_step_role_low_weight) > 0
                or float(self.args.two_step_role_detail_weight) > 0
            )
        )

    @staticmethod
    def _ensure_5d(volume: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        original_shape = tuple(volume.shape)
        if volume.ndim == 5:
            return volume, original_shape
        if volume.ndim == 4:
            return volume.unsqueeze(1), original_shape
        if volume.ndim == 3:
            return volume.unsqueeze(0).unsqueeze(0), original_shape
        raise ValueError(f"Expected a 3D, 4D, or 5D tensor, got shape={original_shape}")

    @staticmethod
    def _restore_shape(volume: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
        if len(original_shape) == 5:
            return volume
        if len(original_shape) == 4:
            return volume.squeeze(1)
        if len(original_shape) == 3:
            return volume.squeeze(0).squeeze(0)
        raise ValueError(f"Cannot restore unsupported original shape={original_shape}")

    def masked_box_lowpass(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kernel_size = int(self.args.two_step_role_lowpass_kernel)
        if kernel_size <= 1:
            return volume * mask
        if kernel_size % 2 == 0:
            raise ValueError(f"two_step_role_lowpass_kernel must be odd, got {kernel_size}")

        volume_5d, original_shape = self._ensure_5d(volume)
        mask_5d, _ = self._ensure_5d(mask)
        mask_5d = mask_5d.to(device=volume_5d.device, dtype=volume_5d.dtype)
        weight = torch.ones(
            (1, 1, kernel_size, kernel_size, kernel_size),
            device=volume_5d.device,
            dtype=volume_5d.dtype,
        )
        padding = kernel_size // 2
        numerator = F.conv3d(volume_5d * mask_5d, weight, padding=padding)
        denominator = F.conv3d(mask_5d, weight, padding=padding).clamp_min(1e-6)
        lowpass = (numerator / denominator) * mask_5d
        return self._restore_shape(lowpass, original_shape)

    def morph_self_supervision_enabled(self) -> bool:
        return bool(
            float(self.args.loss_morph_anchor_weight) > 0
            or float(self.args.loss_field_morph_weight) > 0
        )

    def sobel3d(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        volume_5d, _ = self._ensure_5d(volume)
        dtype = volume_5d.dtype
        device = volume_5d.device
        kx = torch.tensor(
            [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[1, 2, 1], [2, 4, 2], [1, 2, 1]]],
            device=device,
            dtype=dtype,
        )
        ky = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            device=device,
            dtype=dtype,
        )
        kz = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
             [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]],
            device=device,
            dtype=dtype,
        )
        kernels = [kernel.view(1, 1, 3, 3, 3) for kernel in (kx, ky, kz)]
        return tuple(F.conv3d(volume_5d, kernel, padding=1) for kernel in kernels)

    def masked_abs_standardize(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values_5d, _ = self._ensure_5d(values)
        mask_5d, _ = self._ensure_5d(mask)
        mask_5d = mask_5d.to(device=values_5d.device, dtype=values_5d.dtype)
        count = mask_5d.flatten(1).sum(dim=1).clamp_min(1.0)
        flat_values = values_5d.flatten(1)
        flat_mask = mask_5d.flatten(1)
        mean = (flat_values * flat_mask).sum(dim=1) / count
        centered = (flat_values - mean[:, None]) * flat_mask
        var = centered.square().sum(dim=1) / count
        std = var.clamp_min(float(self.args.field_morph_eps)).sqrt()
        normalized = (values_5d - mean.view(-1, 1, 1, 1, 1)) / std.view(-1, 1, 1, 1, 1)
        return normalized.abs() * mask_5d

    def compute_morph_anchor_loss(
        self,
        pred: torch.Tensor,
        morph_qsm0: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if float(self.args.loss_morph_anchor_weight) <= 0:
            return self.zero_loss_anchor()
        if morph_qsm0 is None:
            raise RuntimeError("loss_morph_anchor_weight > 0 requires morph_qsm0.")
        mask_5d, _ = self._ensure_5d(mask)
        qsm0_5d, _ = self._ensure_5d(morph_qsm0)
        pred_5d, _ = self._ensure_5d(pred)
        mask_5d = mask_5d.to(device=pred_5d.device, dtype=pred_5d.dtype)
        qsm0_5d = qsm0_5d.to(device=pred_5d.device, dtype=pred_5d.dtype)
        qsm0_weight = qsm0_5d.abs().clamp_min(0.0).pow(float(self.args.morph_anchor_power))
        weighted_diff = float(self.args.morph_anchor_scale) * qsm0_weight * (pred_5d - qsm0_5d).abs() * mask_5d
        return weighted_diff.sum() / mask_5d.sum().clamp_min(1e-8)

    def compute_field_morphology_loss(
        self,
        pred: torch.Tensor,
        field_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if float(self.args.loss_field_morph_weight) <= 0:
            return self.zero_loss_anchor()
        pred_5d, _ = self._ensure_5d(pred)
        field_5d, _ = self._ensure_5d(field_gt)
        mask_5d, _ = self._ensure_5d(mask)
        mask_5d = mask_5d.to(device=pred_5d.device, dtype=pred_5d.dtype)
        pred_grads = self.sobel3d(pred_5d * mask_5d)
        field_grads = self.sobel3d(field_5d.to(device=pred_5d.device, dtype=pred_5d.dtype) * mask_5d)
        losses = []
        for pred_grad, field_grad in zip(pred_grads, field_grads):
            pred_norm = self.masked_abs_standardize(pred_grad, mask_5d)
            field_norm = self.masked_abs_standardize(field_grad, mask_5d)
            losses.append((pred_norm - field_norm).abs().sum() / mask_5d.sum().clamp_min(1e-8))
        return torch.stack(losses).mean()

    def compute_two_step_role_loss(
        self,
        student_rollout,
        chi_gt: torch.Tensor,
        mask: torch.Tensor,
        train_nfe: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.two_step_role_enabled():
            zero = self.zero_loss_anchor()
            return zero, zero, zero
        if self.args.morph_prior_mode != "residual_delta":
            raise RuntimeError("two_step_role_mode is currently only defined for morph_prior_mode=residual_delta.")
        if int(train_nfe) != 2:
            raise RuntimeError(f"two_step_role_mode requires train_nfe=2, got {train_nfe}.")
        if len(student_rollout.states) < 3:
            raise RuntimeError("two_step_role_mode requires rollout states; rerun with collect_states=True.")

        low_target = self.masked_box_lowpass(chi_gt, mask)
        step1_state = student_rollout.states[1]
        final_state = student_rollout.final_state

        low_loss = self.zero_loss_anchor()
        if float(self.args.two_step_role_low_weight) > 0:
            low_loss = base.pseudo_huber_loss(
                step1_state,
                low_target,
                mask,
                delta=self.args.two_step_role_delta,
            )

        detail_loss = self.zero_loss_anchor()
        if float(self.args.two_step_role_detail_weight) > 0:
            final_low = self.masked_box_lowpass(final_state, mask)
            pred_detail = (final_state - final_low) * mask
            target_detail = (chi_gt - low_target) * mask
            detail_loss = base.pseudo_huber_loss(
                pred_detail,
                target_detail,
                mask,
                delta=self.args.two_step_role_delta,
            )

        role_loss = (
            float(self.args.two_step_role_low_weight) * low_loss
            + float(self.args.two_step_role_detail_weight) * detail_loss
        )
        return role_loss, low_loss, detail_loss

    def extra_training_info_keys(self) -> tuple[str, ...]:
        return ("macro_dc_loss", "macro_dc_contrib", "macro_dc_applied")

    def macro_dc_should_apply(self) -> bool:
        if self.macro_dc_sampler is None or float(self.args.macro_dc_align_weight) <= 0:
            return False
        every = max(int(self.args.macro_dc_every), 1)
        return int(self.global_optimizer_step) % every == 0

    def macro_patch_origins(self) -> list[tuple[int, int, int]]:
        starts_per_axis = []
        for macro, patch, stride in zip(
            self.args.macro_dc_shape,
            self.args.macro_dc_patch_shape,
            self.args.macro_dc_stride,
        ):
            if patch > macro:
                raise ValueError(f"macro_dc_patch_shape {self.args.macro_dc_patch_shape} exceeds {self.args.macro_dc_shape}")
            starts = list(range(0, macro - patch + 1, stride))
            tail = macro - patch
            if starts[-1] != tail:
                starts.append(tail)
            starts_per_axis.append(starts)
        return [(d, h, w) for d in starts_per_axis[0] for h in starts_per_axis[1] for w in starts_per_axis[2]]

    @staticmethod
    def crop_5d(volume: torch.Tensor, origin: tuple[int, int, int], shape: tuple[int, int, int]) -> torch.Tensor:
        d0, h0, w0 = origin
        pd, ph, pw = shape
        return volume[..., d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]

    def stitch_macro_patches(
        self,
        patches: list[torch.Tensor],
        origins: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        if len(patches) != len(origins):
            raise ValueError(f"Expected equal patches and origins, got {len(patches)} vs {len(origins)}")
        md, mh, mw = self.args.macro_dc_shape
        pd, ph, pw = self.args.macro_dc_patch_shape
        stitched = None
        weight = None
        for patch, (d0, h0, w0) in zip(patches, origins):
            pad = (w0, mw - w0 - pw, h0, mh - h0 - ph, d0, md - d0 - pd)
            padded = F.pad(patch, pad)
            padded_weight = F.pad(torch.ones_like(patch), pad)
            stitched = padded if stitched is None else stitched + padded
            weight = padded_weight if weight is None else weight + padded_weight
        if stitched is None or weight is None:
            raise RuntimeError("macro DC stitching received no patches")
        return stitched / weight.clamp_min(1.0)

    def run_macro_patch(self, x_init: torch.Tensor, field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        conditioning_volume = self.build_conditioning_volume(
            x_init,
            field,
            mask,
            cone_risk=None,
            morph_qsm0=None,
        )
        rollout = self.run_rollout_with_model(
            model=self.model,
            x_init=x_init,
            field_gt=field,
            mask=mask,
            conditioning_volume=conditioning_volume,
            num_steps=self.choose_train_nfe(),
            infer=False,
            collect_states=False,
        )
        return rollout.final_state

    def compute_macro_dc_alignment_loss(self) -> tuple[torch.Tensor, float]:
        if not self.macro_dc_should_apply():
            return self.zero_loss_anchor(), 0.0
        if self.macro_dc_sampler is None:
            return self.zero_loss_anchor(), 0.0

        chi_np, mask_np = self.macro_dc_sampler.sample(self.macro_dc_rng)
        chi_macro = torch.from_numpy(chi_np).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask_macro = torch.from_numpy(mask_np).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            field_macro = base.forward_field_from_chi(
                chi_macro,
                self.macro_dc_volume_kernel,
                mask=mask_macro,
            )
            x_init_macro = adjoint_field_to_chi(
                field_macro,
                self.macro_dc_volume_kernel,
                mask=mask_macro,
            )

        origins = self.macro_patch_origins()
        pred_patches = []
        for origin in origins:
            x_patch = self.crop_5d(x_init_macro, origin, self.args.macro_dc_patch_shape).contiguous()
            field_patch = self.crop_5d(field_macro, origin, self.args.macro_dc_patch_shape).contiguous()
            mask_patch = self.crop_5d(mask_macro, origin, self.args.macro_dc_patch_shape).contiguous()
            pred_patch = checkpoint(
                self.run_macro_patch,
                x_patch,
                field_patch,
                mask_patch,
                use_reentrant=False,
            )
            pred_patches.append(pred_patch)

        stitched = self.stitch_macro_patches(pred_patches, origins) * mask_macro
        post_dc = data_consistency_cg(
            chi_init=stitched,
            field_gt=field_macro,
            mask=mask_macro,
            dipole_kernel=self.macro_dc_volume_kernel,
            lam=float(self.args.macro_dc_lam),
            num_cg_steps=int(self.args.macro_dc_cg_steps),
        )
        return base.pseudo_huber_loss(post_dc, chi_macro, mask_macro, delta=self.args.pseudo_huber_delta), 1.0

    def run_rollout_with_model(
        self,
        model: torch.nn.Module,
        x_init: torch.Tensor,
        field_gt: torch.Tensor,
        mask: torch.Tensor,
        conditioning_volume: torch.Tensor | None,
        num_steps: int,
        infer: bool,
        collect_states: bool,
    ):
        sigma_schedule = self.build_sigma_schedule(batch_size=x_init.shape[0], num_steps=num_steps)
        dc_policy = self.build_dc_policy(num_steps=num_steps, infer=infer)
        if self.args.morph_prior_mode == "residual_delta":
            if conditioning_volume is None or not hasattr(conditioning_volume, "morph_qsm0"):
                raise RuntimeError("residual_delta rollout requires ConditioningSpec with morph_qsm0.")
            morph_qsm0 = conditioning_volume.morph_qsm0
            if morph_qsm0 is None:
                raise RuntimeError("residual_delta rollout missing morph_qsm0.")
            return base.run_residual_delta_refinement_rollout(
                model=model,
                residual_base=morph_qsm0,
                residual_scale=self.args.morph_residual_scale,
                sigma_schedule=sigma_schedule,
                field_gt=field_gt,
                mask=mask,
                dipole_kernel=self.dipole_kernel,
                dc_policy=dc_policy,
                sigma_data=self.sigma_data,
                sigma_min=self.sigma_min,
                get_scalings_for_boundary_condition=self.get_scalings_for_boundary_condition,
                use_amp=self.use_amp if model is self.model else False,
                conditioning_volume=conditioning_volume,
                collect_states=collect_states,
                collect_priors=not infer,
            )
        return base.run_refinement_rollout(
            model=model,
            x_start=x_init,
            sigma_schedule=sigma_schedule,
            field_gt=field_gt,
            mask=mask,
            dipole_kernel=self.dipole_kernel,
            dc_policy=dc_policy,
            sigma_data=self.sigma_data,
            sigma_min=self.sigma_min,
            get_scalings_for_boundary_condition=self.get_scalings_for_boundary_condition,
            use_amp=self.use_amp if model is self.model else False,
            conditioning_volume=conditioning_volume,
            collect_states=collect_states,
            # Training needs priors for migrated losses; inference keeps the
            # stable path and avoids extra memory overhead.
            collect_priors=not infer,
        )

    def compute_physics_loss(self, final_prior: torch.Tensor, field_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        predicted_field = base.forward_field_from_chi(final_prior, self.dipole_kernel, mask=mask)
        return base.masked_mse(predicted_field, field_gt, mask)

    def compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.args.loss_gradient_weight <= 0:
            return self.zero_loss_anchor()
        return base.masked_gradient_pseudo_huber_loss(
            pred,
            target,
            mask,
            delta=self.args.gradient_loss_delta,
        )

    def compute_consistency_loss(self, student_rollout, ema_rollout, mask: torch.Tensor) -> torch.Tensor:
        if not self.args.use_ema_consistency or ema_rollout is None:
            return self.zero_loss_anchor()
        if not student_rollout.priors or not ema_rollout.priors:
            return self.zero_loss_anchor()

        losses = []
        for step_idx, (student_prior, ema_prior) in enumerate(zip(student_rollout.priors, ema_rollout.priors)):
            # Each prior corresponds to the sigma used to produce it at the same
            # rollout step. This keeps CT weighting aligned with the model output
            # that generated the prior rather than the following post-DC state.
            losses.append(
                base.pseudo_huber_loss(
                    student_prior,
                    ema_prior.detach(),
                    mask,
                    delta=self.args.pseudo_huber_delta,
                    weight=base.sigma_weight(
                        student_rollout.sigma_schedule[min(step_idx, len(student_rollout.sigma_schedule) - 1)],
                        power=self.args.ct_sigma_weight_power,
                    ),
                )
            )
        if not losses:
            return self.zero_loss_anchor()
        return torch.stack(losses).mean()

    def specialization_loss(self, batch: dict):
        x_init = batch["x_init"].to(self.device, non_blocking=True)
        chi_gt = batch["chi_gt"].to(self.device, non_blocking=True)
        field_gt = batch["field"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        x_init = self.build_rollout_x_init(x_init, field_gt, mask)
        cone_risk = batch["cone_risk"].to(self.device, non_blocking=True) if "cone_risk" in batch else None
        teacher_endpoint = None
        if "teacher_endpoint" in batch:
            teacher_endpoint = batch["teacher_endpoint"].to(self.device, non_blocking=True)
        morph_qsm0 = self.build_morph_prior(field_gt, mask)
        rollout_x_init = morph_qsm0 if self.args.morph_prior_mode == "replace_xinit" else x_init
        if self.args.morph_prior_mode == "residual_delta":
            rollout_x_init = torch.zeros_like(x_init)
        conditioning_volume = self.build_conditioning_volume(
            rollout_x_init,
            field_gt,
            mask,
            cone_risk=cone_risk,
            morph_qsm0=morph_qsm0,
        )
        if self.args.morph_prior_mode == "residual_delta" and conditioning_volume is None:
            conditioning_volume = base.ConditioningSpec(
                mode="none",
                morph_qsm0=morph_qsm0,
            )
        effective_ct_weight = self.current_ct_weight()
        effective_teacher_aux_weight = self.current_teacher_aux_weight()

        train_nfe = self.choose_train_nfe()
        role_enabled = self.two_step_role_enabled()
        student_rollout = self.run_rollout_with_model(
            model=self.model,
            x_init=rollout_x_init,
            field_gt=field_gt,
            mask=mask,
            conditioning_volume=conditioning_volume,
            num_steps=train_nfe,
            infer=False,
            collect_states=role_enabled,
        )

        ema_rollout = None
        if self.args.use_ema_consistency and effective_ct_weight > 0:
            with torch.no_grad():
                ema_rollout = self.run_rollout_with_model(
                    model=self.ema_model,
                    x_init=rollout_x_init,
                    field_gt=field_gt,
                    mask=mask,
                    conditioning_volume=conditioning_volume,
                    num_steps=train_nfe,
                    infer=False,
                    collect_states=False,
                )

        final_prior = student_rollout.priors[-1] if student_rollout.priors else student_rollout.final_state
        gt_target = final_prior if str(self.args.gt_loss_target) == "final_prior" else student_rollout.final_state
        gt_loss = self.compute_gt_loss(gt_target, chi_gt, mask)
        gradient_loss = self.compute_gradient_loss(student_rollout.final_state, chi_gt, mask)
        phys_loss = self.compute_physics_loss(final_prior, field_gt, mask)
        ct_loss = self.compute_consistency_loss(student_rollout, ema_rollout, mask)
        spectral_loss, spectral_weight = self.compute_spectral_loss(final_prior, chi_gt, mask)
        role_loss, role_low_loss, role_detail_loss = self.compute_two_step_role_loss(
            student_rollout=student_rollout,
            chi_gt=chi_gt,
            mask=mask,
            train_nfe=train_nfe,
        )
        morph_anchor_loss = self.compute_morph_anchor_loss(student_rollout.final_state, morph_qsm0, mask)
        field_morph_loss = self.compute_field_morphology_loss(student_rollout.final_state, field_gt, mask)
        teacher_aux_loss = (
            self.compute_teacher_aux_loss(student_rollout.final_state, teacher_endpoint, mask)
            if effective_teacher_aux_weight > 0
            else self.zero_loss_anchor()
        )
        macro_dc_loss, macro_dc_applied = self.compute_macro_dc_alignment_loss()
        macro_dc_contrib = float(self.args.macro_dc_align_weight) * macro_dc_loss

        total_loss = (
            self.args.loss_gt_weight * gt_loss
            + self.args.loss_gradient_weight * gradient_loss
            + self.args.loss_phys_weight * phys_loss
            + effective_ct_weight * ct_loss
            + spectral_weight * spectral_loss
            + role_loss
            + float(self.args.loss_morph_anchor_weight) * morph_anchor_loss
            + float(self.args.loss_field_morph_weight) * field_morph_loss
            + effective_teacher_aux_weight * teacher_aux_loss
            + macro_dc_contrib
        )

        loss_terms = [
            gt_loss,
            gradient_loss,
            phys_loss,
            ct_loss,
            spectral_loss,
            role_loss,
            role_low_loss,
            role_detail_loss,
            morph_anchor_loss,
            field_morph_loss,
            teacher_aux_loss,
            macro_dc_loss,
            total_loss,
        ]
        local_valid = all(bool(torch.isfinite(term).all().item()) for term in loss_terms)
        local_valid = local_valid and all(bool(torch.isfinite(state).all().item()) for state in student_rollout.states)
        local_valid = local_valid and all(bool(torch.isfinite(prior).all().item()) for prior in student_rollout.priors)
        if ema_rollout is not None:
            local_valid = local_valid and all(bool(torch.isfinite(state).all().item()) for state in ema_rollout.states)
            local_valid = local_valid and all(bool(torch.isfinite(prior).all().item()) for prior in ema_rollout.priors)
        global_valid = self._global_valid(local_valid)

        if not global_valid:
            total_loss = self.zero_loss_anchor()

        return total_loss, {
            "gt_loss": float(gt_loss.detach().cpu()) if global_valid else 0.0,
            "gradient_loss": float(gradient_loss.detach().cpu()) if global_valid else 0.0,
            "phys_loss": float(phys_loss.detach().cpu()) if global_valid else 0.0,
            "ct_loss": float(ct_loss.detach().cpu()) if global_valid else 0.0,
            "spectral_loss": float(spectral_loss.detach().cpu()) if global_valid else 0.0,
            "spectral_weight": float(spectral_weight),
            "two_step_role_loss": float(role_loss.detach().cpu()) if global_valid else 0.0,
            "two_step_role_low_loss": float(role_low_loss.detach().cpu()) if global_valid else 0.0,
            "two_step_role_detail_loss": float(role_detail_loss.detach().cpu()) if global_valid else 0.0,
            "two_step_role_low_weight": float(self.args.two_step_role_low_weight),
            "two_step_role_detail_weight": float(self.args.two_step_role_detail_weight),
            "morph_anchor_loss": float(morph_anchor_loss.detach().cpu()) if global_valid else 0.0,
            "field_morph_loss": float(field_morph_loss.detach().cpu()) if global_valid else 0.0,
            "morph_anchor_weight": float(self.args.loss_morph_anchor_weight),
            "field_morph_weight": float(self.args.loss_field_morph_weight),
            "teacher_aux_loss": float(teacher_aux_loss.detach().cpu()) if global_valid else 0.0,
            "teacher_aux_weight": float(effective_teacher_aux_weight),
            "macro_dc_loss": float(macro_dc_loss.detach().cpu()) if global_valid else 0.0,
            "macro_dc_contrib": float(macro_dc_contrib.detach().cpu()) if global_valid else 0.0,
            "macro_dc_applied": float(macro_dc_applied) if global_valid else 0.0,
            "gradient_weight": float(self.args.loss_gradient_weight),
            "train_nfe": float(train_nfe),
            "ct_weight": float(effective_ct_weight),
            "curriculum_alpha": float(self.current_curriculum_alpha()),
            "phase2_active": bool(self.phase2_active()),
            "gt_contrib": float(self.args.loss_gt_weight * gt_loss.detach().cpu()) if global_valid else 0.0,
            "gradient_contrib": float(self.args.loss_gradient_weight * gradient_loss.detach().cpu()) if global_valid else 0.0,
            "phys_contrib": float(self.args.loss_phys_weight * phys_loss.detach().cpu()) if global_valid else 0.0,
            "ct_contrib": float(effective_ct_weight * ct_loss.detach().cpu()) if global_valid else 0.0,
            "spectral_contrib": float(spectral_weight * spectral_loss.detach().cpu()) if global_valid else 0.0,
            "two_step_role_contrib": float(role_loss.detach().cpu()) if global_valid else 0.0,
            "morph_anchor_contrib": (
                float(self.args.loss_morph_anchor_weight * morph_anchor_loss.detach().cpu()) if global_valid else 0.0
            ),
            "field_morph_contrib": (
                float(self.args.loss_field_morph_weight * field_morph_loss.detach().cpu()) if global_valid else 0.0
            ),
            "teacher_aux_contrib": float(effective_teacher_aux_weight * teacher_aux_loss.detach().cpu()) if global_valid else 0.0,
            "is_valid": bool(global_valid),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.set_defaults(preset_name="stage4_lossplacement_fullmigration_v1")
    parser.add_argument("--macro-dc-align-weight", type=float, default=0.0)
    parser.add_argument("--macro-dc-every", type=int, default=8)
    parser.add_argument("--macro-dc-shape", type=str, default="96,96,64")
    parser.add_argument("--macro-dc-patch-shape", type=str, default="64,64,64")
    parser.add_argument("--macro-dc-stride", type=str, default="32,32,64")
    parser.add_argument("--macro-dc-lam", type=float, default=0.08)
    parser.add_argument("--macro-dc-cg-steps", type=int, default=8)
    parser.add_argument("--macro-dc-source-root", type=str, default="data/QSM_benchmark_dataset/train")
    parser.add_argument(
        "--x-init-override-mode",
        choices=["none", "tkd020_scaled", "blend_projected_qsm0", "pure_noise"],
        default="none",
        help=(
            "Training-time x_init perturbation. 'tkd020_scaled' computes centered TKD and LS-matches adjoint scale per patch. "
            "'blend_projected_qsm0' builds msQSM-style qsm0, LS-projects it into the cached x_init state, then applies a small blend. "
            "'pure_noise' replaces x_init with masked Gaussian noise N(0, sigma_max^2) (pure-prior training; pair with "
            "--cg-steps-train 0 and --no-final-polish-train for a DC-free rollout)."
        ),
    )
    parser.add_argument(
        "--gt-loss-target",
        choices=["final_post_dc", "final_prior"],
        default="final_post_dc",
        help="State supervised by the GT loss. 'final_prior' is the bare network endpoint (pre-DC).",
    )
    parser.add_argument("--x-init-tkd-threshold", type=float, default=0.20)
    parser.add_argument("--x-init-qsm0-blend-weight", type=float, default=0.10)
    return parser


def resolve_macro_dc_args(args: argparse.Namespace) -> argparse.Namespace:
    def resolve_triplet(value, cast):
        if isinstance(value, (tuple, list)):
            if len(value) != 3:
                raise ValueError(f"Expected triplet, got {value!r}")
            return tuple(cast(part) for part in value)
        return base.parse_triplet(value, cast)

    args.macro_dc_shape = resolve_triplet(args.macro_dc_shape, int)
    args.macro_dc_patch_shape = resolve_triplet(args.macro_dc_patch_shape, int)
    args.macro_dc_stride = resolve_triplet(args.macro_dc_stride, int)
    if args.macro_dc_align_weight < 0:
        raise ValueError("--macro-dc-align-weight must be >= 0")
    if args.macro_dc_every < 1:
        raise ValueError("--macro-dc-every must be >= 1")
    if args.macro_dc_lam <= 0:
        raise ValueError("--macro-dc-lam must be > 0")
    if args.macro_dc_cg_steps < 0:
        raise ValueError("--macro-dc-cg-steps must be >= 0")
    if args.x_init_tkd_threshold <= 0:
        raise ValueError("--x-init-tkd-threshold must be > 0")
    for macro, patch, stride in zip(args.macro_dc_shape, args.macro_dc_patch_shape, args.macro_dc_stride):
        if macro < patch:
            raise ValueError("--macro-dc-shape must be >= --macro-dc-patch-shape on every axis")
        if stride <= 0:
            raise ValueError("--macro-dc-stride values must be positive")

    root = Path(args.macro_dc_source_root).expanduser()
    if not root.is_absolute():
        candidates = []
        if args.project_root:
            candidates.append(Path(args.project_root).expanduser() / root)
        candidates.extend([Path.cwd() / root, Path(__file__).resolve().parents[2] / root])
        for candidate in candidates:
            if candidate.exists():
                root = candidate
                break
        else:
            root = candidates[0]
    args.macro_dc_source_root = str(root.resolve())
    return args


def main() -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:]
    args = base.parse_runtime_args(parser.parse_args(raw_argv), parser, raw_argv)
    args = resolve_macro_dc_args(args)
    trainer = LossPlacementMigrationTrainer(args)
    if trainer.rank == 0:
        print("Loss placement experiment:")
        for key, value in current_loss_placement_summary(args).items():
            print(f"  {key}: {value}")
        print(f"  rollout_start: {args.x_init_override_mode}")
        print(
            f"  train_dc: cg_steps_train={args.cg_steps_train} "
            f"final_polish_train={bool(args.final_polish_train)}"
        )
    trainer.train()


if __name__ == "__main__":
    main()
