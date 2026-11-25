from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def ensure_5d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    original_shape = tuple(x.shape)
    if x.ndim == 3:
        return x.unsqueeze(0).unsqueeze(0), original_shape
    if x.ndim == 4:
        return x.unsqueeze(1), original_shape
    if x.ndim == 5:
        return x, original_shape
    raise ValueError(f"Expected a 3D, 4D, or 5D tensor, got shape {original_shape}")


def restore_shape(x: torch.Tensor, original_shape: Sequence[int]) -> torch.Tensor:
    if len(original_shape) == 3:
        return x.squeeze(0).squeeze(0)
    if len(original_shape) == 4:
        return x.squeeze(1)
    return x


def expand_kernel(dipole_kernel: torch.Tensor, target_ndim: int) -> torch.Tensor:
    kernel = dipole_kernel
    while kernel.ndim < target_ndim:
        kernel = kernel.unsqueeze(0)
    return kernel


def _fftn(x: torch.Tensor, centered: bool) -> torch.Tensor:
    if not centered:
        return torch.fft.fftn(x, dim=(-3, -2, -1), norm="ortho")
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=(-3, -2, -1)), dim=(-3, -2, -1), norm="ortho"),
        dim=(-3, -2, -1),
    )


def _ifftn(x: torch.Tensor, centered: bool) -> torch.Tensor:
    if not centered:
        return torch.fft.ifftn(x, dim=(-3, -2, -1), norm="ortho")
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-3, -2, -1)), dim=(-3, -2, -1), norm="ortho"),
        dim=(-3, -2, -1),
    )


def masked_minmax(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8) -> torch.Tensor:
    x_5d, original_shape = ensure_5d(x)
    if mask is None:
        mask_5d = torch.ones_like(x_5d)
    else:
        mask_5d, _ = ensure_5d(mask)
        mask_5d = mask_5d.to(device=x_5d.device, dtype=x_5d.dtype)
    out = torch.zeros_like(x_5d)
    for batch_idx in range(x_5d.shape[0]):
        for channel_idx in range(x_5d.shape[1]):
            valid = mask_5d[batch_idx, channel_idx] > 0
            if not torch.any(valid):
                continue
            vals = x_5d[batch_idx, channel_idx][valid]
            lo = vals.min()
            hi = vals.max()
            out[batch_idx, channel_idx] = (x_5d[batch_idx, channel_idx] - lo) / (hi - lo).clamp_min(eps)
            out[batch_idx, channel_idx] = out[batch_idx, channel_idx].clamp(0.0, 1.0) * valid.to(x_5d.dtype)
    return restore_shape(out, original_shape)


def _sobel_kernel(axis: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    smooth = torch.tensor([1.0, 2.0, 1.0], dtype=dtype, device=device)
    deriv = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype, device=device)
    axes = [smooth, smooth, smooth]
    axes[axis] = deriv
    kernel = axes[0].view(3, 1, 1) * axes[1].view(1, 3, 1) * axes[2].view(1, 1, 3)
    return kernel / kernel.abs().sum().clamp_min(1e-8)


def morphology_weight(field: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    field_5d, original_shape = ensure_5d(field)
    channels = field_5d.shape[1]
    gradients = []
    for axis in range(3):
        kernel = _sobel_kernel(axis, dtype=field_5d.dtype, device=field_5d.device)
        kernel = kernel.view(1, 1, 3, 3, 3).repeat(channels, 1, 1, 1, 1)
        gradients.append(F.conv3d(field_5d, kernel, padding=1, groups=channels).abs())
    weight = torch.stack(gradients, dim=0).sum(dim=0)
    if mask is not None:
        mask_5d, _ = ensure_5d(mask)
        weight = weight * mask_5d.to(device=weight.device, dtype=weight.dtype)
    return restore_shape(masked_minmax(weight, mask=mask), original_shape)


def tkd_inversion(
    field: torch.Tensor,
    dipole_kernel: torch.Tensor,
    threshold: float,
    mask: torch.Tensor | None = None,
    centered_fft: bool = True,
) -> torch.Tensor:
    field_5d, original_shape = ensure_5d(field)
    kernel = expand_kernel(dipole_kernel.to(device=field_5d.device, dtype=field_5d.dtype), field_5d.ndim)
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError(f"TKD threshold must be positive, got {threshold}")
    sign = torch.where(kernel >= 0, torch.ones_like(kernel), -torch.ones_like(kernel))
    clipped = torch.where(kernel.abs() < threshold, threshold * sign, kernel)
    clipped = torch.where(clipped == 0, torch.full_like(clipped, threshold), clipped)
    qsm = _ifftn(_fftn(field_5d, centered=centered_fft) / clipped, centered=centered_fft).real
    if mask is not None:
        mask_5d, _ = ensure_5d(mask)
        qsm = qsm * mask_5d.to(device=qsm.device, dtype=qsm.dtype)
    return restore_shape(torch.where(torch.isfinite(qsm), qsm, torch.zeros_like(qsm)), original_shape)


def nonlinear_cone_process(
    qsm: torch.Tensor,
    dipole_kernel: torch.Tensor,
    mask: torch.Tensor | None = None,
    cone_threshold: float = 0.1,
    cone_power: float = 0.2,
    centered_fft: bool = True,
) -> torch.Tensor:
    qsm_5d, original_shape = ensure_5d(qsm)
    kernel = expand_kernel(dipole_kernel.to(device=qsm_5d.device, dtype=qsm_5d.dtype), qsm_5d.ndim)
    k_qsm = _fftn(qsm_5d, centered=centered_fft)
    cone = kernel.abs() < float(cone_threshold)
    factor = kernel.abs().clamp_min(0.0).pow(float(cone_power)) * torch.sign(kernel)
    k_qsm = torch.where(cone, k_qsm * factor, k_qsm)
    out = _ifftn(k_qsm, centered=centered_fft).real
    if mask is not None:
        mask_5d, _ = ensure_5d(mask)
        out = out * mask_5d.to(device=out.device, dtype=out.dtype)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return restore_shape(out, original_shape)


def build_morph_qsm0(
    field: torch.Tensor,
    dipole_kernel: torch.Tensor,
    mask: torch.Tensor,
    tkd_low: float = 0.1,
    tkd_high: float = 0.2,
    cone_threshold: float = 0.1,
    cone_power: float = 0.2,
    centered_fft: bool = True,
) -> torch.Tensor:
    field_5d, original_shape = ensure_5d(field)
    mask_5d, _ = ensure_5d(mask)
    mask_5d = mask_5d.to(device=field_5d.device, dtype=field_5d.dtype)
    morph = morphology_weight(field_5d * mask_5d, mask_5d)
    morph_5d, _ = ensure_5d(morph)
    tkd_low_vol, _ = ensure_5d(
        tkd_inversion(field_5d, dipole_kernel, threshold=tkd_low, mask=mask_5d, centered_fft=centered_fft)
    )
    tkd_high_vol, _ = ensure_5d(
        tkd_inversion(field_5d, dipole_kernel, threshold=tkd_high, mask=mask_5d, centered_fft=centered_fft)
    )
    qsm0 = morph_5d * tkd_low_vol + (1.0 - morph_5d) * tkd_high_vol
    qsm0 = nonlinear_cone_process(
        qsm0,
        dipole_kernel=dipole_kernel,
        mask=mask_5d,
        cone_threshold=cone_threshold,
        cone_power=cone_power,
        centered_fft=centered_fft,
    )
    qsm0_5d, _ = ensure_5d(qsm0)
    return restore_shape(qsm0_5d * mask_5d, original_shape)
