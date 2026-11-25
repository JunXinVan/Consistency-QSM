from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


def _add_diffusion_qsm_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2] / "Diffusion-QSM"
    if repo_root.exists():
        sys.path.append(str(repo_root))
    return repo_root


_DQSM_ROOT = _add_diffusion_qsm_to_path()

try:
    from guided_diffusion.script_util import (  # type: ignore
        model_and_diffusion_defaults,
        create_model_and_diffusion,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import Diffusion-QSM. Ensure ../Diffusion-QSM is available and importable."
    ) from exc


def _append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while x.ndim < target_ndim:
        x = x[..., None]
    return x


def _expand_timesteps(t: torch.Tensor, batch: int) -> torch.Tensor:
    if t.ndim == 0:
        return t.expand(batch)
    if t.ndim == 1 and t.shape[0] == 1 and batch > 1:
        return t.expand(batch)
    return t


@dataclass(frozen=True)
class DiffusionQSMTeacher:
    model: torch.nn.Module
    diffusion: object
    sqrt_alpha_bar: torch.Tensor  # [T]
    sigma_table: torch.Tensor  # [T]

    @staticmethod
    def load(ckpt: Path, device: torch.device) -> "DiffusionQSMTeacher":
        defaults = model_and_diffusion_defaults()
        model, diffusion = create_model_and_diffusion(**defaults)

        def _torch_load(path: Path) -> object:
            try:
                return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
            except TypeError:
                return torch.load(path, map_location="cpu")

        raw = _torch_load(ckpt)
        if isinstance(raw, dict):
            state = raw.get("model_state_dict") or raw.get("state_dict") or raw.get("model_state") or raw
        else:
            state = raw

        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format: expected state_dict dict, got {type(state)}")

        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module.") :]: v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        # Precompute alpha_bar and sigma(t)=sqrt((1-a)/a) on device.
        alpha_bar = torch.tensor(getattr(diffusion, "alphas_cumprod"), dtype=torch.float32, device=device)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sigma_table = torch.sqrt((1.0 - alpha_bar) / torch.clamp(alpha_bar, min=1e-12))
        return DiffusionQSMTeacher(
            model=model,
            diffusion=diffusion,
            sqrt_alpha_bar=sqrt_alpha_bar,
            sigma_table=sigma_table,
        )

    @torch.no_grad()
    def denoise_edm(self, x_edm: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Teacher denoiser D(x, sigma) in EDM variable x = x0 + sigma*eps.

        We convert EDM x to the DDPM x_t expected by Diffusion-QSM at discrete timestep t:
          x_ddpm = sqrt(alpha_bar[t]) * x_edm
        and then use diffusion.p_mean_variance(...).pred_xstart as x0.
        """
        if t.dtype != torch.long:
            t = t.long()
        t = t.to(x_edm.device)
        scale = self.sqrt_alpha_bar[t]  # [B]
        x_ddpm = x_edm * _append_dims(scale, x_edm.ndim)
        out = self.diffusion.p_mean_variance(  # type: ignore[attr-defined]
            self.model,
            x_ddpm,
            t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
        )
        return out["pred_xstart"]

    @torch.no_grad()
    def t_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map continuous sigma to the nearest discrete DDPM timestep index."""
        sigma = sigma.to(self.sigma_table.device, dtype=torch.float32)
        sigma_flat = sigma.reshape(-1)

        idx_hi = torch.bucketize(sigma_flat, self.sigma_table)
        idx_hi = torch.clamp(idx_hi, 0, self.sigma_table.numel() - 1)
        idx_lo = torch.clamp(idx_hi - 1, 0, self.sigma_table.numel() - 1)

        sigma_lo = self.sigma_table[idx_lo]
        sigma_hi = self.sigma_table[idx_hi]
        pick_lo = (sigma_flat - sigma_lo).abs() < (sigma_flat - sigma_hi).abs()
        idx = torch.where(pick_lo, idx_lo, idx_hi)
        return idx.view(sigma.shape).to(dtype=torch.long)

    @torch.no_grad()
    def t_from_sigma_float(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map continuous sigma to fractional DDPM index by linear interpolation."""
        sigma = sigma.to(self.sigma_table.device, dtype=torch.float32)
        sigma_flat = sigma.reshape(-1)
        sigma_min = self.sigma_table[0]
        sigma_max = self.sigma_table[-1]
        sigma_flat = torch.clamp(sigma_flat, sigma_min, sigma_max)

        idx_hi = torch.bucketize(sigma_flat, self.sigma_table)
        idx_hi = torch.clamp(idx_hi, 0, self.sigma_table.numel() - 1)
        idx_lo = torch.clamp(idx_hi - 1, 0, self.sigma_table.numel() - 1)

        sigma_lo = self.sigma_table[idx_lo]
        sigma_hi = self.sigma_table[idx_hi]
        denom = torch.clamp(sigma_hi - sigma_lo, min=1e-12)
        w = (sigma_flat - sigma_lo) / denom
        t_float = idx_lo.to(dtype=torch.float32) + w
        return t_float.view(sigma.shape)

    @torch.no_grad()
    def denoise_sigma(self, x_edm: torch.Tensor, sigma: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
        """Teacher denoiser D(x, sigma) operating on EDM variable x=x0+sigma*eps."""
        mode = mode.lower().strip()
        if mode not in {"nearest", "interp"}:
            raise ValueError(f"Unknown teacher mode: {mode}")

        if mode == "interp":
            t_float = self.t_from_sigma_float(sigma.to(x_edm.device))
            t0 = torch.floor(t_float).to(dtype=torch.long)
            t1 = torch.clamp(t0 + 1, max=self.sigma_table.numel() - 1)
            t0 = _expand_timesteps(t0, x_edm.shape[0])
            t1 = _expand_timesteps(t1, x_edm.shape[0])
            w = (t_float - t0.to(dtype=torch.float32)).to(x_edm.device)
            w = _append_dims(w, x_edm.ndim)

            x0_0 = self.denoise_edm(x_edm, t0)
            if torch.equal(t0, t1):
                return x0_0
            x0_1 = self.denoise_edm(x_edm, t1)
            return x0_0 + (x0_1 - x0_0) * w

        t = self.t_from_sigma(sigma.to(x_edm.device))
        t = _expand_timesteps(t, x_edm.shape[0])
        return self.denoise_edm(x_edm, t)


__all__ = ["DiffusionQSMTeacher"]
