from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn


def _add_diffusion_qsm_to_path() -> None:
    """Ensure Diffusion-QSM is importable."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "Diffusion-QSM",
        here.parents[4] / "repro_diffqsm" / "repos" / "Diffusion-QSM",
    ]
    for repo_root in candidates:
        if repo_root.exists():
            repo_str = str(repo_root)
            if repo_str not in sys.path:
                sys.path.append(repo_str)
            return


_add_diffusion_qsm_to_path()

try:
    from guided_diffusion.unet import UNetModel  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "Failed to import Diffusion-QSM guided_diffusion.UNetModel. "
        "Ensure ../Diffusion-QSM is available and on PYTHONPATH."
    ) from exc


@dataclass
class StudentUNetConfig:
    image_size: int = 64
    model_channels: int = 96
    num_res_blocks: int = 2
    channel_mult: Tuple[int, int, int, int, int, int] = (1, 1, 2, 2, 3, 3)
    attention_resolutions: Tuple[int, int, int] = (8, 16, 32)
    num_heads: int = 4
    dropout: float = 0.0
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_fp16: bool = False
    use_new_attention_order: bool = False


def build_student_unet(cfg: StudentUNetConfig, in_channels: int = 3, out_channels: int = 1) -> UNetModel:
    """Create a 3D UNet aligned with Diffusion-QSM defaults, but with extra channels for conditioning."""
    # In Diffusion-QSM, attention_resolutions gives the downsample rates
    # {8,16,32} directly; no need to divide by image_size. Keep the same
    # here to avoid 3D attention exploding memory at high resolutions.
    attention_ds = cfg.attention_resolutions
    model = UNetModel(
        image_size=cfg.image_size,
        in_channels=in_channels,
        model_channels=cfg.model_channels,
        out_channels=out_channels,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=attention_ds,
        dropout=cfg.dropout,
        channel_mult=cfg.channel_mult,
        num_classes=None,
        use_checkpoint=cfg.use_checkpoint,
        use_fp16=cfg.use_fp16,
        num_heads=cfg.num_heads,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=cfg.use_scale_shift_norm,
        resblock_updown=cfg.resblock_updown,
        use_new_attention_order=cfg.use_new_attention_order,
    )
    return model


class QSMStudent(nn.Module):
    """3D student model: UNet3D conditioned on field map and mask."""

    def __init__(self, cfg: StudentUNetConfig, extra_conditioning_channels: int = 0) -> None:
        super().__init__()
        if extra_conditioning_channels < 0:
            raise ValueError(f"extra_conditioning_channels must be >= 0, got {extra_conditioning_channels}")
        self.extra_conditioning_channels = int(extra_conditioning_channels)
        self.unet = build_student_unet(cfg, in_channels=3 + self.extra_conditioning_channels, out_channels=1)

    def forward(
        self,
        chi_t: torch.Tensor,
        t: torch.Tensor,
        b: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        extra_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            chi_t: [B,1,D,H,W] noisy chi at time t
            t: [B] or scalar time step
            b: [B,1,D,H,W] field map
            mask: [B,1,D,H,W] brain mask (default ones)
        """
        if mask is None:
            mask = torch.ones_like(chi_t)
        channels = [chi_t, b, mask]
        if self.extra_conditioning_channels > 0:
            if extra_conditioning is None:
                raise RuntimeError(
                    f"QSMStudent was built with {self.extra_conditioning_channels} extra conditioning channels, "
                    "but forward() received None."
                )
            if extra_conditioning.shape[1] != self.extra_conditioning_channels:
                raise ValueError(
                    f"Expected {self.extra_conditioning_channels} extra conditioning channels, "
                    f"got {extra_conditioning.shape[1]}."
                )
            channels.append(extra_conditioning)
        elif extra_conditioning is not None:
            raise RuntimeError("extra_conditioning was provided to a QSMStudent without extra conditioning channels.")
        x = torch.cat(channels, dim=1)
        return self.unet(x, t)


__all__ = ["StudentUNetConfig", "QSMStudent", "build_student_unet"]
