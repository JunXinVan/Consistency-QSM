from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from qsm_cm.dataset_qsm import crop_like, random_crop_3d


def _ensure_tuple3(x: Sequence[int]) -> Tuple[int, int, int]:
    vals = tuple(int(v) for v in x)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 ints, got {x}")
    return vals[0], vals[1], vals[2]


def _load_nifti_3d(path: Path) -> tuple[np.ndarray, Tuple[float, float, float]]:
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "nibabel is required to load NIfTI files. Install it in your training env: `pip install nibabel`."
        ) from exc

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    data = np.asarray(data)
    while data.ndim > 3:
        data = np.squeeze(data, axis=-1)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} from {path}")
    zooms = getattr(img.header, "get_zooms", lambda: (0.0, 0.0, 0.0))()
    vox = (float(zooms[0]), float(zooms[1]), float(zooms[2])) if len(zooms) >= 3 else (0.0, 0.0, 0.0)
    return np.ascontiguousarray(data, dtype=np.float32), vox


def _maybe_find_sibling(path: Path, prefix: str) -> Optional[Path]:
    stem = path.name
    # cosmos1.nii.gz -> "1"
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    for ext in [".nii.gz", ".nii"]:
        cand = path.with_name(f"{prefix}{digits}{ext}")
        if cand.exists():
            return cand
    return None


class QSMNiftiDataset(Dataset):
    """Patch dataset for QSM volumes stored as NIfTI (.nii/.nii.gz).

    Intended for consistency distillation/training when data is provided as:
      - cosmos*.nii.gz (chi / COSMOS QSM)
      - optional sibling mask*.nii.gz
      - optional sibling phi*.nii.gz (field), currently unused (b is filled with zeros unless present)
    """

    def __init__(
        self,
        chi_files: Iterable[Union[Path, str]],
        patch_size: Sequence[int] = (64, 64, 64),
        repeats_per_volume: int = 1,
        rng_seed: int = 1234,
        chi_scale: float = 1.0,
        try_load_mask: bool = True,
        try_load_field: bool = False,
    ) -> None:
        self.chi_files: List[Path] = [Path(p) for p in chi_files]
        if not self.chi_files:
            raise ValueError("chi_files is empty")
        self.patch_size = _ensure_tuple3(patch_size)
        self.repeats_per_volume = max(1, int(repeats_per_volume))
        self.rng_seed = int(rng_seed)
        self.chi_scale = float(chi_scale)
        self.try_load_mask = bool(try_load_mask)
        self.try_load_field = bool(try_load_field)

    def __len__(self) -> int:
        return len(self.chi_files) * self.repeats_per_volume

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = random.Random(self.rng_seed + 1000003 * worker_id + 9176 * int(idx))

        vol_idx = idx % len(self.chi_files)
        chi_path = self.chi_files[vol_idx]
        chi, vox = _load_nifti_3d(chi_path)
        chi = chi * self.chi_scale

        mask = None
        if self.try_load_mask:
            mask_path = _maybe_find_sibling(chi_path, "mask")
            if mask_path is not None:
                mask, _ = _load_nifti_3d(mask_path)
        if mask is None:
            mask = np.ones_like(chi, dtype=np.float32)

        field = None
        if self.try_load_field:
            phi_path = _maybe_find_sibling(chi_path, "phi")
            if phi_path is not None:
                field, _ = _load_nifti_3d(phi_path)
        if field is None:
            field = np.zeros_like(chi, dtype=np.float32)

        chi_c, start = random_crop_3d(chi, self.patch_size, rng)
        field_c = crop_like(chi_c, field, start)
        mask_c = crop_like(chi_c, mask, start)

        voxel_t = torch.tensor([vox[0], vox[1], vox[2]], dtype=torch.float32)
        return {
            "chi": torch.from_numpy(chi_c).unsqueeze(0).float(),
            "b": torch.from_numpy(field_c).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_c).unsqueeze(0).float(),
            "voxel_size": voxel_t,
            "meta_path": torch.tensor(vol_idx, dtype=torch.long),
        }


__all__ = ["QSMNiftiDataset"]
