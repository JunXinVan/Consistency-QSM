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


def _pick_key(files: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in files:
            return k
    return None


def _to_3d(arr: np.ndarray) -> np.ndarray:
    """Convert array to [D,H,W] float32."""
    x = np.asarray(arr)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D volume (or 1x3D), got shape {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float32)


class QSMCleanDataset(Dataset):
    """Patch dataset for teacher-free consistency training on real (b, chi) pairs.

    Each case is an .npz file that should contain at least:
      - chi (or chi_COSMOS): [D,H,W] or [1,D,H,W]
      - field (or b): [D,H,W] or [1,D,H,W]
      - mask (optional): [D,H,W] or [1,D,H,W]
      - voxel_size (optional): [3] float
    """

    def __init__(
        self,
        case_files: Iterable[Union[Path, str]],
        patch_size: Sequence[int] = (64, 64, 64),
        repeats_per_volume: int = 1,
        rng_seed: int = 1234,
        require_field: bool = True,
        chi_keys: Sequence[str] = ("chi_COSMOS", "chi", "sus", "qsm"),
        field_keys: Sequence[str] = ("field", "b", "phi", "phi_use"),
        mask_keys: Sequence[str] = ("mask", "msk", "brain_mask"),
        voxel_keys: Sequence[str] = ("voxel_size", "vox", "voxel"),
    ) -> None:
        self.case_files: List[Path] = [Path(p) for p in case_files]
        if not self.case_files:
            raise ValueError("case_files is empty")
        self.patch_size = _ensure_tuple3(patch_size)
        self.repeats_per_volume = max(1, int(repeats_per_volume))
        self.rng_seed = int(rng_seed)
        self.require_field = bool(require_field)
        self.chi_keys = list(chi_keys)
        self.field_keys = list(field_keys)
        self.mask_keys = list(mask_keys)
        self.voxel_keys = list(voxel_keys)

    def __len__(self) -> int:
        return len(self.case_files) * self.repeats_per_volume

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        files = list(data.files)
        chi_key = _pick_key(files, self.chi_keys)
        if chi_key is None:
            raise KeyError(f"{path} missing chi key (tried {self.chi_keys}); has {files}")
        field_key = _pick_key(files, self.field_keys)
        if field_key is None and self.require_field:
            raise KeyError(f"{path} missing field key (tried {self.field_keys}); has {files}")
        mask_key = _pick_key(files, self.mask_keys)
        voxel_key = _pick_key(files, self.voxel_keys)

        out: Dict[str, np.ndarray] = {"chi": data[chi_key]}
        if field_key is not None:
            out["field"] = data[field_key]
        if mask_key is not None:
            out["mask"] = data[mask_key]
        if voxel_key is not None:
            out["voxel_size"] = data[voxel_key]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = random.Random(self.rng_seed + 1000003 * worker_id + 9176 * int(idx))

        vol_idx = idx % len(self.case_files)
        path = self.case_files[vol_idx]
        arrs = self._load_npz(path)

        chi = _to_3d(arrs["chi"])
        field = _to_3d(arrs["field"]) if "field" in arrs else np.zeros_like(chi, dtype=np.float32)
        mask = _to_3d(arrs["mask"]) if "mask" in arrs else np.ones_like(chi, dtype=np.float32)

        chi_c, start = random_crop_3d(chi, self.patch_size, rng)
        field_c = crop_like(chi_c, field, start)
        mask_c = crop_like(chi_c, mask, start)

        voxel_size = arrs.get("voxel_size")
        if voxel_size is None:
            voxel_t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            vox = np.asarray(voxel_size).reshape(-1)
            if vox.size < 3:
                voxel_t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            else:
                voxel_t = torch.tensor([float(vox[0]), float(vox[1]), float(vox[2])], dtype=torch.float32)

        return {
            "chi": torch.from_numpy(chi_c).unsqueeze(0).float(),
            "b": torch.from_numpy(field_c).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_c).unsqueeze(0).float(),
            "voxel_size": voxel_t,
            "meta_path": torch.tensor(vol_idx, dtype=torch.long),
        }


__all__ = ["QSMCleanDataset"]
