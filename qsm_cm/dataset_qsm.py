from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


def _ensure_tuple(x: Union[Sequence[int], int], length: int = 3) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x,) * length
    x = tuple(int(v) for v in x)
    if len(x) != length:
        raise ValueError(f"Expected length {length}, got {len(x)} from {x}")
    return x  # type: ignore[return-value]


def random_crop_3d(
    vol: np.ndarray,
    crop_size: Sequence[int],
    rng: random.Random,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Random 3D crop. vol shape: [D,H,W] or [C,D,H,W]. Returns cropped vol and start idx."""
    if vol.ndim == 4:
        _, d, h, w = vol.shape
    elif vol.ndim == 3:
        d, h, w = vol.shape
    else:
        raise ValueError(f"Unsupported ndim {vol.ndim} for random_crop_3d")
    cd, ch, cw = _ensure_tuple(crop_size)
    if cd > d or ch > h or cw > w:
        raise ValueError(f"Crop {crop_size} larger than volume {vol.shape}")
    sd = rng.randint(0, d - cd)
    sh = rng.randint(0, h - ch)
    sw = rng.randint(0, w - cw)
    if vol.ndim == 4:
        cropped = vol[:, sd : sd + cd, sh : sh + ch, sw : sw + cw]
    else:
        cropped = vol[sd : sd + cd, sh : sh + ch, sw : sw + cw]
    return cropped, (sd, sh, sw)


def crop_like(ref: np.ndarray, vol: np.ndarray, start: Tuple[int, int, int]) -> np.ndarray:
    """Crop `vol` with same start/size as `ref` (ref may be C-first or not)."""
    if ref.ndim == 4:
        cd, ch, cw = ref.shape[1:]
    else:
        cd, ch, cw = ref.shape
    sd, sh, sw = start
    if vol.ndim == 4:
        return vol[:, sd : sd + cd, sh : sh + ch, sw : sw + cw]
    return vol[sd : sd + cd, sh : sh + ch, sw : sw + cw]


class QSMDistillDataset(Dataset):
    """Dataset for QSM consistency distillation based on exported teacher trajectories.

    Each npz is expected to contain:
      - b: field map, [D,H,W]
      - mask: brain mask, [D,H,W]
      - chi_T: teacher final reconstruction, [D,H,W]
      - chis: dict-like of {t: chi_t [D,H,W]}  (stored as np.object in npz)
      - time_pairs: optional (N,2) array of (t_last, t_cur) transitions
      - meta: optional metadata (case id, spacing, etc.)
      - chi_COSMOS: optional ground truth, [D,H,W]
    """

    def __init__(
        self,
        traj_files: Iterable[Union[Path, str]],
        t_pool: Sequence[int],
        patch_size: Sequence[int] = (64, 64, 64),
        cosmos_root: Optional[Path] = None,
        rng_seed: int = 1234,
        repeats_per_volume: int = 1,
        pair_mode: str = "random",
    ) -> None:
        self.traj_files: List[Path] = [Path(p) for p in traj_files]
        self.patch_size = _ensure_tuple(patch_size)
        self.t_pool = [int(t) for t in t_pool]
        self.cosmos_root = cosmos_root
        self.rng_seed = int(rng_seed)
        self.repeats_per_volume = max(1, int(repeats_per_volume))
        if pair_mode not in {"random", "adjacent"}:
            raise ValueError(f"pair_mode must be 'random' or 'adjacent', got {pair_mode}")
        self.pair_mode = pair_mode

    def __len__(self) -> int:
        return len(self.traj_files) * self.repeats_per_volume

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        out: Dict[str, np.ndarray] = {}
        for key in ["b", "mask", "chi_T", "chi_COSMOS", "dipole_kernel", "calci", "time_pairs"]:
            if key in data:
                out[key] = data[key]
        if "chis" in data:
            chis_obj = data["chis"].item() if hasattr(data["chis"], "item") else data["chis"]
            out["chis"] = chis_obj  # type: ignore[assignment]
        if "meta" in data:
            out["meta"] = data["meta"]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = random.Random(self.rng_seed + 1000003 * worker_id + 9176 * int(idx))

        vol_idx = idx % len(self.traj_files)
        path = self.traj_files[vol_idx]
        arrs = self._load_npz(path)
        chis: Dict[int, np.ndarray] = arrs["chis"]  # type: ignore[assignment]

        # pick two time steps
        if self.pair_mode == "adjacent" and "time_pairs" in arrs:
            pairs = arrs["time_pairs"]
            candidates = [
                (int(t_last), int(t_cur))
                for t_last, t_cur in pairs
                if int(t_last) > int(t_cur) and int(t_last) in chis and int(t_cur) in chis
            ]
            if candidates:
                t_k, t_l = rng.choice(candidates)
            else:
                t_k, t_l = rng.sample(self.t_pool, 2)
        else:
            t_k, t_l = rng.sample(self.t_pool, 2)
        chi_tk = chis[int(t_k)]
        chi_tl = chis[int(t_l)]
        chi_T = arrs["chi_T"]
        b = arrs["b"]
        mask = arrs.get("mask", np.ones_like(b))

        # crop
        chi_tk_c, start = random_crop_3d(chi_tk, self.patch_size, rng)
        chi_tl_c = crop_like(chi_tk_c, chi_tl, start)
        chi_T_c = crop_like(chi_tk_c, chi_T, start)
        b_c = crop_like(chi_tk_c, b, start)
        mask_c = crop_like(chi_tk_c, mask, start)

        sample: Dict[str, torch.Tensor] = {
            "chi_tk": torch.from_numpy(chi_tk_c).unsqueeze(0).float(),
            "chi_tl": torch.from_numpy(chi_tl_c).unsqueeze(0).float(),
            "chi_T": torch.from_numpy(chi_T_c).unsqueeze(0).float(),
            "b": torch.from_numpy(b_c).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_c).unsqueeze(0).float(),
            "t_k": torch.tensor([t_k], dtype=torch.long),
            "t_l": torch.tensor([t_l], dtype=torch.long),
            "meta_path": torch.tensor(vol_idx, dtype=torch.long),
        }

        if "chi_COSMOS" in arrs:
            chi_COSMOS_c = crop_like(chi_tk_c, arrs["chi_COSMOS"], start)
            sample["chi_COSMOS"] = torch.from_numpy(chi_COSMOS_c).unsqueeze(0).float()
        if "dipole_kernel" in arrs:
            dip = arrs["dipole_kernel"]
            dip_c = crop_like(chi_tk_c, dip, start)
            sample["dipole_kernel"] = torch.from_numpy(dip_c).unsqueeze(0).float()
        if "calci" in arrs:
            calci_c = crop_like(chi_tk_c, arrs["calci"], start)
            sample["calci"] = torch.from_numpy(calci_c).unsqueeze(0).float()
        return sample


__all__ = ["QSMDistillDataset", "random_crop_3d", "crop_like"]
