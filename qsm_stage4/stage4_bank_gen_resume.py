#!/usr/bin/env python3
"""
Resumable teacher-refinement cache generation for QSM Stage 4 specialization.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from qsm_bank_utils import (
    adjoint_field_to_chi,
    build_spectral_weight_map,
    build_dc_rollout_policy,
    configure_import_paths,
    create_dipole_kernel,
    create_cone_mask,
    forward_field_from_chi,
    geometric_sigma_schedule,
    load_student_components,
    masked_mse,
    parse_triplet,
    run_refinement_rollout,
    seed_everything,
    select_teacher_state_indices,
)


BANK_FORMAT = "qsm_teacher_refinement_cache_v1"


@dataclass
class TeacherTrajectoryRecord:
    backend: str
    states: list[torch.Tensor]
    state_steps: list[int]
    metadata: dict


class TeacherBackendBase:
    name = "base"

    def manifest_metadata(self) -> dict:
        return {}

    def generate(
        self,
        x_init: torch.Tensor,
        field_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        dipole_kernel: torch.Tensor,
        sample_seed: int,
    ) -> TeacherTrajectoryRecord:
        raise NotImplementedError


def resolve_constant_schedule_arg(raw_value, cast, name: str):
    if isinstance(raw_value, str):
        parsed = parse_schedule_arg(raw_value)
    else:
        parsed = raw_value

    if isinstance(parsed, list):
        values = [cast(value) for value in parsed]
        unique_values = {value for value in values}
        if len(unique_values) != 1:
            raise ValueError(
                f"{name} must be a scalar/constant schedule for this backend, got {parsed!r}"
            )
        return values[0]
    return cast(parsed)


def load_model_state_dict(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("ema_state", "model_state", "state_dict"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                return state_dict
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type in {path}: {type(payload)!r}")


class StudentCGTeacherBackend(TeacherBackendBase):
    name = "student_cg"

    def __init__(
        self,
        student_ckpt: str,
        project_root: Optional[str],
        diffusion_root: Optional[str],
        device: torch.device,
        teacher_rollout_steps: int,
        teacher_dc_policy: str,
        teacher_lambda_dc,
        teacher_cg_steps,
        final_polish: bool,
        final_polish_lam: Optional[float],
        final_polish_steps: Optional[int],
        max_sigma_factor: float,
        use_amp: bool,
    ) -> None:
        build_student_unet, StudentUNetConfig, get_scalings_for_boundary_condition = load_student_components(
            project_root=project_root,
            diffusion_root=diffusion_root,
            script_path=__file__,
        )
        ckpt = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        teacher = build_student_unet(StudentUNetConfig(), in_channels=1, out_channels=2)
        state_dict = ckpt.get("ema_state", ckpt.get("model_state", ckpt))
        teacher.load_state_dict(state_dict, strict=True)
        teacher = teacher.to(device)
        teacher.eval()

        config = ckpt.get("config", {})
        sigma_data = float(config.get("sigma_data", 0.5))
        sigma_min = float(ckpt.get("sigma_min", 0.01))
        sigma_max = max(sigma_data * max_sigma_factor, sigma_min * 1.01)

        self.teacher = teacher
        self.device = device
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.teacher_rollout_steps = int(teacher_rollout_steps)
        self.sigma_schedule_values = geometric_sigma_schedule(sigma_max, sigma_min, self.teacher_rollout_steps + 1)
        self.get_scalings_for_boundary_condition = get_scalings_for_boundary_condition
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.dc_policy = build_dc_rollout_policy(
            num_steps=self.teacher_rollout_steps,
            mode=teacher_dc_policy,
            lam=teacher_lambda_dc,
            num_cg_steps=teacher_cg_steps,
            final_polish=final_polish,
            final_polish_lam=final_polish_lam,
            final_polish_steps=final_polish_steps,
        )

    def manifest_metadata(self) -> dict:
        return {
            "teacher_backend": self.name,
            "teacher_rollout_steps": self.teacher_rollout_steps,
            "teacher_sigma_data": self.sigma_data,
            "teacher_sigma_min": self.sigma_min,
            "teacher_sigma_max": self.sigma_max,
            "teacher_sigma_schedule": [float(value) for value in self.sigma_schedule_values],
            "teacher_dc_policy": self.dc_policy.mode,
            "teacher_dc_lams": [step.lam for step in self.dc_policy.steps],
            "teacher_dc_cg_steps": [step.num_cg_steps for step in self.dc_policy.steps],
            "teacher_final_polish": self.dc_policy.final_polish is not None,
        }

    def generate(
        self,
        x_init: torch.Tensor,
        field_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        dipole_kernel: torch.Tensor,
        sample_seed: int,
    ) -> TeacherTrajectoryRecord:
        sigma_schedule = [
            torch.full((x_init.shape[0],), float(value), device=self.device, dtype=torch.float32)
            for value in self.sigma_schedule_values
        ]

        with torch.inference_mode():
            with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []):
                torch.manual_seed(int(sample_seed))
                rollout = run_refinement_rollout(
                    model=self.teacher,
                    x_start=x_init,
                    sigma_schedule=sigma_schedule,
                    field_gt=field_patch,
                    mask=mask_patch,
                    dipole_kernel=dipole_kernel,
                    dc_policy=self.dc_policy,
                    sigma_data=self.sigma_data,
                    sigma_min=self.sigma_min,
                    get_scalings_for_boundary_condition=self.get_scalings_for_boundary_condition,
                    use_amp=self.use_amp,
                    collect_states=True,
                    collect_priors=False,
                )

        state_steps = list(range(len(rollout.states)))
        return TeacherTrajectoryRecord(
            backend=self.name,
            states=[state.detach().clone() for state in rollout.states[1:]],
            state_steps=state_steps[1:],
            metadata={"teacher_sigma_schedule": [float(value) for value in self.sigma_schedule_values]},
        )


class DiffusionQSMTeacherBackend(TeacherBackendBase):
    name = "diffusion_qsm"

    def __init__(
        self,
        teacher_ckpt: str,
        diffusion_root: Optional[str],
        device: torch.device,
        patch_size: tuple[int, int, int],
        fast_timesteps: int,
        starting_point: int,
        repetition_time: int,
        eta: float,
        dc_lam: float,
        dc_cg_steps: int,
    ) -> None:
        configure_import_paths(project_root=None, diffusion_root=diffusion_root, script_path=__file__)
        try:
            from guided_diffusion.scheduler import get_schedule_jump, space_timesteps as scheduler_space_timesteps
            from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Unable to import Diffusion-QSM modules. Ensure --diffusion-root points to the Diffusion-QSM "
                "repository and the runtime environment has its dependencies installed."
            ) from exc

        if not Path(teacher_ckpt).expanduser().exists():
            raise FileNotFoundError(f"Diffusion-QSM checkpoint not found: {teacher_ckpt}")
        if fast_timesteps < 1:
            raise ValueError(f"fast_timesteps must be >= 1, got {fast_timesteps}")
        if starting_point < 0 or starting_point >= fast_timesteps:
            raise ValueError(
                f"starting_point must satisfy 0 <= starting_point < fast_timesteps, got {starting_point} vs {fast_timesteps}"
            )
        if repetition_time < 0:
            raise ValueError(f"repetition_time must be >= 0, got {repetition_time}")
        if dc_cg_steps < 1:
            raise ValueError(f"dc_cg_steps must be >= 1 for Diffusion-QSM teacher DC, got {dc_cg_steps}")
        if dc_lam <= 0:
            raise ValueError(f"dc_lam must be > 0 for Diffusion-QSM teacher DC, got {dc_lam}")
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta}")

        defaults = model_and_diffusion_defaults()
        defaults["learn_sigma"] = True
        defaults["predict_xstart"] = False
        defaults["timestep_respacing"] = ["add", scheduler_space_timesteps(1000, str(int(fast_timesteps)))]

        model, diffusion = create_model_and_diffusion(**defaults)
        state_dict = load_model_state_dict(teacher_ckpt)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.patch_size = tuple(int(v) for v in patch_size)
        self.fast_timesteps = int(fast_timesteps)
        self.starting_point = int(starting_point)
        self.repetition_time = int(repetition_time)
        self.eta = float(eta)
        self.dc_lam = float(dc_lam)
        self.dc_cg_steps = int(dc_cg_steps)
        self.time_pairs = self._build_time_pairs(get_schedule_jump)
        self.reverse_steps = sum(1 for t_last, t_cur in self.time_pairs if t_cur < t_last)

    def _build_time_pairs(self, get_schedule_jump) -> list[tuple[int, int]]:
        t_t = int(self.fast_timesteps)
        t_t2 = int(self.starting_point)
        repetition = int(self.repetition_time) + 1
        times1 = get_schedule_jump(t_t, 1, 1, 1)
        if t_t2 >= 0:
            times1 = times1[: -(t_t2 + 1)]
        times2 = get_schedule_jump(t_t2 + 1, 1, 1, repetition)
        times = times1 + times2
        return [(t_last, t_cur) for t_last, t_cur in zip(times[:-1], times[1:]) if t_last != t_cur]

    def manifest_metadata(self) -> dict:
        return {
            "teacher_backend": self.name,
            "teacher_fast_timesteps": self.fast_timesteps,
            "teacher_starting_point": self.starting_point,
            "teacher_repetition_time": self.repetition_time,
            "teacher_eta": self.eta,
            "teacher_dc_lam": self.dc_lam,
            "teacher_dc_cg_steps": self.dc_cg_steps,
            "teacher_reverse_steps": self.reverse_steps,
        }

    def generate(
        self,
        x_init: torch.Tensor,
        field_patch: torch.Tensor,
        mask_patch: torch.Tensor,
        dipole_kernel: torch.Tensor,
        sample_seed: int,
    ) -> TeacherTrajectoryRecord:
        del x_init  # Diffusion-QSM sampling starts from noise; x_init is still stored in the cache for student training.

        kernel_5d = dipole_kernel
        while kernel_5d.ndim < field_patch.ndim:
            kernel_5d = kernel_5d.unsqueeze(0)

        args1 = {
            "measurement": field_patch,
            "measurement_new": field_patch,
            "dipole_kernel": kernel_5d,
            "msk": mask_patch,
            "calci": torch.ones_like(field_patch),
            "lam": [self.dc_lam for _ in range(max(self.fast_timesteps + 1, 1000))],
            "lam_step": self.dc_cg_steps,
            "p_size": list(self.patch_size),
            "stride": list(self.patch_size),
            "corners": [(0, 0, 0)],
            "x_grid_mask": torch.ones_like(field_patch),
            "time_pairs": self.time_pairs,
            "manual_batching": False,
            "weight_mat": None,
        }

        shape = tuple(field_patch.shape)
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        states: list[torch.Tensor] = []
        state_steps: list[int] = []

        with torch.inference_mode():
            with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []):
                torch.manual_seed(int(sample_seed))
                image_after_step = torch.randn(shape, device=self.device)
                latest_x0 = None
                step_counter = 0

                for t_last, t_cur in self.time_pairs:
                    t_last_t = torch.tensor([t_last] * shape[0], device=self.device)
                    if t_cur < t_last:
                        image_after_step, x0_dc = self.diffusion.ddim_sample_total(
                            self.model,
                            image_after_step,
                            torch.zeros_like(image_after_step),
                            t_last,
                            t_last_t,
                            indices,
                            shape,
                            args1,
                            eta=self.eta,
                            clip_denoised=True,
                            denoised_fn=None,
                            cond_fn=None,
                            model_kwargs=None,
                        )
                        latest_x0 = x0_dc.detach().clone()
                        if not torch.isfinite(latest_x0).all():
                            raise RuntimeError("Diffusion-QSM teacher produced non-finite DC-refined states.")
                        step_counter += 1
                        states.append(latest_x0)
                        state_steps.append(step_counter)
                    elif latest_x0 is not None:
                        image_after_step = self.diffusion.q_sample(
                            latest_x0,
                            t_last_t + 1,
                            torch.randn_like(latest_x0),
                        )

        if not states:
            raise RuntimeError("Diffusion-QSM teacher generated no reverse-step states.")

        return TeacherTrajectoryRecord(
            backend=self.name,
            states=states,
            state_steps=state_steps,
            metadata={"teacher_reverse_steps": step_counter},
        )


def atomic_save_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp_path.replace(path)


def maybe_write_cache_metadata(
    output_dir: Path,
    dipole_kernel: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    metadata = {
        "save_cone_mask": bool(args.save_cone_mask),
        "save_spectral_weight": bool(args.save_spectral_weight),
        "save_teacher_field_residual": bool(args.save_teacher_field_residual),
    }

    if args.save_cone_mask:
        cone_mask = create_cone_mask(dipole_kernel, cone_threshold=args.cache_spectral_cone_threshold)
        cone_mask_file = "cache_cone_mask.npy"
        atomic_save_npy(output_dir / cone_mask_file, cone_mask.detach().cpu().numpy().astype(np.uint8))
        metadata["cone_mask_file"] = cone_mask_file

    if args.save_spectral_weight:
        spectral_weight = build_spectral_weight_map(
            dipole_kernel,
            mode=args.cache_spectral_mode,
            cone_threshold=args.cache_spectral_cone_threshold,
            cone_weight=args.cache_spectral_cone_weight,
            background_weight=args.cache_spectral_background_weight,
        )
        spectral_weight_file = "cache_spectral_weight.npy"
        atomic_save_npy(output_dir / spectral_weight_file, spectral_weight.detach().cpu().numpy().astype(np.float32))
        metadata["spectral_weight_file"] = spectral_weight_file
        metadata["cache_spectral_meta"] = {
            "mode": args.cache_spectral_mode,
            "cone_threshold": float(args.cache_spectral_cone_threshold),
            "cone_weight": float(args.cache_spectral_cone_weight),
            "background_weight": float(args.cache_spectral_background_weight),
        }

    return metadata


def collect_cosmos_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files: list[Path] = []
    split_dirs = [data_dir / "train", data_dir / "test"]
    if any(split_dir.exists() for split_dir in split_dirs):
        roots = [split_dir for split_dir in split_dirs if split_dir.exists()]
    else:
        roots = [data_dir]

    for root in roots:
        files.extend(sorted(root.rglob("cosmos*.nii*")))

    if not files:
        raise FileNotFoundError(f"No cosmos*.nii* files found under: {data_dir}")
    return files


def compute_rank_target(bank_size: int, rank: int, world_size: int) -> int:
    base = bank_size // world_size
    remainder = bank_size % world_size
    return base + (1 if rank < remainder else 0)


def manifest_path_for_rank(output_dir: Path, rank: int) -> Path:
    return output_dir / f"manifest_rank_{rank}.json"


def load_existing_manifest(output_dir: Path, rank: int) -> dict:
    manifest_path = manifest_path_for_rank(output_dir, rank)
    if not manifest_path.exists():
        return {"format": BANK_FORMAT, "rank": rank, "shards": []}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != BANK_FORMAT:
        raise RuntimeError(
            f"Found manifest with incompatible format {manifest.get('format')!r} in {manifest_path}. "
            f"Expected {BANK_FORMAT!r}. Use a fresh output directory."
        )

    valid_shards = []
    for shard in manifest.get("shards", []):
        files = [output_dir / rel_path for rel_path in shard["files"].values()]
        if not all(path.exists() for path in files):
            print(f"[GPU {rank}] Skipping incomplete shard {shard.get('prefix', '<unknown>')}", flush=True)
            continue

        x_init = np.load(output_dir / shard["files"]["x_init"], mmap_mode="r")
        teacher_states = np.load(output_dir / shard["files"]["teacher_states"], mmap_mode="r")
        teacher_state_steps = np.load(output_dir / shard["files"]["teacher_state_steps"], mmap_mode="r")
        count = int(x_init.shape[0])
        if count != int(shard["count"]):
            print(
                f"[GPU {rank}] Skipping shard {shard.get('prefix', '<unknown>')} with mismatched count "
                f"{count} != {shard['count']}",
                flush=True,
            )
            continue
        if teacher_states.shape[0] != count or teacher_state_steps.shape[0] != count:
            print(
                f"[GPU {rank}] Skipping shard {shard.get('prefix', '<unknown>')} due to inconsistent leading dims "
                f"{teacher_states.shape} / {teacher_state_steps.shape} vs count {count}",
                flush=True,
            )
            continue
        if teacher_states.ndim != 5 or teacher_state_steps.ndim != 2:
            print(
                f"[GPU {rank}] Skipping shard {shard.get('prefix', '<unknown>')} due to invalid teacher tensor ranks "
                f"{teacher_states.ndim} / {teacher_state_steps.ndim}",
                flush=True,
            )
            continue
        if teacher_states.shape[1] != teacher_state_steps.shape[1]:
            print(
                f"[GPU {rank}] Skipping shard {shard.get('prefix', '<unknown>')} due to teacher state count mismatch "
                f"{teacher_states.shape[1]} != {teacher_state_steps.shape[1]}",
                flush=True,
            )
            continue
        valid_shards.append(shard)

    manifest["shards"] = valid_shards
    manifest.setdefault("generated_fields", 0)
    return manifest


def ensure_manifest_compatible(manifest: dict, expected: dict) -> None:
    if not manifest.get("shards"):
        return

    mismatches = []
    for key, expected_value in expected.items():
        if key not in manifest:
            continue
        actual_value = manifest[key]
        if actual_value != expected_value:
            mismatches.append(f"{key}: existing={actual_value!r}, expected={expected_value!r}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise RuntimeError(
            "Existing cache manifest is incompatible with the requested configuration. "
            f"Use a fresh output directory or finish the previous run first. Differences: {mismatch_text}"
        )


def check_for_legacy_cache_files(output_dir: Path, rank: int) -> None:
    legacy_patterns = [
        f"samples_rank_{rank}.npy",
        f"samples_rank_{rank}_chk*.npy",
        f"traj_rank_{rank}_shard*.npy",
    ]
    legacy_files = []
    for pattern in legacy_patterns:
        legacy_files.extend(output_dir.glob(pattern))

    if legacy_files and not manifest_path_for_rank(output_dir, rank).exists():
        file_list = ", ".join(path.name for path in sorted(legacy_files))
        raise RuntimeError(
            "Legacy bank/cache files detected in the output directory. Use a fresh output directory for the new "
            f"teacher-refinement cache, or remove the legacy files first: {file_list}"
        )


def load_volume(volume_path: Path, mask_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nibabel is required to load COSMOS volumes. Install it in the runtime environment before cache generation."
        ) from exc

    image = nib.load(str(volume_path))
    volume = image.get_fdata(dtype=np.float32)
    mask = (np.abs(volume) > mask_threshold).astype(np.float32)
    return volume, mask


def save_shard(output_dir: Path, rank: int, shard_index: int, samples: list[dict]) -> dict:
    prefix = f"cache_rank_{rank}_shard{shard_index:04d}"
    file_map = {
        "field": f"{prefix}_field.npy",
        "mask": f"{prefix}_mask.npy",
        "chi_gt": f"{prefix}_chi_gt.npy",
        "x_init": f"{prefix}_x_init.npy",
        "teacher_states": f"{prefix}_teacher_states.npy",
        "teacher_state_steps": f"{prefix}_teacher_state_steps.npy",
        "sample_id": f"{prefix}_sample_id.npy",
        "coords": f"{prefix}_coords.npy",
        "subjects": f"{prefix}_subjects.npy",
    }
    if "teacher_field_residual" in samples[0]:
        file_map["teacher_field_residual"] = f"{prefix}_teacher_field_residual.npy"

    arrays = {
        "field": np.stack([item["field"] for item in samples]).astype(np.float16),
        "mask": np.stack([item["mask"] for item in samples]).astype(np.uint8),
        "chi_gt": np.stack([item["chi_gt"] for item in samples]).astype(np.float16),
        "x_init": np.stack([item["x_init"] for item in samples]).astype(np.float16),
        "teacher_states": np.stack([item["teacher_states"] for item in samples]).astype(np.float16),
        "teacher_state_steps": np.stack([item["teacher_state_steps"] for item in samples]).astype(np.int16),
        "sample_id": np.asarray([item["sample_id"] for item in samples], dtype=np.int64),
        "coords": np.asarray([item["pos"] for item in samples], dtype=np.int16),
        "subjects": np.asarray([item["subject_id"] for item in samples], dtype="U128"),
    }
    if "teacher_field_residual" in samples[0]:
        arrays["teacher_field_residual"] = np.asarray(
            [item["teacher_field_residual"] for item in samples],
            dtype=np.float32,
        )

    for key, rel_path in file_map.items():
        atomic_save_npy(output_dir / rel_path, arrays[key])

    shard_meta = {
        "prefix": prefix,
        "count": int(arrays["sample_id"].shape[0]),
        "format": BANK_FORMAT,
        "files": file_map,
        "teacher_state_count": int(arrays["teacher_states"].shape[1]),
    }
    atomic_save_json(output_dir / f"{prefix}_meta.json", shard_meta)
    return shard_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-ckpt", "--ckpt", dest="student_ckpt", type=str, default=None)
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./teacher_refinement_cache_v1")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--diffusion-root", type=str, default=None)
    parser.add_argument("--bank-size", type=int, default=50000)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--patch-size", type=str, default="64,64,64")
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--mask-threshold", type=float, default=1e-3)
    parser.add_argument("--min-mask-voxels", type=int, default=5000)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=str, default="1,1,1")
    parser.add_argument("--b0-dir", type=str, default="0,0,1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache-size", type=int, default=5)
    parser.add_argument("--bank-mode", choices=["teacher_refinement"], default="teacher_refinement")
    parser.add_argument("--teacher-backend", choices=["auto", "student_cg", "diffusion_qsm"], default="auto")
    parser.add_argument("--saved-teacher-states", type=int, default=4)
    parser.add_argument("--teacher-rollout-steps", type=int, default=4)
    parser.add_argument(
        "--teacher-dc-policy",
        choices=["fixed", "stepwise_lambda", "stepwise_lambda+final_polish"],
        default="fixed",
    )
    parser.add_argument("--teacher-lambda-dc", type=str, default="0.1")
    parser.add_argument("--teacher-cg-steps", type=str, default="8")
    parser.add_argument("--teacher-final-polish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-final-polish-lam", type=float, default=None)
    parser.add_argument("--teacher-final-polish-steps", type=int, default=None)
    parser.add_argument("--max-sigma-factor", type=float, default=2.0)
    parser.add_argument("--x-init-mode", choices=["adjoint", "adjoint_noise"], default="adjoint")
    parser.add_argument("--x-init-noise-std", type=float, default=0.0)
    parser.add_argument("--diffusion-fast-timesteps", type=int, default=200)
    parser.add_argument("--diffusion-starting-point", type=int, default=10)
    parser.add_argument("--diffusion-repetition-time", type=int, default=2)
    parser.add_argument("--diffusion-eta", type=float, default=1.0)
    parser.add_argument("--save-cone-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-spectral-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cache-spectral-mode",
        choices=["none", "cone_weighted", "cone_weighted_residual"],
        default="cone_weighted_residual",
    )
    parser.add_argument("--cache-spectral-cone-threshold", type=float, default=0.05)
    parser.add_argument("--cache-spectral-cone-weight", type=float, default=4.0)
    parser.add_argument("--cache-spectral-background-weight", type=float, default=1.0)
    parser.add_argument("--save-teacher-field-residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def parse_schedule_arg(raw_value: str):
    parts = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Expected a non-empty scalar or comma-separated schedule, got {raw_value!r}")
    if len(parts) == 1:
        value = parts[0]
        try:
            return int(value)
        except ValueError:
            return float(value)
    all_int = all(part.lstrip("-").isdigit() for part in parts)
    if all_int:
        return [int(part) for part in parts]
    return [float(part) for part in parts]


def create_initial_state(
    field_patch: torch.Tensor,
    mask_patch: torch.Tensor,
    dipole_kernel: torch.Tensor,
    init_mode: str,
    init_noise_std: float,
    sample_seed: int,
) -> torch.Tensor:
    with torch.inference_mode():
        x_init = adjoint_field_to_chi(field_patch, dipole_kernel, mask=mask_patch)

    if init_mode == "adjoint":
        return x_init

    if init_mode != "adjoint_noise":
        raise ValueError(f"Unknown x-init mode: {init_mode}")

    if init_noise_std <= 0:
        return x_init

    with torch.random.fork_rng(devices=[field_patch.device.index] if field_patch.is_cuda else []):
        torch.manual_seed(int(sample_seed))
        return x_init + torch.randn_like(x_init) * float(init_noise_std) * mask_patch


def create_teacher_backend(
    args: argparse.Namespace,
    device: torch.device,
    patch_size: tuple[int, int, int],
) -> TeacherBackendBase:
    errors: list[str] = []

    def try_diffusion() -> TeacherBackendBase:
        if not args.teacher_ckpt:
            raise ValueError("--teacher-ckpt is required for teacher-backend=diffusion_qsm")
        return DiffusionQSMTeacherBackend(
            teacher_ckpt=args.teacher_ckpt,
            diffusion_root=args.diffusion_root,
            device=device,
            patch_size=patch_size,
            fast_timesteps=args.diffusion_fast_timesteps,
            starting_point=args.diffusion_starting_point,
            repetition_time=args.diffusion_repetition_time,
            eta=args.diffusion_eta,
            dc_lam=resolve_constant_schedule_arg(args.teacher_lambda_dc, float, "teacher-lambda-dc"),
            dc_cg_steps=resolve_constant_schedule_arg(args.teacher_cg_steps, int, "teacher-cg-steps"),
        )

    def try_student() -> TeacherBackendBase:
        if not args.student_ckpt:
            raise ValueError("--student-ckpt/--ckpt is required for teacher-backend=student_cg")
        return StudentCGTeacherBackend(
            student_ckpt=args.student_ckpt,
            project_root=args.project_root,
            diffusion_root=args.diffusion_root,
            device=device,
            teacher_rollout_steps=args.teacher_rollout_steps,
            teacher_dc_policy=args.teacher_dc_policy,
            teacher_lambda_dc=parse_schedule_arg(args.teacher_lambda_dc),
            teacher_cg_steps=parse_schedule_arg(args.teacher_cg_steps),
            final_polish=args.teacher_final_polish,
            final_polish_lam=args.teacher_final_polish_lam,
            final_polish_steps=args.teacher_final_polish_steps,
            max_sigma_factor=args.max_sigma_factor,
            use_amp=args.amp,
        )

    if args.teacher_backend in {"auto", "diffusion_qsm"}:
        try:
            return try_diffusion()
        except Exception as exc:
            if args.teacher_backend == "diffusion_qsm":
                raise
            errors.append(f"diffusion_qsm unavailable: {exc}")

    if args.teacher_backend in {"auto", "student_cg"}:
        try:
            return try_student()
        except Exception as exc:
            if args.teacher_backend == "student_cg":
                raise
            errors.append(f"student_cg unavailable: {exc}")

    raise RuntimeError("Unable to construct a teacher backend. " + " | ".join(errors))


def main() -> None:
    args = build_parser().parse_args()
    patch_size = tuple(int(v) for v in parse_triplet(args.patch_size, int))
    voxel_size = tuple(float(v) for v in parse_triplet(args.voxel_size, float))
    b0_dir = tuple(float(v) for v in parse_triplet(args.b0_dir, float))

    if args.saved_teacher_states < 1:
        raise ValueError("--saved-teacher-states must be >= 1")
    if args.teacher_rollout_steps < 1:
        raise ValueError("--teacher-rollout-steps must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.cache_size < 1:
        raise ValueError("--cache-size must be >= 1")
    if args.bank_size < 1:
        raise ValueError("--bank-size must be >= 1")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_for_legacy_cache_files(output_dir, args.rank)

    seed_everything(args.seed + args.rank)
    rng = np.random.default_rng(args.seed + args.rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.rank}")
        torch.cuda.set_device(device)
    else:
        if args.rank != 0:
            raise RuntimeError("CPU execution only supports rank 0.")
        device = torch.device("cpu")

    teacher_backend = create_teacher_backend(args, device=device, patch_size=patch_size)

    dipole_kernel = create_dipole_kernel(
        patch_size,
        voxel_size=voxel_size,
        b0_dir=b0_dir,
        device=device,
    )
    cache_metadata = maybe_write_cache_metadata(output_dir, dipole_kernel, args)

    all_files = collect_cosmos_files(Path(args.data_dir).expanduser().resolve())
    my_files = all_files[args.rank::args.world_size]
    if not my_files:
        raise RuntimeError(
            f"Rank {args.rank} has no files assigned. Found {len(all_files)} total files for world size {args.world_size}."
        )

    manifest = load_existing_manifest(output_dir, args.rank)
    target_samples = compute_rank_target(args.bank_size, args.rank, args.world_size)
    completed_samples = sum(int(shard["count"]) for shard in manifest["shards"])
    field_counter = int(manifest.get("generated_fields", 0))

    ensure_manifest_compatible(
        manifest,
        {
            "format": BANK_FORMAT,
            "bank_mode": args.bank_mode,
            "teacher_backend": teacher_backend.name,
            "teacher_backend_meta": teacher_backend.manifest_metadata(),
            "cache_metadata": cache_metadata,
            "patch_size": list(patch_size),
            "voxel_size": list(voxel_size),
            "b0_dir": list(b0_dir),
            "noise_std": args.noise_std,
            "saved_teacher_states": args.saved_teacher_states,
            "x_init_mode": args.x_init_mode,
            "x_init_noise_std": args.x_init_noise_std,
        },
    )

    manifest.update(
        {
            "format": BANK_FORMAT,
            "rank": args.rank,
            "world_size": args.world_size,
            "bank_mode": args.bank_mode,
            "teacher_backend": teacher_backend.name,
            "target_samples": target_samples,
            "generated_fields": field_counter,
            "save_every": args.save_every,
            "patch_size": list(patch_size),
            "voxel_size": list(voxel_size),
            "b0_dir": list(b0_dir),
            "noise_std": args.noise_std,
            "saved_teacher_states": args.saved_teacher_states,
            "x_init_mode": args.x_init_mode,
            "x_init_noise_std": args.x_init_noise_std,
            "shards": manifest.get("shards", []),
            "teacher_backend_meta": teacher_backend.manifest_metadata(),
            "cache_metadata": cache_metadata,
        }
    )
    atomic_save_json(manifest_path_for_rank(output_dir, args.rank), manifest)

    samples_remaining = target_samples - completed_samples
    shard_index = len(manifest["shards"])
    pending_samples: list[dict] = []
    volumes_cache: list[tuple[np.ndarray, np.ndarray, str]] = []
    file_index = 0
    generated_samples = 0
    start_time = time.time()
    last_print = start_time

    print(f"[GPU {args.rank}] Teacher backend: {teacher_backend.name}", flush=True)
    print(f"[GPU {args.rank}] Target samples: {target_samples}", flush=True)
    print(f"[GPU {args.rank}] Existing samples: {completed_samples}", flush=True)

    if completed_samples >= target_samples:
        print(f"[GPU {args.rank}] Already complete. Nothing to do.", flush=True)
        return

    def flush_shard() -> None:
        nonlocal pending_samples, shard_index, manifest
        if not pending_samples:
            return
        shard_meta = save_shard(output_dir, args.rank, shard_index, pending_samples)
        manifest["shards"].append(shard_meta)
        manifest["generated_fields"] = field_counter
        atomic_save_json(manifest_path_for_rank(output_dir, args.rank), manifest)
        completed = completed_samples + generated_samples
        print(
            f"[GPU {args.rank}] Saved {shard_meta['prefix']} with {shard_meta['count']} samples "
            f"(Total: {completed}/{target_samples})",
            flush=True,
        )
        pending_samples = []
        shard_index += 1

    while generated_samples < samples_remaining:
        while len(volumes_cache) < args.cache_size and file_index < len(my_files):
            volume_path = my_files[file_index]
            try:
                volume, mask = load_volume(volume_path, args.mask_threshold)
                volumes_cache.append((volume, mask, volume_path.name))
            except Exception as exc:
                print(f"[GPU {args.rank}] Failed to load {volume_path}: {exc}", flush=True)
            file_index += 1

        if not volumes_cache:
            file_index = 0
            continue

        volume, mask, subject_name = volumes_cache[int(rng.integers(len(volumes_cache)))]
        if any(dim < size for dim, size in zip(volume.shape, patch_size)):
            continue

        d = int(rng.integers(0, volume.shape[0] - patch_size[0] + 1))
        h = int(rng.integers(0, volume.shape[1] - patch_size[1] + 1))
        w = int(rng.integers(0, volume.shape[2] - patch_size[2] + 1))
        slices = (
            slice(d, d + patch_size[0]),
            slice(h, h + patch_size[1]),
            slice(w, w + patch_size[2]),
        )

        chi_patch = volume[slices]
        mask_patch = mask[slices]
        if int(mask_patch.sum()) < args.min_mask_voxels:
            continue

        pos = (d, h, w)
        sample_id = int(args.rank * 10_000_000_000 + field_counter)
        sample_seed = int(args.seed + sample_id)
        field_counter += 1

        chi_patch_t = torch.from_numpy(chi_patch).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask_patch_t = torch.from_numpy(mask_patch).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            field_patch_t = forward_field_from_chi(chi_patch_t, dipole_kernel, mask=mask_patch_t)
            if args.noise_std > 0:
                with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                    torch.manual_seed(sample_seed)
                    field_patch_t = field_patch_t + torch.randn_like(field_patch_t) * args.noise_std * mask_patch_t

            x_init_t = create_initial_state(
                field_patch_t,
                mask_patch_t,
                dipole_kernel,
                init_mode=args.x_init_mode,
                init_noise_std=args.x_init_noise_std,
                sample_seed=sample_seed,
            )

            teacher_record = teacher_backend.generate(
                x_init=x_init_t,
                field_patch=field_patch_t,
                mask_patch=mask_patch_t,
                dipole_kernel=dipole_kernel,
                sample_seed=sample_seed,
            )

        full_states = [x_init_t.detach().clone()]
        full_state_steps = [0]
        full_states.extend(teacher_record.states)
        full_state_steps.extend(int(step) for step in teacher_record.state_steps)

        if len(full_states) < 2:
            print(f"[GPU {args.rank}] Sample {sample_id} yielded no teacher states, skipping.", flush=True)
            continue

        selected_indices = select_teacher_state_indices(
            num_available_states=len(full_states),
            num_requested_states=args.saved_teacher_states,
        )
        selected_states = [
            full_states[index].squeeze(0).squeeze(0).cpu().numpy().astype(np.float16) for index in selected_indices
        ]
        selected_steps = [full_state_steps[index] for index in selected_indices]

        sample = {
            "field": field_patch_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float16),
            "mask": mask_patch.astype(np.uint8),
            "chi_gt": chi_patch.astype(np.float16),
            "x_init": x_init_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float16),
            "teacher_states": np.stack(selected_states).astype(np.float16),
            "teacher_state_steps": np.asarray(selected_steps, dtype=np.int16),
            "sample_id": sample_id,
            "pos": pos,
            "subject_id": subject_name,
        }
        if args.save_teacher_field_residual:
            teacher_final_field = forward_field_from_chi(full_states[-1], dipole_kernel, mask=mask_patch_t)
            sample["teacher_field_residual"] = float(
                masked_mse(teacher_final_field, field_patch_t, mask_patch_t).detach().cpu()
            )
        pending_samples.append(sample)
        generated_samples += 1

        if len(pending_samples) >= args.save_every:
            flush_shard()

        if generated_samples and (generated_samples % 100 == 0 or time.time() - last_print > 10):
            elapsed = max(time.time() - start_time, 1e-6)
            rate = generated_samples / elapsed
            completed = completed_samples + generated_samples
            print(
                f"[GPU {args.rank}] {completed}/{target_samples} ({100.0 * completed / target_samples:.1f}%) "
                f"- {rate:.2f} samples/sec",
                flush=True,
            )
            last_print = time.time()

    flush_shard()

    elapsed = time.time() - start_time
    manifest["generated_fields"] = field_counter
    manifest["generated_samples"] = sum(int(shard["count"]) for shard in manifest["shards"])
    atomic_save_json(manifest_path_for_rank(output_dir, args.rank), manifest)

    print(
        f"[GPU {args.rank}] DONE. Generated {generated_samples} new samples, "
        f"{manifest['generated_samples']} total samples recorded. Time: {elapsed / 60.0:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
