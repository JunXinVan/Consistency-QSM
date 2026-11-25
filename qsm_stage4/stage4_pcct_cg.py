#!/usr/bin/env python3
"""
Stage 4 v3 QSM specialization training:
GT + physics + EMA-style consistency, with optional weak Diffusion-QSM endpoint regularization.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, DistributedSampler
from tqdm import tqdm

from qsm_bank_utils import (
    ConditioningSpec,
    StepAdaptiveStudentModel,
    build_conditioning_spec,
    build_dc_rollout_policy,
    build_spectral_weight_map,
    conditioning_channels_for_mode,
    create_dipole_kernel,
    expected_base_model_in_channels,
    forward_field_from_chi,
    geometric_sigma_schedule,
    maybe_expand_input_conv_weight,
    load_student_components,
    masked_frequency_l2,
    masked_gradient_pseudo_huber_loss,
    masked_hybrid_frequency_dipole_loss,
    masked_mse,
    parse_int_list,
    parse_triplet,
    pseudo_huber_loss,
    run_refinement_rollout,
    run_residual_delta_refinement_rollout,
    sigma_weight,
)


CACHE_FORMAT_BY_BANK_MODE = {
    "gt_core": "qsm_stage4_v3_cache",
    "reconstruction_faithful": "qsm_stage4_v6_faithful_cache",
}
SUPPORTED_BANK_MODES = tuple(CACHE_FORMAT_BY_BANK_MODE.keys())
DEFAULT_STAGE4_OUTPUT_DIR = "./runs/stage4_v3"
RESUME_ALLOWED_OVERRIDE_DESTS = {
    "resume_ckpt",
    "output_dir",
    "rank",
    "local_rank",
    "world_size",
    "eval_every",
    "eval_cache_dir",
    "debug_heartbeat",
}
RESUME_REQUIRED_STATE_KEYS = {
    "epoch",
    "args",
    "effective_config",
    "model_state",
    "ema_state",
    "optimizer_state",
    "global_optimizer_step",
}


def resolve_saved_arg(saved_args: dict, key: str, fallback):
    value = saved_args.get(key, fallback)
    return fallback if value is None else value


def parse_int_list_maybe(value) -> list[int]:
    if isinstance(value, str):
        return parse_int_list(value)
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    raise ValueError(f"Expected string/list integer schedule, got {type(value).__name__}: {value!r}")


def parse_float_list_maybe(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    raise ValueError(f"Expected string/list float schedule, got {type(value).__name__}: {value!r}")


def parse_triplet_maybe(value, cast=float) -> tuple:
    if isinstance(value, str):
        return parse_triplet(value, cast)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(cast(part) for part in value)
    raise ValueError(f"Expected string/list triplet, got {type(value).__name__}: {value!r}")


def resolve_optional_path_str(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    return str(Path(path_value).expanduser().resolve())


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
        if dest:
            provided.add(dest)
    return provided


def capture_local_rng_state(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_local_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def validate_resume_checkpoint_payload(ckpt: dict, ckpt_path: Path) -> None:
    missing_keys = sorted(key for key in RESUME_REQUIRED_STATE_KEYS if key not in ckpt)
    if missing_keys:
        raise RuntimeError(
            f"{ckpt_path} does not contain a full Stage4 resume payload (missing={missing_keys}). "
            "Use --init-ckpt for warm-starting old checkpoints instead."
        )

    if ckpt.get("model_state") is None or ckpt.get("ema_state") is None:
        raise RuntimeError(f"{ckpt_path} is missing model/ema state required for exact resume. Use --init-ckpt instead.")
    if ckpt.get("optimizer_state") is None:
        raise RuntimeError(f"{ckpt_path} is missing optimizer_state required for exact resume. Use --init-ckpt instead.")
    if int(ckpt.get("global_optimizer_step", -1)) < 0:
        raise RuntimeError(f"{ckpt_path} is missing a valid global_optimizer_step. Use --init-ckpt instead.")
    if int(ckpt.get("epoch", -1)) < 0:
        raise RuntimeError(f"{ckpt_path} is missing a valid saved epoch. Use --init-ckpt instead.")
    if "rng_state_by_rank" not in ckpt and "rng_state" not in ckpt:
        raise RuntimeError(f"{ckpt_path} is missing RNG state required for exact resume. Use --init-ckpt instead.")

    saved_args = ckpt.get("args")
    if not isinstance(saved_args, dict) or not saved_args:
        raise RuntimeError(f"{ckpt_path} is missing saved args required for exact resume. Use --init-ckpt instead.")

    lr_schedule = saved_args.get("lr_schedule", "manual_epoch")
    if lr_schedule == "warmup_cosine" and "scheduler_state" not in ckpt:
        raise RuntimeError(
            f"{ckpt_path} is missing scheduler_state required to exactly resume warmup_cosine scheduling."
        )
    if bool(saved_args.get("amp", False)) and "scaler_state" not in ckpt:
        raise RuntimeError(f"{ckpt_path} is missing scaler_state required to exactly resume AMP training.")


def merge_resume_args(
    parser: argparse.ArgumentParser,
    cli_args: argparse.Namespace,
    raw_argv: list[str],
) -> argparse.Namespace:
    provided_dests = extract_provided_cli_dests(parser, raw_argv)
    illegal_dests = sorted(dest for dest in provided_dests if dest not in RESUME_ALLOWED_OVERRIDE_DESTS)
    if illegal_dests:
        raise ValueError(
            "Resume checkpoints make saved training/data semantics authoritative. "
            f"These CLI overrides are not allowed with --resume-ckpt: {illegal_dests}"
        )

    resume_path = Path(cli_args.resume_ckpt).expanduser().resolve()
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint_payload(ckpt, resume_path)

    saved_args = dict(ckpt.get("args", {}))
    effective_config = dict(ckpt.get("effective_config", {}))
    parser_dests = {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != "help"
    }

    merged = dict(vars(cli_args))
    for dest in parser_dests:
        if dest in saved_args:
            merged[dest] = saved_args[dest]

    resume_output_dir = str(resume_path.parent)
    if cli_args.output_dir is not None:
        requested_output_dir = str(Path(cli_args.output_dir).expanduser().resolve())
        if requested_output_dir != resume_output_dir:
            raise ValueError(
                f"--output-dir must be omitted or match the checkpoint run directory when resuming. "
                f"expected={resume_output_dir}, got={requested_output_dir}"
            )
    merged["output_dir"] = resume_output_dir

    for dest in RESUME_ALLOWED_OVERRIDE_DESTS:
        if dest in provided_dests and dest not in {"resume_ckpt", "output_dir"}:
            merged[dest] = getattr(cli_args, dest)

    merged["resume_ckpt"] = str(resume_path)
    merged["init_ckpt"] = saved_args.get("init_ckpt", effective_config.get("init_ckpt", cli_args.init_ckpt))
    merged["provided_cli_dests"] = sorted(provided_dests)
    merged["resume_expected_output_dir"] = resume_output_dir
    merged["resume_restore_epoch"] = int(ckpt["epoch"])
    merged["resume_restore_global_optimizer_step"] = int(ckpt["global_optimizer_step"])
    return argparse.Namespace(**merged)


def detect_stage4_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(
        key.startswith("base_model.")
        or key.startswith("adapter.")
        or key.startswith("conditioning_stem.")
        for key in state_dict.keys()
    )


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


class Stage4V3Dataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        cache_shards: int = 1,
        expected_bank_mode: str = "gt_core",
        filter_invalid: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_shards = max(int(cache_shards), 1)
        self.expected_bank_mode = str(expected_bank_mode)
        self.filter_invalid = bool(filter_invalid)
        self.cache_metadata: dict = {}
        self.shards = self._discover_shards()
        if not self.shards:
            raise FileNotFoundError(
                f"No compatible Stage 4 cache shards found under {self.cache_dir} for bank_mode={self.expected_bank_mode!r}"
            )

        has_teacher_flags = {bool(shard["has_teacher_endpoint"]) for shard in self.shards}
        if len(has_teacher_flags) != 1:
            raise RuntimeError(f"Inconsistent teacher_endpoint presence across shards: {sorted(has_teacher_flags)}")
        self.has_teacher_endpoint = next(iter(has_teacher_flags))

        if self.filter_invalid:
            self.valid_indices = self._build_valid_index_map()
            self.cumulative_counts = np.array([len(self.valid_indices)], dtype=np.int64)
        else:
            self.valid_indices = None
            self.cumulative_counts = np.cumsum([shard["count"] for shard in self.shards]).astype(np.int64)

        self.hard_score_sidecar_count = 0
        self.has_any_hard_score_sidecar = False
        self.has_complete_hard_score_sidecars = False
        self.hard_scores = self._load_hard_scores()
        self.cache: OrderedDict[str, object] = OrderedDict()

    def _discover_shards(self) -> list[dict]:
        shards: list[dict] = []
        expected_format = CACHE_FORMAT_BY_BANK_MODE[self.expected_bank_mode]
        manifest_files = sorted(self.cache_dir.glob("manifest_rank_*.json"))
        for manifest_path in manifest_files:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != expected_format:
                continue
            manifest_bank_mode = manifest.get("bank_mode")
            if manifest_bank_mode is not None and manifest_bank_mode != self.expected_bank_mode:
                raise RuntimeError(
                    f"Incompatible bank_mode {manifest_bank_mode!r} in {manifest_path}. "
                    f"Expected {self.expected_bank_mode!r}."
                )

            manifest_cache_metadata = dict(manifest.get("cache_metadata", {}))
            if not self.cache_metadata:
                self.cache_metadata = manifest_cache_metadata
            elif manifest_cache_metadata != self.cache_metadata:
                raise RuntimeError(
                    f"Inconsistent cache_metadata across manifests under {self.cache_dir}: "
                    f"{self.cache_metadata!r} vs {manifest_cache_metadata!r}"
                )

            for shard in manifest.get("shards", []):
                shard_files = {key: self.cache_dir / rel_path for key, rel_path in shard["files"].items()}
                field_key = "field_input" if "field_input" in shard_files else "field"
                required = {field_key, "mask", "chi_gt", "x_init", "sample_id"}
                if not required.issubset(shard_files):
                    continue
                if not all(path.exists() for path in shard_files.values()):
                    continue
                shards.append(
                    {
                        "key": str(self.cache_dir / shard["prefix"]),
                        "prefix": shard["prefix"],
                        "count": int(shard["count"]),
                        "format": manifest.get("format"),
                        "bank_mode": manifest_bank_mode,
                        "field_key": field_key,
                        "has_teacher_endpoint": bool(shard.get("has_teacher_endpoint", "teacher_endpoint" in shard_files)),
                        "files": shard_files,
                    }
                )
        return shards

    def _load_hard_scores(self) -> np.ndarray:
        shard_scores: list[np.ndarray] = []
        has_any_sidecar = False
        for shard in self.shards:
            sidecar_path = self.cache_dir / f"{shard['prefix']}_hard_score.npy"
            if sidecar_path.exists():
                scores = np.load(sidecar_path, mmap_mode="r")
                if int(scores.shape[0]) != int(shard["count"]):
                    raise RuntimeError(
                        f"Hard-score sidecar {sidecar_path} has leading dim {scores.shape[0]}, "
                        f"expected {shard['count']}"
                    )
                shard_scores.append(np.asarray(scores, dtype=np.float32))
                has_any_sidecar = True
                self.hard_score_sidecar_count += 1
            else:
                shard_scores.append(np.zeros(int(shard["count"]), dtype=np.float32))

        self.has_any_hard_score_sidecar = has_any_sidecar
        self.has_complete_hard_score_sidecars = self.hard_score_sidecar_count == len(self.shards)
        if not has_any_sidecar:
            scores = np.zeros(int(self.cumulative_counts[-1]), dtype=np.float32)
        else:
            scores = np.concatenate(shard_scores, axis=0).astype(np.float32)

        if self.valid_indices is None:
            return scores

        filtered = np.empty(len(self.valid_indices), dtype=np.float32)
        shard_offsets = np.cumsum([0] + [int(shard["count"]) for shard in self.shards[:-1]]).astype(np.int64)
        for dataset_idx, (shard_idx, local_idx) in enumerate(self.valid_indices):
            filtered[dataset_idx] = scores[int(shard_offsets[shard_idx] + local_idx)]
        return filtered

    def _build_valid_index_map(self) -> list[tuple[int, int]]:
        valid_indices: list[tuple[int, int]] = []
        invalid_count = 0
        total_count = 0
        print("[Dataset] Scanning cache for invalid samples...", flush=True)

        for shard_idx, shard in enumerate(self.shards):
            arrays = {name: np.load(path, mmap_mode="r") for name, path in shard["files"].items()}
            fields_to_check = [shard["field_key"], "mask", "chi_gt", "x_init"]
            if "field_clean" in arrays:
                fields_to_check.append("field_clean")
            if "cone_risk" in arrays:
                fields_to_check.append("cone_risk")
            if shard["has_teacher_endpoint"]:
                fields_to_check.append("teacher_endpoint")

            for local_idx in range(shard["count"]):
                total_count += 1
                is_valid = True
                for field_name in fields_to_check:
                    sample = np.asarray(arrays[field_name][local_idx])
                    if not np.isfinite(sample).all():
                        is_valid = False
                        break
                if is_valid:
                    valid_indices.append((shard_idx, local_idx))
                else:
                    invalid_count += 1

        if invalid_count > 0:
            print(
                f"[Dataset] WARNING: Found {invalid_count}/{total_count} ({100 * invalid_count / total_count:.2f}%) "
                "invalid samples",
                flush=True,
            )
            print(f"[Dataset] Using {len(valid_indices)} valid samples for training", flush=True)
        else:
            print(f"[Dataset] All {total_count} samples are valid", flush=True)

        return valid_indices

    def __len__(self) -> int:
        return int(self.cumulative_counts[-1])

    def _load_shard(self, shard: dict):
        cache_key = shard["key"]
        if cache_key in self.cache:
            payload = self.cache.pop(cache_key)
            self.cache[cache_key] = payload
            return payload

        payload = {key: np.load(path, mmap_mode="r") for key, path in shard["files"].items()}
        self.cache[cache_key] = payload
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int):
        if self.valid_indices is not None:
            if index < 0 or index >= len(self.valid_indices):
                raise IndexError(f"Index {index} out of range for {len(self.valid_indices)} valid samples")
            shard_index, local_index = self.valid_indices[index]
            shard = self.shards[shard_index]
        else:
            shard_index = int(bisect_right(self.cumulative_counts, index))
            shard_start = 0 if shard_index == 0 else int(self.cumulative_counts[shard_index - 1])
            local_index = int(index - shard_start)
            shard = self.shards[shard_index]

        payload = self._load_shard(shard)
        field_key = shard["field_key"]
        item = {
            "field": torch.from_numpy(np.array(payload[field_key][local_index], dtype=np.float32, copy=True)).unsqueeze(0),
            "mask": torch.from_numpy(np.array(payload["mask"][local_index], dtype=np.float32, copy=True)).unsqueeze(0),
            "chi_gt": torch.from_numpy(np.array(payload["chi_gt"][local_index], dtype=np.float32, copy=True)).unsqueeze(0),
            "x_init": torch.from_numpy(np.array(payload["x_init"][local_index], dtype=np.float32, copy=True)).unsqueeze(0),
            "sample_id": torch.tensor(int(payload["sample_id"][local_index]), dtype=torch.int64),
            "hard_score": torch.tensor(float(self.hard_scores[index]), dtype=torch.float32),
        }
        if "field_clean" in payload:
            item["field_clean"] = torch.from_numpy(
                np.array(payload["field_clean"][local_index], dtype=np.float32, copy=True)
            ).unsqueeze(0)
        if "cone_risk" in payload:
            item["cone_risk"] = torch.from_numpy(
                np.array(payload["cone_risk"][local_index], dtype=np.float32, copy=True)
            ).unsqueeze(0)
        if shard["has_teacher_endpoint"]:
            item["teacher_endpoint"] = torch.from_numpy(
                np.array(payload["teacher_endpoint"][local_index], dtype=np.float32, copy=True)
            ).unsqueeze(0)
        return item


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: Sequence[float],
        num_replicas: int = 1,
        rank: int = 0,
        replacement: bool = True,
        seed: int = 2026,
        curriculum_alpha: float = 1.0,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape {tuple(self.weights.shape)}")
        self.dataset_size = int(self.weights.numel())
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.curriculum_alpha = float(curriculum_alpha)
        self.num_samples = int(math.ceil(self.dataset_size / max(self.num_replicas, 1)))
        self.total_size = self.num_samples * max(self.num_replicas, 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_curriculum_alpha(self, alpha: float) -> None:
        self.curriculum_alpha = float(min(max(alpha, 0.0), 1.0))

    def _effective_weights(self) -> torch.Tensor:
        scores = self.weights
        if scores.numel() == 0:
            return scores
        max_score = float(scores.max().item()) if scores.numel() else 0.0
        normalized = scores / max(max_score, 1e-8) if max_score > 0 else torch.zeros_like(scores)
        alpha = float(self.curriculum_alpha)
        effective = alpha + (1.0 - alpha) * normalized
        if not torch.isfinite(effective).all() or float(effective.sum().item()) <= 0:
            effective = torch.ones_like(scores)
        return effective

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        effective = self._effective_weights()
        sampled = torch.multinomial(
            effective,
            num_samples=self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        rank_indices = sampled[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


class Stage4SpecializationTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.current_epoch = 0
        self.global_optimizer_step = 0
        self.start_epoch = 0
        self.stop_requested = False
        self.run_mode = "resume" if self.args.resume_ckpt else ("random_init" if self.args.random_init else "fresh_init")
        self.resume_checkpoint = Path(self.args.resume_ckpt).expanduser().resolve() if self.args.resume_ckpt else None
        self.restored_epoch = None
        self.restored_global_optimizer_step = None
        self.resume_payload = None
        self.output_dir = Path(self.args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_distributed()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.write_run_registry()

    def setup_distributed(self) -> None:
        self.rank = int(self.args.rank if self.args.rank is not None else 0)
        self.local_rank = int(self.args.local_rank if self.args.local_rank is not None else 0)
        self.world_size = int(self.args.world_size if self.args.world_size is not None else 1)

        self.rank = int(os.environ.get("RANK", self.rank))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.local_rank))
        self.world_size = int(os.environ.get("WORLD_SIZE", self.world_size))

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
            self.local_rank = 0

        if self.world_size > 1 and not dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)

        self.use_amp = bool(self.args.amp and self.device.type == "cuda")

        if self.rank == 0:
            print("=" * 72)
            print("QSM Stage 4 v3 Specialization Training")
            print("=" * 72)
            print(f"World size: {self.world_size}")
            print(f"Device: {self.device}")
            print(f"Train NFE schedule: {self.args.train_nfe_schedule}")
            print(f"Inference NFE: {self.args.infer_nfe}")
            print(f"Batch size per process: {self.args.batch_size}")
            print(f"Gradient accumulation: {self.args.grad_accum_steps}")
            print(f"Effective batch size: {self.args.batch_size * self.world_size * self.args.grad_accum_steps}")

    def setup_model(self) -> None:
        build_student_unet, StudentUNetConfig, get_scalings_for_boundary_condition = load_student_components(
            project_root=self.args.project_root,
            diffusion_root=self.args.diffusion_root,
            script_path=__file__,
        )
        self.get_scalings_for_boundary_condition = get_scalings_for_boundary_condition
        checkpoint_path = self.resume_checkpoint or (
            Path(self.args.init_ckpt).expanduser().resolve() if self.args.init_ckpt else None
        )
        ckpt = None
        saved_args = {}
        state_dict = None
        is_stage4 = False
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if self.run_mode == "resume":
                validate_resume_checkpoint_payload(ckpt, checkpoint_path)
                self.resume_payload = ckpt
            saved_args = dict(ckpt.get("args", {}))
            if self.run_mode == "resume":
                state_dict = ckpt.get("model_state")
            else:
                state_dict = ckpt.get("ema_state")
                if state_dict is None:
                    state_dict = ckpt.get("model_state", ckpt)
            is_stage4 = detect_stage4_checkpoint(state_dict)
        self.saved_args_present = bool(saved_args)
        effective_step_adapter_mode = (
            resolve_saved_arg(saved_args, "step_adapter_mode", self.args.step_adapter_mode)
            if is_stage4
            else self.args.step_adapter_mode
        )
        effective_step_adapter_hidden_dim = int(
            resolve_saved_arg(saved_args, "step_adapter_hidden_dim", self.args.step_adapter_hidden_dim)
            if is_stage4
            else self.args.step_adapter_hidden_dim
        )
        self.step_adapter_overridden_by_checkpoint = bool(
            is_stage4
            and (
                effective_step_adapter_mode != self.args.step_adapter_mode
                or effective_step_adapter_hidden_dim != int(self.args.step_adapter_hidden_dim)
            )
        )
        effective_model_dropout = float(
            resolve_saved_arg(saved_args, "model_dropout", self.args.model_dropout)
            if is_stage4
            else self.args.model_dropout
        )
        self.model_dropout_overridden_by_checkpoint = bool(
            is_stage4
            and not math.isclose(
                effective_model_dropout,
                float(self.args.model_dropout),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        effective_conditioning_mode = (
            resolve_saved_arg(saved_args, "conditioning_mode", self.args.conditioning_mode)
            if is_stage4
            else self.args.conditioning_mode
        )
        effective_conditioning_hidden_dim = int(
            resolve_saved_arg(saved_args, "conditioning_hidden_dim", self.args.conditioning_hidden_dim)
            if is_stage4
            else self.args.conditioning_hidden_dim
        )
        effective_conditioning_norm_mode = (
            resolve_saved_arg(saved_args, "conditioning_norm_mode", self.args.conditioning_norm_mode)
            if is_stage4
            else self.args.conditioning_norm_mode
        )
        effective_conditioning_cone_threshold = float(
            resolve_saved_arg(saved_args, "conditioning_cone_threshold", self.args.conditioning_cone_threshold)
            if is_stage4
            else self.args.conditioning_cone_threshold
        )
        effective_init_error_proxy_sigma = float(
            resolve_saved_arg(saved_args, "init_error_proxy_sigma", self.args.init_error_proxy_sigma)
            if is_stage4
            else self.args.init_error_proxy_sigma
        )
        self.conditioning_overridden_by_checkpoint = bool(
            is_stage4
            and (
                effective_conditioning_mode != self.args.conditioning_mode
                or effective_conditioning_hidden_dim != int(self.args.conditioning_hidden_dim)
                or effective_conditioning_norm_mode != self.args.conditioning_norm_mode
                or abs(effective_conditioning_cone_threshold - float(self.args.conditioning_cone_threshold)) > 1e-12
                or abs(effective_init_error_proxy_sigma - float(self.args.init_error_proxy_sigma)) > 1e-12
            )
        )
        self.conditioning_channels = conditioning_channels_for_mode(effective_conditioning_mode)
        base_model_in_channels = expected_base_model_in_channels(
            effective_conditioning_mode,
            conditioning_channels=self.conditioning_channels,
        )
        base_model = build_student_unet(
            StudentUNetConfig(
                dropout=effective_model_dropout,
                use_checkpoint=bool(getattr(self.args, "student_use_checkpoint", False)),
            ),
            in_channels=base_model_in_channels,
            out_channels=2,
        )

        config = ckpt.get("config", {}) if ckpt is not None else {}
        self.sigma_data = float(ckpt.get("sigma_data", config.get("sigma_data", 0.5)) if ckpt is not None else 0.5)
        self.sigma_min = float(ckpt.get("sigma_min", 0.01) if ckpt is not None else 0.01)
        if self.run_mode == "resume":
            self.sigma_max = float(ckpt.get("sigma_max", max(self.sigma_data * self.args.student_sigma_max_factor, self.sigma_min * 1.01)))
        else:
            self.sigma_max = max(self.sigma_data * self.args.student_sigma_max_factor, self.sigma_min * 1.01)

        if self.rank == 0 and is_stage4 and self.model_dropout_overridden_by_checkpoint:
            print(f"[Init] Using model dropout from Stage4 checkpoint: {effective_model_dropout:.4f}")
        if self.rank == 0:
            print(f"[Init] Student model dropout: {effective_model_dropout:.4f}")
        self.effective_model_dropout = effective_model_dropout

        student_model = StepAdaptiveStudentModel(
            base_model,
            mode=effective_step_adapter_mode,
            hidden_dim=effective_step_adapter_hidden_dim,
            conditioning_mode=effective_conditioning_mode,
            conditioning_channels=self.conditioning_channels,
            conditioning_hidden_dim=effective_conditioning_hidden_dim,
        ).to(self.device)

        if is_stage4:
            has_conditioning_in_ckpt = any(key.startswith("conditioning_stem.") for key in state_dict.keys())
            load_stage4_wrapper_state(
                student_model,
                state_dict,
                allow_missing_conditioning_stem=bool(
                    student_model.conditioning_stem is not None and not has_conditioning_in_ckpt
                ),
            )
            self.init_checkpoint_kind = "stage4_resume" if self.run_mode == "resume" else "stage4"
        elif checkpoint_path is not None:
            if self.run_mode == "resume":
                raise RuntimeError(
                    f"{checkpoint_path} is not a Stage4 checkpoint and cannot be used with --resume-ckpt. "
                    "Use --init-ckpt for warm-starting stage3/base checkpoints."
                )
            state_dict = maybe_expand_input_conv_weight(
                state_dict,
                weight_key="input_blocks.0.0.weight",
                target_in_channels=base_model_in_channels,
            )
            base_model.load_state_dict(state_dict, strict=True)
            self.init_checkpoint_kind = "stage3_or_base_student"
        else:
            self.init_checkpoint_kind = "random_init"
        self.effective_step_adapter_mode = str(effective_step_adapter_mode)
        self.effective_step_adapter_hidden_dim = int(effective_step_adapter_hidden_dim)
        self.effective_conditioning_mode = str(effective_conditioning_mode)
        self.effective_conditioning_hidden_dim = int(effective_conditioning_hidden_dim)
        self.effective_conditioning_norm_mode = str(effective_conditioning_norm_mode)
        self.effective_conditioning_cone_threshold = float(effective_conditioning_cone_threshold)
        self.effective_init_error_proxy_sigma = float(effective_init_error_proxy_sigma)
        self.conditioning_active = bool(self.effective_conditioning_mode != "none")
        self.effective_base_model_in_channels = int(base_model_in_channels)

        self.ema_model = copy.deepcopy(student_model).to(self.device)
        if self.run_mode == "resume":
            load_stage4_wrapper_state(
                self.ema_model,
                ckpt["ema_state"],
                allow_missing_conditioning_stem=False,
            )
        self.ema_model.eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

        self.model = student_model
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                # Historical Stage 4 runs repeatedly logged that no parameters were
                # actually unused under DDP. Keeping this enabled only adds an extra
                # autograd traversal and burns throughput without scientific benefit.
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        self.scaler = GradScaler(enabled=self.use_amp)
        self.dipole_kernel = create_dipole_kernel(
            self.args.patch_shape,
            voxel_size=self.args.voxel_size,
            b0_dir=self.args.b0_dir,
            device=self.device,
        )

        self.spectral_weight_map = None
        if self.args.spectral_loss_mode != "none":
            self.spectral_weight_map = build_spectral_weight_map(
                self.dipole_kernel,
                mode=self.args.spectral_loss_mode,
                cone_threshold=self.args.spectral_cone_threshold,
                cone_weight=self.args.spectral_cone_weight,
                background_weight=self.args.spectral_background_weight,
            )

        self.teacher_aux_weight_map = None
        if self.args.teacher_aux_mode == "cone_weighted":
            self.teacher_aux_weight_map = build_spectral_weight_map(
                self.dipole_kernel,
                mode="cone_weighted",
                cone_threshold=self.args.teacher_aux_cone_threshold,
                cone_weight=self.args.teacher_aux_cone_weight,
                background_weight=self.args.teacher_aux_background_weight,
            )

        if self.rank == 0:
            param_count = sum(param.numel() for param in self.unwrap_model().parameters()) / 1e6
            adapter_param_count = getattr(self.unwrap_model(), "adapter_parameter_count", lambda: 0)()
            print(
                f"\nModel loaded. sigma_data={self.sigma_data:.4f}, sigma_min={self.sigma_min:.4f}, "
                f"sigma_max={self.sigma_max:.4f}"
            )
            print(f"Run mode: {self.run_mode}")
            print(f"Init checkpoint kind: {self.init_checkpoint_kind}")
            print(f"Checkpoint path: {checkpoint_path if checkpoint_path is not None else 'None (random init)'}")
            print(
                f"Effective adapter: mode={self.effective_step_adapter_mode}, "
                f"hidden_dim={self.effective_step_adapter_hidden_dim}"
            )
            print(
                f"Effective conditioning: mode={self.effective_conditioning_mode}, "
                f"channels={self.conditioning_channels}, hidden_dim={self.effective_conditioning_hidden_dim}, "
                f"norm={self.effective_conditioning_norm_mode}, "
                f"cone_threshold={self.effective_conditioning_cone_threshold}, "
                f"init_proxy_sigma={self.effective_init_error_proxy_sigma}, "
                f"base_in={base_model_in_channels}"
            )
            if self.step_adapter_overridden_by_checkpoint:
                print(
                    "WARNING: stage4 checkpoint saved adapter settings override CLI values: "
                    f"cli=({self.args.step_adapter_mode}, {self.args.step_adapter_hidden_dim}) -> "
                    f"effective=({self.effective_step_adapter_mode}, {self.effective_step_adapter_hidden_dim})"
                )
            if self.conditioning_overridden_by_checkpoint:
                print(
                    "WARNING: stage4 checkpoint saved conditioning settings override CLI values: "
                    f"cli=({self.args.conditioning_mode}, {self.args.conditioning_hidden_dim}, "
                    f"{self.args.conditioning_norm_mode}, {self.args.conditioning_cone_threshold}, "
                    f"{self.args.init_error_proxy_sigma}) -> "
                    f"effective=({self.effective_conditioning_mode}, {self.effective_conditioning_hidden_dim}, "
                    f"{self.effective_conditioning_norm_mode}, {self.effective_conditioning_cone_threshold}, "
                    f"{self.effective_init_error_proxy_sigma})"
                )
            print(f"Parameters: {param_count:.1f}M")
            print(f"Step adapter mode: {self.effective_step_adapter_mode} ({adapter_param_count} params)")
            if getattr(self.args, "loss_gt_mse_weight", 0.0) > 0:
                print(
                    f"GT auxiliary MSE: weight={float(self.args.loss_gt_mse_weight):.4f} "
                    "(added on top of pseudo-Huber GT loss)"
                )
            if self.args.spectral_loss_mode != "none" and self.args.loss_spectral_weight > 0:
                print(
                    "Spectral loss: "
                    f"mode={self.args.spectral_loss_mode}, "
                    f"weight={self.args.loss_spectral_weight}, "
                    f"chi_w={self.args.spectral_chi_weight}, "
                    f"dipole_w={self.args.spectral_dipole_weight}, "
                    f"residual_power={self.args.spectral_residual_power}, "
                    f"warmup={self.args.spectral_warmup_fraction} ({self.args.spectral_warmup_schedule})"
                )
            if self.args.curriculum_enabled:
                print(
                    "Curriculum sampling: "
                    f"start_epoch={self.args.curriculum_start_epoch + 1}, "
                    f"alpha={self.args.curriculum_alpha:.3f}, "
                    f"active_dataset={self.curriculum_active if hasattr(self, 'curriculum_active') else 'pending'}"
                )
            else:
                print("Curriculum sampling: disabled")
            if self.args.phase2_enabled:
                print(
                    "Phase2 schedule: "
                    f"start_epoch={self.args.phase2_start_epoch + 1}, "
                    f"ct={self.args.phase2_loss_ct_weight}, "
                    f"spectral={self.args.phase2_loss_spectral_weight}, "
                    f"lr={self.args.phase2_lr}"
                )
            else:
                print("Phase2 schedule: disabled")
            if self.args.loss_ct_weight_schedule:
                print(
                    "CT weight schedule: "
                    + ", ".join(f"epoch{idx + 1}={value:g}" for idx, value in enumerate(self.args.loss_ct_weight_schedule))
                )
            if self.args.loss_traj_weight > 0 and self.args.rank in (None, 0):
                print("[Stage4 v3] Ignoring deprecated --loss-traj-weight; v3 uses EMA consistency instead.")
            if self.args.teacher_target_mode != "trajectory":
                print("[Stage4 v3] Ignoring deprecated --teacher-target-mode; v3 does not use teacher trajectories.")
            if self.args.replay_mode != "none" or self.args.loss_replay_weight > 0:
                print("[Stage4 v3] Ignoring replay settings in v3 mainline.")

    def setup_data(self) -> None:
        self.dataset = Stage4V3Dataset(
            self.args.cache_dir,
            cache_shards=self.args.cache_shards,
            expected_bank_mode=self.args.bank_mode,
        )
        if self.args.use_teacher_aux and not self.dataset.has_teacher_endpoint:
            raise RuntimeError(
                "Teacher auxiliary regularization is enabled, but the cache does not contain teacher_endpoint. "
                "Regenerate the cache with --teacher-backend diffusion_qsm or pass --no-use-teacher-aux."
            )

        self.curriculum_active = bool(self.args.curriculum_enabled and np.any(self.dataset.hard_scores > 0))
        self.curriculum_reason = "active"
        if self.args.curriculum_enabled and not self.dataset.has_any_hard_score_sidecar:
            self.curriculum_reason = "curriculum inactive because hard-score sidecars are absent"
        elif self.args.curriculum_enabled and not np.any(self.dataset.hard_scores > 0):
            self.curriculum_reason = "curriculum inactive because all hard scores are zero"
        elif not self.args.curriculum_enabled:
            self.curriculum_reason = "curriculum disabled by CLI"
        if self.args.bank_mode == "reconstruction_faithful" and self.args.curriculum_enabled:
            if not self.dataset.has_complete_hard_score_sidecars:
                raise RuntimeError(
                    "V6 reconstruction_faithful training requires complete hard-score sidecars. "
                    "Run stage4_build_hard_scores.py and verify shard alignment before training."
                )

        self.sampler = None
        if self.curriculum_active:
            self.sampler = DistributedWeightedSampler(
                self.dataset.hard_scores,
                num_replicas=self.world_size,
                rank=self.rank,
                replacement=True,
                seed=self.args.seed,
                curriculum_alpha=1.0,
            )
        elif self.world_size > 1:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.args.num_workers > 0,
        )

        if self.rank == 0:
            print(f"\nDataset samples: {len(self.dataset)}")
            print(f"Has teacher endpoint: {self.dataset.has_teacher_endpoint}")
            print(f"Shard cache size: {self.args.cache_shards}")
            print(f"Curriculum enabled: {self.args.curriculum_enabled}")
            print(
                f"Hard-score sidecars: {self.dataset.hard_score_sidecar_count}/{len(self.dataset.shards)} "
                f"(complete={self.dataset.has_complete_hard_score_sidecars})"
            )
            if len(self.dataset.hard_scores):
                print(
                    f"Hard score stats: mean={float(np.mean(self.dataset.hard_scores)):.4f}, "
                    f"std={float(np.std(self.dataset.hard_scores)):.4f}, "
                    f"max={float(np.max(self.dataset.hard_scores)):.4f}"
                )
            if self.args.curriculum_enabled and not self.curriculum_active:
                print(f"WARNING: {self.curriculum_reason}")

        self.eval_dataset = None
        self.eval_sampler = None
        self.eval_dataloader = None
        if self.args.eval_cache_dir:
            self.eval_dataset = Stage4V3Dataset(
                self.args.eval_cache_dir,
                cache_shards=self.args.cache_shards,
                expected_bank_mode=self.args.bank_mode,
            )
            if self.args.use_teacher_aux and not self.eval_dataset.has_teacher_endpoint:
                raise RuntimeError(
                    "Eval cache is missing teacher_endpoint while teacher aux is enabled."
                )
            self.eval_sampler = None
            if self.world_size > 1:
                self.eval_sampler = DistributedSampler(
                    self.eval_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    drop_last=False,
                )
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                sampler=self.eval_sampler,
                num_workers=self.args.num_workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.args.num_workers > 0,
            )
            if self.rank == 0:
                print(f"Eval samples: {len(self.eval_dataset)}")
                print(f"Eval NFE list: {self.args.eval_nfe_list}")

    def setup_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = None
        self.total_optimizer_steps = (
            int(self.args.max_train_steps)
            if int(self.args.max_train_steps) > 0
            else int(math.ceil(len(self.dataloader) / max(self.args.grad_accum_steps, 1))) * int(self.args.epochs)
        )
        if self.args.lr_schedule == "warmup_cosine":
            min_ratio = float(self.args.min_lr) / max(float(self.args.lr), 1e-12)
            min_ratio = min(max(min_ratio, 0.0), 1.0)
            warmup_steps = int(self.args.warmup_steps)
            if warmup_steps <= 0:
                warmup_steps = max(int(math.ceil(0.02 * max(self.total_optimizer_steps, 1))), 1)
            warmup_steps = min(warmup_steps, max(self.total_optimizer_steps - 1, 1))
            self.effective_warmup_steps = int(warmup_steps)

            def lr_lambda(step_index: int) -> float:
                if self.total_optimizer_steps <= 1:
                    return 1.0
                if step_index < self.effective_warmup_steps:
                    return float(step_index + 1) / max(self.effective_warmup_steps, 1)
                cosine_progress = min(
                    max(
                        float(step_index - self.effective_warmup_steps)
                        / max(self.total_optimizer_steps - self.effective_warmup_steps - 1, 1),
                        0.0,
                    ),
                    1.0,
                )
                cosine_scale = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
                return min_ratio + (1.0 - min_ratio) * cosine_scale

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            self.effective_warmup_steps = 0

        if self.run_mode == "resume":
            assert self.resume_payload is not None
            self.optimizer.load_state_dict(self.resume_payload["optimizer_state"])
            if self.scheduler is not None:
                scheduler_state = self.resume_payload.get("scheduler_state")
                if scheduler_state is None:
                    raise RuntimeError(
                        f"{self.resume_checkpoint} is missing scheduler_state required for resume."
                    )
                self.scheduler.load_state_dict(scheduler_state)
            if self.use_amp:
                scaler_state = self.resume_payload.get("scaler_state")
                if scaler_state is None:
                    raise RuntimeError(f"{self.resume_checkpoint} is missing scaler_state required for AMP resume.")
                self.scaler.load_state_dict(scaler_state)
            self.global_optimizer_step = int(self.resume_payload["global_optimizer_step"])
            self.start_epoch = int(self.resume_payload["epoch"])
            self.restored_epoch = int(self.resume_payload["epoch"])
            self.restored_global_optimizer_step = int(self.resume_payload["global_optimizer_step"])
            self.restore_rng_state()

        if self.rank == 0:
            print(
                f"\nOptimizer: AdamW, lr={self.args.lr}, weight_decay={self.args.weight_decay}, "
                f"lambda_dc={self.args.lambda_dc}"
            )
            if self.args.lr_schedule == "warmup_cosine":
                print(
                    "LR policy: warmup_cosine (authoritative), "
                    f"min_lr={self.args.min_lr}, warmup_steps={self.effective_warmup_steps}, "
                    f"total_optimizer_steps={self.total_optimizer_steps}"
                )
            else:
                print("LR policy: manual epoch schedule (authoritative), scheduler stepping disabled")
            if self.run_mode == "resume":
                print(
                    f"Resume restore: epoch={self.restored_epoch}, "
                    f"global_optimizer_step={self.restored_global_optimizer_step}, "
                    f"optimizer_lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

    def build_effective_config(self) -> dict:
        return {
            "entrypoint": str(Path(__file__).resolve()),
            "preset_name": self.args.preset_name,
            "run_mode": self.run_mode,
            "init_ckpt": resolve_optional_path_str(self.args.init_ckpt),
            "resume_ckpt": resolve_optional_path_str(self.args.resume_ckpt),
            "random_init": bool(self.args.random_init),
            "init_checkpoint_kind": self.init_checkpoint_kind,
            "cache_dir": str(Path(self.args.cache_dir).expanduser().resolve()),
            "output_dir": str(self.output_dir),
            "bank_mode": self.args.bank_mode,
            "cache_format": CACHE_FORMAT_BY_BANK_MODE[self.args.bank_mode],
            "dataset_samples": int(len(self.dataset)),
            "cache_metadata": self.dataset.cache_metadata,
            "curriculum_enabled": bool(self.args.curriculum_enabled),
            "curriculum_active": bool(self.curriculum_active),
            "curriculum_reason": self.curriculum_reason,
            "hard_score_sidecar_count": int(self.dataset.hard_score_sidecar_count),
            "hard_score_sidecars_complete": bool(self.dataset.has_complete_hard_score_sidecars),
            "hard_score_stats": {
                "mean": float(np.mean(self.dataset.hard_scores)) if len(self.dataset.hard_scores) else 0.0,
                "std": float(np.std(self.dataset.hard_scores)) if len(self.dataset.hard_scores) else 0.0,
                "max": float(np.max(self.dataset.hard_scores)) if len(self.dataset.hard_scores) else 0.0,
            },
            "effective_step_adapter_mode": self.effective_step_adapter_mode,
            "effective_step_adapter_hidden_dim": self.effective_step_adapter_hidden_dim,
            "effective_conditioning_mode": self.effective_conditioning_mode,
            "effective_conditioning_hidden_dim": self.effective_conditioning_hidden_dim,
            "effective_conditioning_norm_mode": self.effective_conditioning_norm_mode,
            "effective_conditioning_cone_threshold": self.effective_conditioning_cone_threshold,
            "effective_init_error_proxy_sigma": self.effective_init_error_proxy_sigma,
            "conditioning_channels": int(self.conditioning_channels),
            "effective_base_model_in_channels": int(self.effective_base_model_in_channels),
            "saved_args_present": bool(self.saved_args_present),
            "step_adapter_overridden_by_checkpoint": bool(self.step_adapter_overridden_by_checkpoint),
            "conditioning_overridden_by_checkpoint": bool(self.conditioning_overridden_by_checkpoint),
            "sigma_data": float(self.sigma_data),
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
            "restored_epoch": None if self.restored_epoch is None else int(self.restored_epoch),
            "restored_global_optimizer_step": (
                None if self.restored_global_optimizer_step is None else int(self.restored_global_optimizer_step)
            ),
            "lr_policy": {
                "authoritative_mechanism": self.args.lr_schedule,
                "base_lr": float(self.args.lr),
                "min_lr": float(self.args.min_lr),
                "warmup_steps": int(self.effective_warmup_steps),
                "phase2_lr": None if self.args.phase2_lr is None else float(self.args.phase2_lr),
                "total_optimizer_steps": int(self.total_optimizer_steps),
            },
            "curriculum_policy": {
                "enabled": bool(self.args.curriculum_enabled),
                "start_epoch": int(self.args.curriculum_start_epoch),
                "alpha": float(self.args.curriculum_alpha),
            },
            "phase_policy": {
                "enabled": bool(self.args.phase2_enabled),
                "start_epoch": int(self.args.phase2_start_epoch),
                "phase2_ct_weight": self.args.phase2_loss_ct_weight,
                "phase2_spectral_weight": self.args.phase2_loss_spectral_weight,
                "phase2_lr": self.args.phase2_lr,
                "ct_weight_schedule": list(self.args.loss_ct_weight_schedule),
            },
            "max_train_steps": int(self.args.max_train_steps),
            "cli_args": vars(self.args),
        }

    def write_run_registry(self) -> None:
        if self.rank != 0:
            return
        payload = self.build_effective_config()
        (self.output_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def phase2_active(self) -> bool:
        return bool(self.args.phase2_enabled and self.current_epoch >= self.args.phase2_start_epoch)

    def current_curriculum_alpha(self) -> float:
        if not self.curriculum_active:
            return 1.0
        if self.current_epoch < self.args.curriculum_start_epoch:
            return 1.0
        return float(self.args.curriculum_alpha)

    def curriculum_status_string(self) -> str:
        if not self.args.curriculum_enabled:
            return "disabled"
        if not self.curriculum_active:
            return self.curriculum_reason
        if self.current_epoch < self.args.curriculum_start_epoch:
            return f"warmup_uniform(epoch<{self.args.curriculum_start_epoch + 1})"
        return f"weighted(alpha={self.current_curriculum_alpha():.3f})"

    def phase2_status_string(self) -> str:
        if not self.args.phase2_enabled:
            return "disabled"
        if self.current_epoch < self.args.phase2_start_epoch:
            return f"pending(epoch<{self.args.phase2_start_epoch + 1})"
        return "active"

    def current_target_lr(self) -> float:
        if self.args.lr_schedule != "manual_epoch":
            return float(self.optimizer.param_groups[0]["lr"])
        if self.args.phase2_lr is not None and self.phase2_active():
            return float(self.args.phase2_lr)
        return float(self.args.lr)

    def apply_epoch_lr(self) -> None:
        if self.args.lr_schedule != "manual_epoch":
            return
        target_lr = self.current_target_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = target_lr

    def unwrap_model(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def choose_train_nfe(self) -> int:
        if len(self.args.train_nfe_schedule) == 1:
            return int(self.args.train_nfe_schedule[0])
        if self.world_size > 1:
            if self.rank == 0:
                index = torch.randint(len(self.args.train_nfe_schedule), (1,), device=self.device, dtype=torch.int64)
                choice = torch.tensor(
                    [int(self.args.train_nfe_schedule[int(index.item())])],
                    device=self.device,
                    dtype=torch.int64,
                )
            else:
                choice = torch.zeros(1, device=self.device, dtype=torch.int64)
            dist.broadcast(choice, src=0)
            return int(choice.item())
        return int(self.args.train_nfe_schedule[torch.randint(len(self.args.train_nfe_schedule), (1,)).item()])

    def build_sigma_schedule(self, batch_size: int, num_steps: int) -> list[torch.Tensor]:
        sigma_values = geometric_sigma_schedule(self.sigma_max, self.sigma_min, num_steps + 1)
        return [
            torch.full((batch_size,), float(value), device=self.device, dtype=torch.float32) for value in sigma_values
        ]

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

    def build_dc_policy(self, num_steps: int, infer: bool):
        return build_dc_rollout_policy(
            num_steps=num_steps,
            mode=self.args.dc_policy,
            lam=self.adapt_schedule_arg(self.args.lambda_dc, num_steps=num_steps),
            num_cg_steps=self.adapt_schedule_arg(
                self.args.cg_steps_infer if infer else self.args.cg_steps_train,
                num_steps=num_steps,
            ),
            final_polish=self.args.final_polish_infer if infer else self.args.final_polish_train,
            final_polish_lam=self.args.final_polish_lam,
            final_polish_steps=self.args.final_polish_steps_infer if infer else self.args.final_polish_steps_train,
        )

    def collect_rng_state_by_rank(self) -> dict | None:
        local_state = capture_local_rng_state(self.device)
        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, local_state)
            if self.rank != 0:
                return None
            return {str(rank): state for rank, state in enumerate(gathered)}
        return {str(self.rank): local_state}

    def restore_rng_state(self) -> None:
        if self.resume_payload is None:
            return
        state = None
        if "rng_state_by_rank" in self.resume_payload:
            state = self.resume_payload["rng_state_by_rank"].get(str(self.rank))
        if state is None:
            state = self.resume_payload.get("rng_state")
        if state is None:
            raise RuntimeError(f"{self.resume_checkpoint} is missing RNG state required for exact resume.")
        restore_local_rng_state(state, self.device)

    def build_conditioning_volume(
        self,
        x_init: torch.Tensor,
        field_gt: torch.Tensor,
        mask: torch.Tensor,
        cone_risk: torch.Tensor | None = None,
    ) -> torch.Tensor | ConditioningSpec | None:
        if not self.conditioning_active:
            return None
        return build_conditioning_spec(
            mode=self.effective_conditioning_mode,
            x_init=x_init,
            field_input=field_gt,
            mask=mask,
            dipole_kernel=self.dipole_kernel,
            cone_threshold=self.effective_conditioning_cone_threshold,
            cone_risk=cone_risk,
            init_error_proxy_sigma=self.effective_init_error_proxy_sigma,
            norm_mode=self.effective_conditioning_norm_mode,
        )

    def run_rollout_with_model(
        self,
        model: torch.nn.Module,
        x_init: torch.Tensor,
        field_gt: torch.Tensor,
        mask: torch.Tensor,
        conditioning_volume: torch.Tensor | ConditioningSpec | None,
        num_steps: int,
        infer: bool,
        collect_states: bool,
    ):
        sigma_schedule = self.build_sigma_schedule(batch_size=x_init.shape[0], num_steps=num_steps)
        dc_policy = self.build_dc_policy(num_steps=num_steps, infer=infer)
        return run_refinement_rollout(
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
            collect_priors=False,
        )

    def zero_loss_anchor(self) -> torch.Tensor:
        first_param = next(self.unwrap_model().parameters())
        return first_param.reshape(-1)[0] * 0.0

    def current_gpu_mem_reserved_gib(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.memory_reserved(self.device) / (1024**3))

    def current_teacher_aux_weight(self) -> float:
        if not self.args.use_teacher_aux or self.args.teacher_aux_weight <= 0:
            return 0.0
        if self.args.teacher_aux_decay_fraction <= 0:
            return float(self.args.teacher_aux_weight)
        decay_epochs = max(int(math.ceil(self.args.teacher_aux_decay_fraction * self.args.epochs)), 1)
        progress = min(max(self.current_epoch / decay_epochs, 0.0), 1.0)
        if progress >= 1.0:
            return 0.0
        if self.args.teacher_aux_decay_schedule == "linear":
            scale = 1.0 - progress
        else:
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.args.teacher_aux_weight) * float(scale)

    def current_ct_weight(self) -> float:
        schedule = getattr(self.args, "loss_ct_weight_schedule", None) or []
        if schedule:
            epoch_index = min(max(int(self.current_epoch), 0), len(schedule) - 1)
            return float(schedule[epoch_index])
        base = float(self.args.loss_ct_weight)
        phase2 = self.args.phase2_loss_ct_weight
        if phase2 is None or not self.phase2_active():
            return base
        return float(phase2)

    def current_spectral_weight(self) -> float:
        base_target = float(self.args.loss_spectral_weight)
        if self.args.phase2_loss_spectral_weight is not None and self.phase2_active():
            base_target = float(self.args.phase2_loss_spectral_weight)
        if base_target <= 0 or self.args.spectral_loss_mode == "none":
            return 0.0
        if self.args.spectral_warmup_fraction <= 0:
            return base_target

        warmup_epochs = max(int(math.ceil(self.args.spectral_warmup_fraction * self.args.epochs)), 1)
        progress = min(max((self.current_epoch + 1) / warmup_epochs, 0.0), 1.0)
        if self.args.spectral_warmup_schedule == "linear":
            scale = progress
        else:
            scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        return base_target * float(scale)

    def compute_gt_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        huber = pseudo_huber_loss(pred, target, mask, delta=self.args.pseudo_huber_delta)
        mse_weight = float(getattr(self.args, "loss_gt_mse_weight", 0.0))
        if mse_weight <= 0:
            return huber
        mse = masked_mse(pred, target, mask)
        return huber + mse_weight * mse

    def compute_physics_loss(self, states: list[torch.Tensor], field_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for state in states[1:]:
            predicted_field = forward_field_from_chi(state, self.dipole_kernel, mask=mask)
            losses.append(masked_mse(predicted_field, field_gt, mask))
        if not losses:
            return self.zero_loss_anchor()
        return torch.stack(losses).mean()

    def compute_consistency_loss(self, student_rollout, ema_rollout, mask: torch.Tensor) -> torch.Tensor:
        if not self.args.use_ema_consistency or ema_rollout is None:
            return self.zero_loss_anchor()

        losses = []
        for step_idx, (student_state, ema_state) in enumerate(zip(student_rollout.states[1:], ema_rollout.states[1:]), start=1):
            losses.append(
                pseudo_huber_loss(
                    student_state,
                    ema_state.detach(),
                    mask,
                    delta=self.args.pseudo_huber_delta,
                    weight=sigma_weight(
                        student_rollout.sigma_schedule[min(step_idx, len(student_rollout.sigma_schedule) - 1)],
                        power=self.args.ct_sigma_weight_power,
                    ),
                )
            )
        if not losses:
            return self.zero_loss_anchor()
        return torch.stack(losses).mean()

    def compute_spectral_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
        spectral_weight = self.current_spectral_weight()
        if self.spectral_weight_map is None or spectral_weight <= 0:
            return self.zero_loss_anchor(), 0.0
        if self.args.spectral_loss_mode == "hybrid_cone_residual_dipole":
            return (
                masked_hybrid_frequency_dipole_loss(
                    pred,
                    target,
                    dipole_kernel=self.dipole_kernel,
                    mask=mask,
                    weight_map=self.spectral_weight_map,
                    chi_weight=self.args.spectral_chi_weight,
                    dipole_weight=self.args.spectral_dipole_weight,
                    residual_power=self.args.spectral_residual_power,
                ),
                spectral_weight,
            )
        return masked_frequency_l2(pred, target, mask=mask, weight_map=self.spectral_weight_map), spectral_weight

    def compute_teacher_aux_loss(
        self,
        pred: torch.Tensor,
        teacher_endpoint: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_endpoint is None or not self.args.use_teacher_aux or self.args.teacher_aux_weight <= 0:
            return self.zero_loss_anchor()
        if self.args.teacher_aux_mode == "endpoint":
            return pseudo_huber_loss(pred, teacher_endpoint, mask, delta=self.args.teacher_aux_delta)
        if self.teacher_aux_weight_map is None:
            return self.zero_loss_anchor()
        return masked_frequency_l2(pred, teacher_endpoint, mask=mask, weight_map=self.teacher_aux_weight_map)

    def _global_valid(self, local_valid: bool) -> bool:
        if self.world_size == 1:
            return bool(local_valid)
        valid_tensor = torch.tensor([1 if local_valid else 0], device=self.device, dtype=torch.int64)
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
        return bool(valid_tensor.item())

    def _maybe_write_heartbeat(self, epoch: int, step: int, batch: dict, stage: str, info: dict | None = None) -> None:
        if not self.args.debug_heartbeat:
            return
        output_dir = Path(self.args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        hb_file = output_dir / f"heartbeat_rank{self.rank}.jsonl"
        sample_id = -1
        if "sample_id" in batch:
            sample_id = int(batch["sample_id"][0].item())
        record = {
            "epoch": int(epoch),
            "step": int(step),
            "stage": stage,
            "rank": int(self.rank),
            "sample_id": sample_id,
        }
        if info:
            record.update(info)
        with hb_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def specialization_loss(self, batch: dict):
        x_init = batch["x_init"].to(self.device, non_blocking=True)
        chi_gt = batch["chi_gt"].to(self.device, non_blocking=True)
        field_gt = batch["field"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        cone_risk = batch["cone_risk"].to(self.device, non_blocking=True) if "cone_risk" in batch else None
        teacher_endpoint = None
        if "teacher_endpoint" in batch:
            teacher_endpoint = batch["teacher_endpoint"].to(self.device, non_blocking=True)
        conditioning_volume = self.build_conditioning_volume(x_init, field_gt, mask, cone_risk=cone_risk)
        effective_ct_weight = self.current_ct_weight()
        effective_teacher_aux_weight = self.current_teacher_aux_weight()

        train_nfe = self.choose_train_nfe()
        student_rollout = self.run_rollout_with_model(
            model=self.model,
            x_init=x_init,
            field_gt=field_gt,
            mask=mask,
            conditioning_volume=conditioning_volume,
            num_steps=train_nfe,
            infer=False,
            collect_states=True,
        )

        ema_rollout = None
        if self.args.use_ema_consistency and effective_ct_weight > 0:
            with torch.no_grad():
                ema_rollout = self.run_rollout_with_model(
                    model=self.ema_model,
                    x_init=x_init,
                    field_gt=field_gt,
                    mask=mask,
                    conditioning_volume=conditioning_volume,
                    num_steps=train_nfe,
                    infer=False,
                    collect_states=True,
                )

        gt_loss = self.compute_gt_loss(student_rollout.final_state, chi_gt, mask)
        phys_loss = self.compute_physics_loss(student_rollout.states, field_gt, mask)
        ct_loss = self.compute_consistency_loss(student_rollout, ema_rollout, mask)
        spectral_loss, spectral_weight = self.compute_spectral_loss(student_rollout.final_state, chi_gt, mask)
        teacher_aux_loss = (
            self.compute_teacher_aux_loss(student_rollout.final_state, teacher_endpoint, mask)
            if effective_teacher_aux_weight > 0
            else self.zero_loss_anchor()
        )

        total_loss = (
            self.args.loss_gt_weight * gt_loss
            + self.args.loss_phys_weight * phys_loss
            + effective_ct_weight * ct_loss
            + spectral_weight * spectral_loss
            + effective_teacher_aux_weight * teacher_aux_loss
        )

        loss_terms = [gt_loss, phys_loss, ct_loss, spectral_loss, teacher_aux_loss, total_loss]
        local_valid = all(bool(torch.isfinite(term).all().item()) for term in loss_terms)
        local_valid = local_valid and all(bool(torch.isfinite(state).all().item()) for state in student_rollout.states)
        if ema_rollout is not None:
            local_valid = local_valid and all(bool(torch.isfinite(state).all().item()) for state in ema_rollout.states)
        global_valid = self._global_valid(local_valid)

        if not global_valid:
            total_loss = self.zero_loss_anchor()

        return total_loss, {
            "gt_loss": float(gt_loss.detach().cpu()) if global_valid else 0.0,
            "phys_loss": float(phys_loss.detach().cpu()) if global_valid else 0.0,
            "ct_loss": float(ct_loss.detach().cpu()) if global_valid else 0.0,
            "spectral_loss": float(spectral_loss.detach().cpu()) if global_valid else 0.0,
            "spectral_weight": float(spectral_weight),
            "ct_weight": float(effective_ct_weight),
            "teacher_aux_loss": float(teacher_aux_loss.detach().cpu()) if global_valid else 0.0,
            "teacher_aux_weight": float(effective_teacher_aux_weight),
            "train_nfe": float(train_nfe),
            "is_valid": bool(global_valid),
            "local_valid": bool(local_valid),
            "curriculum_alpha": float(self.current_curriculum_alpha()),
            "phase2_active": bool(self.phase2_active()),
            "gt_contrib": float(self.args.loss_gt_weight * gt_loss.detach().cpu()) if global_valid else 0.0,
            "phys_contrib": float(self.args.loss_phys_weight * phys_loss.detach().cpu()) if global_valid else 0.0,
            "ct_contrib": float(effective_ct_weight * ct_loss.detach().cpu()) if global_valid else 0.0,
            "spectral_contrib": float(spectral_weight * spectral_loss.detach().cpu()) if global_valid else 0.0,
            "teacher_aux_contrib": (
                float(effective_teacher_aux_weight * teacher_aux_loss.detach().cpu()) if global_valid else 0.0
            ),
        }

    def infer_from_batch(self, batch: dict, infer_nfe: int | None = None) -> torch.Tensor:
        nfe = int(self.args.infer_nfe if infer_nfe is None else infer_nfe)
        x_init = batch["x_init"].to(self.device, non_blocking=True)
        field_gt = batch["field"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        cone_risk = batch["cone_risk"].to(self.device, non_blocking=True) if "cone_risk" in batch else None
        infer_model = self.ema_model if self.args.infer_use_ema else self.model
        conditioning_volume = self.build_conditioning_volume(x_init, field_gt, mask, cone_risk=cone_risk)
        with torch.no_grad():
            rollout = self.run_rollout_with_model(
                model=infer_model,
                x_init=x_init,
                field_gt=field_gt,
                mask=mask,
                conditioning_volume=conditioning_volume,
                num_steps=nfe,
                infer=True,
                collect_states=False,
            )
        return rollout.final_state

    @torch.no_grad()
    def update_ema_model(self) -> None:
        if (
            not self.args.use_ema_consistency
            and not self.args.infer_use_ema
            and not self.args.save_ema_state
        ):
            return
        src_model = self.unwrap_model()
        decay = float(self.args.ema_decay)
        one_minus_decay = 1.0 - decay
        for ema_param, src_param in zip(self.ema_model.parameters(), src_model.parameters()):
            ema_param.data.mul_(decay).add_(src_param.data, alpha=one_minus_decay)
        for ema_buffer, src_buffer in zip(self.ema_model.buffers(), src_model.buffers()):
            ema_buffer.copy_(src_buffer)

    def optimizer_step(self) -> None:
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.update_ema_model()
        self.global_optimizer_step += 1
        if self.args.max_train_steps > 0 and self.global_optimizer_step >= int(self.args.max_train_steps):
            self.stop_requested = True

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.apply_epoch_lr()
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
            if isinstance(self.sampler, DistributedWeightedSampler):
                self.sampler.set_curriculum_alpha(self.current_curriculum_alpha())

        self.optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        extra_info_keys = list(
            dict.fromkeys(
                (
                    *tuple(self.extra_training_info_keys()),
                    *tuple(getattr(self, "extra_training_metric_keys", ())),
                )
            )
        )
        builtin_extra_keys = {"macro_dc_loss", "macro_dc_contrib", "macro_dc_applied"}
        extra_info_keys = [key for key in extra_info_keys if key not in builtin_extra_keys]
        sums = {
            "loss": 0.0,
            "gt_loss": 0.0,
            "gradient_loss": 0.0,
            "phys_loss": 0.0,
            "ct_loss": 0.0,
            "spectral_loss": 0.0,
            "spectral_weight": 0.0,
            "teacher_aux_loss": 0.0,
            "teacher_aux_weight": 0.0,
            "gradient_weight": 0.0,
            "train_nfe": 0.0,
            "hard_score": 0.0,
            "ct_weight": 0.0,
            "gpu_mem_reserved": 0.0,
            "valid_batches": 0.0,
            "invalid_batches": 0.0,
            "curriculum_alpha": 0.0,
            "phase2_active": 0.0,
            "gt_contrib": 0.0,
            "gradient_contrib": 0.0,
            "phys_contrib": 0.0,
            "ct_contrib": 0.0,
            "spectral_contrib": 0.0,
            "teacher_aux_contrib": 0.0,
            "macro_dc_loss": 0.0,
            "macro_dc_contrib": 0.0,
            "macro_dc_applied": 0.0,
        }
        for key in extra_info_keys:
            sums[key] = 0.0

        progress = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}", disable=self.rank != 0)
        for step, batch in enumerate(progress):
            self._maybe_write_heartbeat(epoch, step, batch, "before_forward")
            loss, info = self.specialization_loss(batch)
            self._maybe_write_heartbeat(epoch, step, batch, "after_forward", info)
            for key in info.keys():
                if key.startswith("macro_dc_") and key not in sums:
                    sums[key] = 0.0
                    extra_info_keys.append(key)

            batch_weight = float(batch["x_init"].shape[0])
            if not info["is_valid"]:
                sums["invalid_batches"] += batch_weight
                self._maybe_write_heartbeat(epoch, step, batch, "skip_invalid", info)
                if self.rank == 0:
                    progress.set_postfix(invalid="1", nfe=f"{int(info['train_nfe'])}")
                continue

            scaled_loss = loss / self.args.grad_accum_steps
            self._maybe_write_heartbeat(epoch, step, batch, "before_backward", info)
            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            self._maybe_write_heartbeat(epoch, step, batch, "after_backward", info)

            accum_step += 1
            sums["loss"] += float(loss.detach().cpu()) * batch_weight
            sums["gt_loss"] += info["gt_loss"] * batch_weight
            sums["gradient_loss"] += info.get("gradient_loss", 0.0) * batch_weight
            sums["phys_loss"] += info["phys_loss"] * batch_weight
            sums["ct_loss"] += info["ct_loss"] * batch_weight
            sums["spectral_loss"] += info["spectral_loss"] * batch_weight
            sums["spectral_weight"] += info["spectral_weight"] * batch_weight
            sums["teacher_aux_loss"] += info["teacher_aux_loss"] * batch_weight
            sums["teacher_aux_weight"] += info["teacher_aux_weight"] * batch_weight
            sums["gradient_weight"] += info.get("gradient_weight", 0.0) * batch_weight
            sums["train_nfe"] += info["train_nfe"] * batch_weight
            sums["ct_weight"] += info["ct_weight"] * batch_weight
            sums["hard_score"] += float(batch["hard_score"].mean().item()) * batch_weight
            sums["gpu_mem_reserved"] += self.current_gpu_mem_reserved_gib() * batch_weight
            sums["valid_batches"] += batch_weight
            sums["curriculum_alpha"] += info["curriculum_alpha"] * batch_weight
            sums["phase2_active"] += (1.0 if info["phase2_active"] else 0.0) * batch_weight
            sums["gt_contrib"] += info["gt_contrib"] * batch_weight
            sums["gradient_contrib"] += info.get("gradient_contrib", 0.0) * batch_weight
            sums["phys_contrib"] += info["phys_contrib"] * batch_weight
            sums["ct_contrib"] += info["ct_contrib"] * batch_weight
            sums["spectral_contrib"] += info["spectral_contrib"] * batch_weight
            sums["teacher_aux_contrib"] += info["teacher_aux_contrib"] * batch_weight
            sums["macro_dc_loss"] += float(info.get("macro_dc_loss", 0.0)) * batch_weight
            sums["macro_dc_contrib"] += float(info.get("macro_dc_contrib", 0.0)) * batch_weight
            sums["macro_dc_applied"] += float(info.get("macro_dc_applied", 0.0)) * batch_weight
            for key in extra_info_keys:
                sums[key] += float(info.get(key, 0.0)) * batch_weight

            if accum_step >= self.args.grad_accum_steps:
                self._maybe_write_heartbeat(epoch, step, batch, "before_optimizer_step", info)
                self.optimizer_step()
                self._maybe_write_heartbeat(epoch, step, batch, "after_optimizer_step", info)
                accum_step = 0
                if self.stop_requested:
                    break

            if self.rank == 0:
                progress.set_postfix(
                    loss=f"{float(loss.detach().cpu()):.6f}",
                    gt=f"{info['gt_loss']:.6f}",
                    grad=f"{info.get('gradient_loss', 0.0):.6f}",
                    phys=f"{info['phys_loss']:.6f}",
                    ct=f"{info['ct_loss']:.6f}",
                    ct_w=f"{info['ct_weight']:.3f}",
                    spec=f"{info['spectral_loss']:.6f}",
                    spec_w=f"{info['spectral_weight']:.4f}",
                    hard=f"{float(batch['hard_score'].mean().item()):.3f}",
                    mem=f"{self.current_gpu_mem_reserved_gib():.1f}G",
                    teacher=f"{info['teacher_aux_loss']:.6f}",
                    nfe=f"{int(info['train_nfe'])}",
                )

        if accum_step > 0:
            self.optimizer_step()
        stats = torch.tensor(
            [
                sums["loss"],
                sums["gt_loss"],
                sums["gradient_loss"],
                sums["phys_loss"],
                sums["ct_loss"],
                sums["spectral_loss"],
                sums["spectral_weight"],
                sums["teacher_aux_loss"],
                sums["teacher_aux_weight"],
                sums["gradient_weight"],
                sums["train_nfe"],
                sums["hard_score"],
                sums["ct_weight"],
                sums["gpu_mem_reserved"],
                sums["valid_batches"],
                sums["invalid_batches"],
                sums["curriculum_alpha"],
                sums["phase2_active"],
                sums["gt_contrib"],
                sums["gradient_contrib"],
                sums["phys_contrib"],
                sums["ct_contrib"],
                sums["spectral_contrib"],
                sums["teacher_aux_contrib"],
                sums["macro_dc_loss"],
                sums["macro_dc_contrib"],
                sums["macro_dc_applied"],
                *(sums[key] for key in extra_info_keys),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total_valid = max(float(stats[14].item()), 1.0)
        return {
            "loss": float(stats[0].item() / total_valid),
            "gt_loss": float(stats[1].item() / total_valid),
            "gradient_loss": float(stats[2].item() / total_valid),
            "phys_loss": float(stats[3].item() / total_valid),
            "ct_loss": float(stats[4].item() / total_valid),
            "spectral_loss": float(stats[5].item() / total_valid),
            "spectral_weight_mean": float(stats[6].item() / total_valid),
            "teacher_aux_loss": float(stats[7].item() / total_valid),
            "teacher_aux_weight_mean": float(stats[8].item() / total_valid),
            "gradient_weight_mean": float(stats[9].item() / total_valid),
            "train_nfe_mean": float(stats[10].item() / total_valid),
            "hard_score_mean": float(stats[11].item() / total_valid),
            "ct_weight_mean": float(stats[12].item() / total_valid),
            "gpu_mem_reserved_gib": float(stats[13].item() / total_valid),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "valid_batches": int(stats[14].item()),
            "invalid_batches": int(stats[15].item()),
            "optimizer_step": int(self.global_optimizer_step),
            "curriculum_alpha_mean": float(stats[16].item() / total_valid),
            "phase2_active_fraction": float(stats[17].item() / total_valid),
            "gt_contrib_mean": float(stats[18].item() / total_valid),
            "gradient_contrib_mean": float(stats[19].item() / total_valid),
            "phys_contrib_mean": float(stats[20].item() / total_valid),
            "ct_contrib_mean": float(stats[21].item() / total_valid),
            "spectral_contrib_mean": float(stats[22].item() / total_valid),
            "teacher_aux_contrib_mean": float(stats[23].item() / total_valid),
            "macro_dc_loss": float(stats[24].item() / total_valid),
            "macro_dc_contrib": float(stats[25].item() / total_valid),
            "macro_dc_applied": float(stats[26].item() / total_valid),
        }
        for offset, key in enumerate(extra_info_keys, start=27):
            result[key] = float(stats[offset].item() / total_valid)
        return result

    def extra_training_info_keys(self) -> tuple[str, ...]:
        return ()

    @staticmethod
    def compute_masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs() * mask
        return diff.sum() / mask.sum().clamp_min(1e-8)

    @staticmethod
    def compute_masked_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_sum = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1e-8)
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        mask_flat = mask.reshape(mask.shape[0], -1)

        pred_mean = (pred_flat * mask_flat).sum(dim=1) / mask_sum
        target_mean = (target_flat * mask_flat).sum(dim=1) / mask_sum

        pred_centered = (pred_flat - pred_mean.unsqueeze(1)) * mask_flat
        target_centered = (target_flat - target_mean.unsqueeze(1)) * mask_flat
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = (
            pred_centered.square().sum(dim=1).sqrt() * target_centered.square().sum(dim=1).sqrt()
        ).clamp_min(1e-8)
        return (numerator / denominator).mean()

    def evaluate(self, epoch: int) -> dict:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        if self.eval_sampler is not None:
            self.eval_sampler.set_epoch(epoch)

        accumulators = {
            nfe: {"mae": 0.0, "mse": 0.0, "psnr": 0.0, "pearson": 0.0, "field_resid": 0.0}
            for nfe in self.args.eval_nfe_list
        }
        sample_counter = 0.0

        with torch.no_grad():
            for batch_index, batch in enumerate(self.eval_dataloader):
                if self.args.eval_max_batches > 0 and batch_index >= self.args.eval_max_batches:
                    break

                chi_gt = batch["chi_gt"].to(self.device, non_blocking=True)
                field_gt = batch["field"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                batch_weight = float(chi_gt.shape[0])

                for nfe in self.args.eval_nfe_list:
                    pred = self.infer_from_batch(batch, infer_nfe=nfe)
                    mse = masked_mse(pred, chi_gt, mask)
                    mae = self.compute_masked_l1(pred, chi_gt, mask)
                    pearson = self.compute_masked_pearson(pred, chi_gt, mask)
                    predicted_field = forward_field_from_chi(pred, self.dipole_kernel, mask=mask)
                    field_resid = masked_mse(predicted_field, field_gt, mask)
                    psnr = 10.0 * torch.log10(
                        torch.tensor(self.args.eval_data_range**2, device=self.device, dtype=torch.float32)
                        / mse.clamp_min(1e-8)
                    )

                    accumulators[nfe]["mae"] += float(mae.detach().cpu()) * batch_weight
                    accumulators[nfe]["mse"] += float(mse.detach().cpu()) * batch_weight
                    accumulators[nfe]["psnr"] += float(psnr.detach().cpu()) * batch_weight
                    accumulators[nfe]["pearson"] += float(pearson.detach().cpu()) * batch_weight
                    accumulators[nfe]["field_resid"] += float(field_resid.detach().cpu()) * batch_weight

                sample_counter += batch_weight

        stats = []
        for nfe in self.args.eval_nfe_list:
            stats.extend(
                [
                    accumulators[nfe]["mae"],
                    accumulators[nfe]["mse"],
                    accumulators[nfe]["psnr"],
                    accumulators[nfe]["pearson"],
                    accumulators[nfe]["field_resid"],
                ]
            )
        stats.append(sample_counter)

        stats_tensor = torch.tensor(stats, device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        total_samples = max(float(stats_tensor[-1].item()), 1.0)
        result = {}
        offset = 0
        for nfe in self.args.eval_nfe_list:
            result[f"eval_mae_nfe{nfe}"] = float(stats_tensor[offset].item() / total_samples)
            result[f"eval_mse_nfe{nfe}"] = float(stats_tensor[offset + 1].item() / total_samples)
            result[f"eval_psnr_nfe{nfe}"] = float(stats_tensor[offset + 2].item() / total_samples)
            result[f"eval_pearson_nfe{nfe}"] = float(stats_tensor[offset + 3].item() / total_samples)
            result[f"eval_field_resid_nfe{nfe}"] = float(stats_tensor[offset + 4].item() / total_samples)
            offset += 5
        result["eval_samples"] = int(stats_tensor[-1].item())
        return result

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        rng_state_by_rank = self.collect_rng_state_by_rank()
        if self.rank != 0:
            return

        output_dir = Path(self.args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch + 1,
            "model_state": self.unwrap_model().state_dict() if self.args.save_model_state else None,
            "ema_state": self.ema_model.state_dict() if self.args.save_ema_state else None,
            "optimizer_state": self.optimizer.state_dict() if self.args.save_optimizer_state else None,
            "scheduler_state": self.scheduler.state_dict() if (self.scheduler is not None and self.args.save_optimizer_state) else None,
            "effective_config": self.build_effective_config(),
            "scaler_state": self.scaler.state_dict() if self.args.save_scaler_state else None,
            "metrics": metrics,
            "sigma_data": self.sigma_data,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "global_optimizer_step": int(self.global_optimizer_step),
            "run_mode": self.run_mode,
            "resume_ckpt": None if self.args.resume_ckpt is None else str(Path(self.args.resume_ckpt).expanduser().resolve()),
            "restored_epoch": self.restored_epoch,
            "restored_global_optimizer_step": self.restored_global_optimizer_step,
            "rng_state_by_rank": rng_state_by_rank,
            "args": vars(self.args),
        }

        if self.args.checkpoint_policy == "epoch_and_latest":
            torch.save(ckpt, output_dir / f"stage4_v3_epoch_{epoch + 1:03d}.pth")
        torch.save(ckpt, output_dir / "latest.pth")

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.args.epochs):
            self.current_epoch = epoch
            metrics = self.train_epoch(epoch)
            if self.args.eval_every > 0 and ((epoch + 1) % self.args.eval_every == 0 or epoch == self.args.epochs - 1):
                metrics.update(self.evaluate(epoch))
            if self.rank == 0:
                print(
                    f"\nEpoch {epoch + 1}/{self.args.epochs} | "
                    f"loss={metrics['loss']:.6f} | "
                    f"gt={metrics['gt_loss']:.6f} | "
                    f"grad={metrics.get('gradient_loss', 0.0):.6f} | "
                    f"phys={metrics['phys_loss']:.6f} | "
                    f"ct={metrics['ct_loss']:.6f} | "
                    f"spectral={metrics['spectral_loss']:.6f} | "
                    f"spectral_w={metrics['spectral_weight_mean']:.4f} | "
                    f"grad_w={metrics.get('gradient_weight_mean', 0.0):.4f} | "
                    f"ct_w={metrics['ct_weight_mean']:.4f} | "
                    f"hard={metrics['hard_score_mean']:.4f} | "
                    f"lr={metrics['lr']:.2e} | "
                    f"gpu_mem_reserved={metrics['gpu_mem_reserved_gib']:.2f}GiB | "
                    f"opt_step={metrics['optimizer_step']} | "
                    f"teacher_aux={metrics['teacher_aux_loss']:.6f} | "
                    f"teacher_w={metrics['teacher_aux_weight_mean']:.4f} | "
                    f"nfe={metrics['train_nfe_mean']:.2f} | "
                    f"valid={metrics['valid_batches']} | "
                    f"invalid={metrics['invalid_batches']} | "
                    f"curriculum_alpha={metrics['curriculum_alpha_mean']:.3f} | "
                    f"phase2_active={metrics['phase2_active_fraction']:.2f}"
                )
                print(f"  Curriculum state: {self.curriculum_status_string()}")
                print(f"  Phase2 state: {self.phase2_status_string()}")
                print(
                    "  Effective weights: "
                    f"gradient={metrics.get('gradient_weight_mean', 0.0):.4f}, "
                    f"ct={metrics['ct_weight_mean']:.4f}, "
                    f"spectral={metrics['spectral_weight_mean']:.4f}, "
                    f"teacher_aux={metrics['teacher_aux_weight_mean']:.4f}"
                )
                print(
                    "  Effective contributions: "
                    f"gt={metrics['gt_contrib_mean']:.6f}, "
                    f"gradient={metrics.get('gradient_contrib_mean', 0.0):.6f}, "
                    f"phys={metrics['phys_contrib_mean']:.6f}, "
                    f"ct={metrics['ct_contrib_mean']:.6f}, "
                    f"spectral={metrics['spectral_contrib_mean']:.6f}, "
                    f"teacher_aux={metrics['teacher_aux_contrib_mean']:.6f}"
                )
                if "macro_dc_loss" in metrics:
                    print(
                        "  Macro DC alignment: "
                        f"loss={metrics.get('macro_dc_loss', 0.0):.6f}, "
                        f"contrib={metrics.get('macro_dc_contrib', 0.0):.6f}, "
                        f"applied={metrics.get('macro_dc_applied', 0.0):.3f}"
                    )
                for nfe in self.args.eval_nfe_list:
                    mae_key = f"eval_mae_nfe{nfe}"
                    if mae_key in metrics:
                        print(
                            f"  Eval NFE={nfe}: "
                            f"MAE={metrics[mae_key]:.6f}, "
                            f"PSNR={metrics[f'eval_psnr_nfe{nfe}']:.4f}, "
                            f"Pearson={metrics[f'eval_pearson_nfe{nfe}']:.4f}, "
                            f"FieldResid={metrics[f'eval_field_resid_nfe{nfe}']:.6f}"
                        )
            self.save_checkpoint(epoch, metrics)
            if self.stop_requested:
                break

        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-dir", "--bank-dir", dest="cache_dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--diffusion-root", type=str, default=None)
    parser.add_argument("--preset-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-schedule", choices=["manual_epoch", "warmup_cosine"], default="manual_epoch")
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-dropout", type=float, default=0.0)
    parser.add_argument("--lambda-dc", type=str, default="0.1")
    parser.add_argument("--cg-steps-train", type=str, default="8")
    parser.add_argument("--cg-steps-infer", type=str, default="8")
    parser.add_argument("--dc-policy", choices=["fixed", "stepwise_lambda", "stepwise_lambda+final_polish"], default="fixed")
    parser.add_argument("--final-polish-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final-polish-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-polish-lam", type=float, default=None)
    parser.add_argument("--final-polish-steps-train", type=int, default=None)
    parser.add_argument("--final-polish-steps-infer", type=int, default=None)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--bank-mode", choices=SUPPORTED_BANK_MODES, default="gt_core")
    parser.add_argument("--train-nfe-schedule", type=str, default="2")
    parser.add_argument("--infer-nfe", type=int, default=2)
    parser.add_argument("--infer-use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student-sigma-max-factor", type=float, default=1.0)
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
    parser.add_argument(
        "--two-step-role-mode",
        choices=["none", "low_detail"],
        default="none",
        help=(
            "Experimental 2-NFE residual-delta role separation. "
            "'low_detail' supervises step 1 as a low-frequency physics-safe correction "
            "and step 2/final as the detail-carrying correction."
        ),
    )
    parser.add_argument("--two-step-role-low-weight", type=float, default=0.0)
    parser.add_argument("--two-step-role-detail-weight", type=float, default=0.0)
    parser.add_argument("--two-step-role-lowpass-kernel", type=int, default=9)
    parser.add_argument("--two-step-role-delta", type=float, default=0.01)
    parser.add_argument(
        "--loss-morph-anchor-weight",
        type=float,
        default=0.0,
        help="msQSM-style morphology-prior anchor loss weight around qsm0_morph.",
    )
    parser.add_argument("--morph-anchor-scale", type=float, default=10.0)
    parser.add_argument("--morph-anchor-power", type=float, default=1.0)
    parser.add_argument(
        "--loss-field-morph-weight",
        type=float,
        default=0.0,
        help="msQSM-style Sobel morphology matching loss weight between output QSM and measured field.",
    )
    parser.add_argument("--field-morph-eps", type=float, default=1e-6)
    parser.add_argument("--use-ema-consistency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ct-sigma-weight-power", type=float, default=0.5)
    parser.add_argument("--loss-gt-weight", type=float, default=1.0)
    parser.add_argument("--loss-gt-mse-weight", type=float, default=0.0)
    parser.add_argument("--loss-gradient-weight", type=float, default=0.0)
    parser.add_argument("--loss-phys-weight", type=float, default=0.5)
    parser.add_argument("--loss-ct-weight", type=float, default=0.25)
    parser.add_argument(
        "--loss-ct-weight-schedule",
        type=str,
        default="",
        help=(
            "Comma-separated per-epoch CT weights. When provided, this overrides "
            "--loss-ct-weight and phase2 CT scheduling for the corresponding epochs."
        ),
    )
    parser.add_argument("--loss-spectral-weight", type=float, default=0.0)
    parser.add_argument("--curriculum-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--curriculum-start-epoch", type=int, default=2)
    parser.add_argument("--curriculum-alpha", type=float, default=0.4)
    parser.add_argument("--phase2-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase2-start-epoch", type=int, default=2)
    parser.add_argument("--phase2-loss-ct-weight", type=float, default=None)
    parser.add_argument("--phase2-loss-spectral-weight", type=float, default=None)
    parser.add_argument("--phase2-lr", type=float, default=None)
    parser.add_argument("--pseudo-huber-delta", type=float, default=0.01)
    parser.add_argument("--gradient-loss-delta", type=float, default=0.01)
    parser.add_argument(
        "--spectral-loss-mode",
        choices=["none", "cone_weighted", "cone_weighted_residual", "hybrid_cone_residual_dipole"],
        default="none",
    )
    parser.add_argument("--spectral-cone-threshold", type=float, default=0.05)
    parser.add_argument("--spectral-cone-weight", type=float, default=4.0)
    parser.add_argument("--spectral-background-weight", type=float, default=1.0)
    parser.add_argument("--spectral-chi-weight", type=float, default=1.0)
    parser.add_argument("--spectral-dipole-weight", type=float, default=1.0)
    parser.add_argument("--spectral-residual-power", type=float, default=1.0)
    parser.add_argument("--spectral-warmup-fraction", type=float, default=0.25)
    parser.add_argument("--spectral-warmup-schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--use-teacher-aux", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-aux-mode", choices=["endpoint", "cone_weighted"], default="endpoint")
    parser.add_argument("--teacher-aux-weight", type=float, default=0.05)
    parser.add_argument("--teacher-aux-decay-fraction", type=float, default=0.25)
    parser.add_argument("--teacher-aux-decay-schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--teacher-aux-delta", type=float, default=0.01)
    parser.add_argument("--teacher-aux-cone-threshold", type=float, default=0.05)
    parser.add_argument("--teacher-aux-cone-weight", type=float, default=4.0)
    parser.add_argument("--teacher-aux-background-weight", type=float, default=1.0)
    parser.add_argument("--cache-shards", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-policy", choices=["epoch_and_latest", "latest_only"], default="epoch_and_latest")
    parser.add_argument("--save-model-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-ema-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-optimizer-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-scaler-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-cache-dir", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=10)
    parser.add_argument("--eval-nfe-list", type=str, default="2")
    parser.add_argument("--eval-data-range", type=float, default=1.0)
    parser.add_argument("--patch-shape", type=str, default="64,64,64")
    parser.add_argument("--voxel-size", type=str, default="1,1,1")
    parser.add_argument("--b0-dir", type=str, default="0,0,1")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--student-use-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing inside the student UNet (numerically identical; ~30%% slower, much lower activation memory).",
    )
    parser.add_argument("--debug-heartbeat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)

    # Deprecated compatibility flags: accepted but ignored in v3 mainline.
    parser.add_argument("--teacher-target-mode", type=str, default="trajectory")
    parser.add_argument("--loss-traj-weight", type=float, default=0.0)
    parser.add_argument("--replay-mode", type=str, default="none")
    parser.add_argument("--loss-replay-weight", type=float, default=0.0)
    parser.add_argument("--replay-sigma-weight-power", type=float, default=0.5)
    return parser


def parse_runtime_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    raw_argv: list[str],
) -> argparse.Namespace:
    init_mode_count = int(bool(args.init_ckpt)) + int(bool(args.resume_ckpt)) + int(bool(args.random_init))
    if init_mode_count != 1:
        raise ValueError("Exactly one of --init-ckpt, --resume-ckpt, or --random-init must be provided.")
    if args.resume_ckpt:
        args = merge_resume_args(parser, args, raw_argv)
    else:
        if not args.cache_dir:
            raise ValueError("--cache-dir is required when using --init-ckpt or --random-init.")
        args.output_dir = args.output_dir or DEFAULT_STAGE4_OUTPUT_DIR
        args.provided_cli_dests = sorted(extract_provided_cli_dests(parser, raw_argv))
        args.resume_expected_output_dir = None
        args.resume_restore_epoch = None
        args.resume_restore_global_optimizer_step = None

    args.train_nfe_schedule = parse_int_list_maybe(args.train_nfe_schedule)
    args.eval_nfe_list = parse_int_list_maybe(args.eval_nfe_list)
    args.loss_ct_weight_schedule = parse_float_list_maybe(getattr(args, "loss_ct_weight_schedule", ""))
    args.patch_shape = parse_triplet_maybe(args.patch_shape, int)
    args.voxel_size = parse_triplet_maybe(args.voxel_size, float)
    args.b0_dir = parse_triplet_maybe(args.b0_dir, float)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.max_train_steps < 0:
        raise ValueError("--max-train-steps must be >= 0")
    if args.lr_schedule == "warmup_cosine" and args.min_lr <= 0:
        raise ValueError("--min-lr must be > 0 when --lr-schedule=warmup_cosine")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if not args.save_model_state and not args.save_ema_state:
        raise ValueError("At least one of --save-model-state or --save-ema-state must be enabled")
    if any(value < 1 for value in args.train_nfe_schedule):
        raise ValueError(f"--train-nfe-schedule must contain positive integers, got {args.train_nfe_schedule}")
    if args.infer_nfe < 1:
        raise ValueError("--infer-nfe must be >= 1")
    if any(value < 1 for value in args.eval_nfe_list):
        raise ValueError(f"--eval-nfe-list must contain positive integers, got {args.eval_nfe_list}")
    if (
        args.loss_gt_weight < 0
        or args.loss_gt_mse_weight < 0
        or args.loss_phys_weight < 0
        or args.loss_ct_weight < 0
        or args.loss_spectral_weight < 0
    ):
        raise ValueError("All primary loss weights must be >= 0")
    if args.curriculum_start_epoch < 0:
        raise ValueError("--curriculum-start-epoch must be >= 0")
    if not 0 <= args.curriculum_alpha <= 1:
        raise ValueError("--curriculum-alpha must be in [0, 1]")
    if args.phase2_start_epoch < 0:
        raise ValueError("--phase2-start-epoch must be >= 0")
    if args.phase2_loss_ct_weight is not None and args.phase2_loss_ct_weight < 0:
        raise ValueError("--phase2-loss-ct-weight must be >= 0")
    if any(value < 0 for value in args.loss_ct_weight_schedule):
        raise ValueError("--loss-ct-weight-schedule values must be >= 0")
    if args.loss_ct_weight_schedule and args.phase2_enabled:
        raise ValueError("--loss-ct-weight-schedule cannot be combined with --phase2-enabled.")
    if args.phase2_loss_spectral_weight is not None and args.phase2_loss_spectral_weight < 0:
        raise ValueError("--phase2-loss-spectral-weight must be >= 0")
    if args.phase2_lr is not None and args.phase2_lr <= 0:
        raise ValueError("--phase2-lr must be > 0")
    if args.spectral_chi_weight < 0 or args.spectral_dipole_weight < 0:
        raise ValueError("--spectral-chi-weight and --spectral-dipole-weight must be >= 0")
    if args.spectral_residual_power < 0:
        raise ValueError("--spectral-residual-power must be >= 0")
    if args.spectral_warmup_fraction < 0:
        raise ValueError("--spectral-warmup-fraction must be >= 0")
    if args.teacher_aux_weight < 0:
        raise ValueError("--teacher-aux-weight must be >= 0")
    if args.teacher_aux_decay_fraction < 0:
        raise ValueError("--teacher-aux-decay-fraction must be >= 0")
    if args.student_sigma_max_factor <= 0:
        raise ValueError("--student-sigma-max-factor must be > 0")
    if args.step_adapter_hidden_dim < 1:
        raise ValueError("--step-adapter-hidden-dim must be >= 1")
    if args.conditioning_hidden_dim < 1:
        raise ValueError("--conditioning-hidden-dim must be >= 1")
    if args.conditioning_cone_threshold <= 0:
        raise ValueError("--conditioning-cone-threshold must be > 0")
    if args.init_error_proxy_sigma <= 0:
        raise ValueError("--init-error-proxy-sigma must be > 0")
    if args.ct_sigma_weight_power < 0:
        raise ValueError("--ct-sigma-weight-power must be >= 0")
    if not 0 < args.ema_decay < 1:
        raise ValueError("--ema-decay must satisfy 0 < ema_decay < 1")
    if not args.phase2_enabled and any(
        value is not None
        for value in (
            args.phase2_loss_ct_weight,
            args.phase2_loss_spectral_weight,
            args.phase2_lr,
        )
    ):
        raise ValueError("Phase2 overrides require --phase2-enabled.")
    if args.phase2_loss_ct_weight is not None and args.phase2_loss_ct_weight > args.loss_ct_weight:
        raise ValueError("Phase 2 policy requires CT down-or-hold; --phase2-loss-ct-weight cannot exceed --loss-ct-weight")
    if (
        args.phase2_loss_spectral_weight is not None
        and args.phase2_loss_spectral_weight < args.loss_spectral_weight
    ):
        raise ValueError(
            "Phase 2 policy requires spectral up-or-hold; --phase2-loss-spectral-weight cannot be below --loss-spectral-weight"
        )
    if args.phase2_lr is not None and args.phase2_lr > args.lr:
        raise ValueError("Phase 2 policy requires LR down-or-hold; --phase2-lr cannot exceed --lr")
    if args.bank_mode == "reconstruction_faithful" and args.conditioning_mode == "none":
        raise ValueError(
            "reconstruction_faithful training requires an active conditioning mode; "
            "pass a physics_aware_* or residual_aware_* option."
        )
    return args


def main() -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:]
    args = parse_runtime_args(parser.parse_args(raw_argv), parser, raw_argv)
    trainer = Stage4SpecializationTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
