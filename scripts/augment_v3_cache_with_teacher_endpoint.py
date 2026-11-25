#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


def atomic_save_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp_path.replace(path)


def ensure_link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def parse_triplet(raw: str, cast):
    parts = [cast(part.strip()) for part in str(raw).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got {raw!r}")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone an existing stage4 v3 cache and add teacher_endpoint arrays per shard."
    )
    parser.add_argument("--src-cache-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--diffusion-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher-fast-timesteps", type=int, default=200)
    parser.add_argument("--teacher-starting-point", type=int, default=10)
    parser.add_argument("--teacher-repetition-time", type=int, default=2)
    parser.add_argument("--teacher-eta", type=float, default=0.0)
    parser.add_argument("--teacher-lambda-dc", type=float, default=0.1)
    parser.add_argument("--teacher-cg-steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--start-shard", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    src_cache_dir = Path(args.src_cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = src_cache_dir / "manifest_rank_0.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "qsm_stage4_v3_cache":
        raise RuntimeError(f"Unsupported source cache format: {manifest.get('format')!r}")

    patch_size = tuple(int(v) for v in manifest.get("patch_size", [64, 64, 64]))
    voxel_size = tuple(float(v) for v in manifest.get("voxel_size", [1.0, 1.0, 1.0]))
    b0_dir = tuple(float(v) for v in manifest.get("b0_dir", [0.0, 0.0, 1.0]))

    src_shards = list(manifest.get("shards", []))
    if not src_shards:
        raise RuntimeError("Source cache manifest has no shards.")

    end_index = len(src_shards)
    if args.max_shards is not None:
        end_index = min(end_index, args.start_shard + int(args.max_shards))
    selected = list(range(int(args.start_shard), end_index))
    if not selected:
        raise RuntimeError("No shards selected for augmentation.")

    import sys

    sys.path.insert(0, str(src_cache_dir.parents[2] / "src" / "qsm_stage4"))
    from stage4_bank_gen_resume import DiffusionQSMTeacherBackend
    from qsm_bank_utils import create_dipole_kernel

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    teacher = DiffusionQSMTeacherBackend(
        teacher_ckpt=args.teacher_ckpt,
        diffusion_root=args.diffusion_root,
        device=device,
        patch_size=patch_size,
        fast_timesteps=args.teacher_fast_timesteps,
        starting_point=args.teacher_starting_point,
        repetition_time=args.teacher_repetition_time,
        eta=args.teacher_eta,
        dc_lam=args.teacher_lambda_dc,
        dc_cg_steps=args.teacher_cg_steps,
    )
    dipole_kernel = create_dipole_kernel(
        patch_size,
        voxel_size=voxel_size,
        b0_dir=b0_dir,
        device=device,
        dtype=torch.float32,
    )

    output_manifest = dict(manifest)
    cache_metadata = dict(output_manifest.get("cache_metadata") or {})
    cache_metadata.update(
        {
            "teacher_endpoint": True,
            "teacher_backend": "diffusion_qsm",
            "teacher_backend_meta": teacher.manifest_metadata(),
            "generator": "augment_v3_cache_with_teacher_endpoint.py",
            "source_cache_dir": str(src_cache_dir),
        }
    )
    output_manifest["cache_metadata"] = cache_metadata

    output_shards = []
    start_time = time.time()
    completed = 0
    total_samples = 0
    for shard_idx in selected:
        src_shard = src_shards[shard_idx]
        file_map = dict(src_shard["files"])
        prefix = str(src_shard["prefix"])
        teacher_key = "teacher_endpoint"
        teacher_rel = f"{prefix}_teacher_endpoint.npy"
        teacher_path = output_dir / teacher_rel

        for key, rel in file_map.items():
            ensure_link(src_cache_dir / rel, output_dir / rel)

        file_map[teacher_key] = teacher_rel
        shard_meta_path = output_dir / f"{prefix}_meta.json"

        shard_format = str(src_shard.get("format", manifest.get("format", "qsm_stage4_v3_cache")))

        if teacher_path.exists():
            shard_meta = {
                "prefix": prefix,
                "count": int(src_shard["count"]),
                "format": shard_format,
                "has_teacher_endpoint": True,
                "files": file_map,
            }
            atomic_save_json(shard_meta_path, shard_meta)
            output_shards.append(shard_meta)
            completed += 1
            total_samples += int(src_shard["count"])
            continue

        field = np.load(src_cache_dir / src_shard["files"]["field"], mmap_mode="r")
        mask = np.load(src_cache_dir / src_shard["files"]["mask"], mmap_mode="r")
        x_init = np.load(src_cache_dir / src_shard["files"]["x_init"], mmap_mode="r")
        sample_ids = np.load(src_cache_dir / src_shard["files"]["sample_id"], mmap_mode="r")
        count = int(src_shard["count"])
        endpoints = np.empty((count, *patch_size), dtype=np.float16)

        shard_t0 = time.time()
        for start in range(0, count, int(args.batch_size)):
            stop = min(count, start + int(args.batch_size))
            batch_field = torch.from_numpy(np.array(field[start:stop], dtype=np.float32, copy=True)).to(device).unsqueeze(1)
            batch_mask = torch.from_numpy(np.array(mask[start:stop], dtype=np.float32, copy=True)).to(device).unsqueeze(1)
            batch_x_init = torch.from_numpy(np.array(x_init[start:stop], dtype=np.float32, copy=True)).to(device).unsqueeze(1)
            seed = int(args.seed + int(sample_ids[start]))
            record = teacher.generate(
                x_init=batch_x_init,
                field_patch=batch_field,
                mask_patch=batch_mask,
                dipole_kernel=dipole_kernel,
                sample_seed=seed,
            )
            final_state = record.states[-1].detach().cpu().numpy().astype(np.float16)
            endpoints[start:stop] = final_state[:, 0]
            if device.type == "cuda":
                torch.cuda.empty_cache()

        atomic_save_npy(teacher_path, endpoints)
        shard_meta = {
            "prefix": prefix,
            "count": count,
            "format": shard_format,
            "has_teacher_endpoint": True,
            "files": file_map,
        }
        atomic_save_json(shard_meta_path, shard_meta)
        output_shards.append(shard_meta)
        completed += 1
        total_samples += count

        elapsed = time.time() - shard_t0
        overall = time.time() - start_time
        sec_per_patch = elapsed / max(count, 1)
        eta_shards = len(selected) - completed
        eta_hours = (eta_shards * elapsed) / 3600.0
        print(
            f"[teacher-cache] shard={shard_idx:04d} done | "
            f"samples={count} | {sec_per_patch:.2f}s/patch | "
            f"elapsed={elapsed/60:.1f}m | overall={overall/3600:.2f}h | eta~{eta_hours:.1f}h",
            flush=True,
        )

    # Preserve unselected shards as linked base-only entries only if max_shards is not set.
    if args.max_shards is None and args.start_shard == 0:
        for shard_idx, src_shard in enumerate(src_shards):
            if shard_idx in selected:
                continue
            file_map = dict(src_shard["files"])
            for _, rel in file_map.items():
                ensure_link(src_cache_dir / rel, output_dir / rel)
            shard_meta = {
                "prefix": str(src_shard["prefix"]),
                "count": int(src_shard["count"]),
                "format": str(src_shard.get("format", manifest.get("format", "qsm_stage4_v3_cache"))),
                "has_teacher_endpoint": False,
                "files": file_map,
            }
            atomic_save_json(output_dir / f"{src_shard['prefix']}_meta.json", shard_meta)
            output_shards.append(shard_meta)

    output_manifest["shards"] = output_shards
    output_manifest["cache_metadata"] = cache_metadata
    atomic_save_json(output_dir / "manifest_rank_0.json", output_manifest)
    print(
        f"[teacher-cache] complete | shards={completed}/{len(selected)} | "
        f"samples={total_samples} | output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
