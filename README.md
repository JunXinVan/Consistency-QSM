# Consistency-QSM

Official code for:

**Consistency-QSM: Few-Step Quantitative Susceptibility Mapping with a Physics-Guided Generative Prior**
Junxin Fan, Anhong Jiang, Chenhuan Wu, Yu-Xuan Ren. *IEEE Transactions on Biomedical Engineering* (under review), 2026.
Preprint: [Zenodo, doi:10.5281/zenodo.24549979](https://doi.org/10.5281/zenodo.24549979)

Consistency-QSM distills the multi-step Diffusion-QSM generative prior into a few-step consistency student and applies physics-guided fine-tuning with data-consistency (DC) updates, reconstructing quantitative susceptibility maps in two network evaluations.

## Repository layout

```
qsm_cm/                      # consistency-model package: student U-Net, distillation, datasets
qsm_stage4/                  # physics-guided fine-tuning and inference
qsm_morph.py                 # TKD / morphology-based initialization utilities
scripts/                     # training-cache augmentation
configs/run_config.json      # full configuration record of the paper's fine-tuning run
checkpoints/                 # model weights (see checkpoints/README.md)
```

## Installation

- Python >= 3.10, PyTorch >= 2.0 with CUDA (developed on Python 3.12, torch 2.5.1+cu124)

```bash
pip install -r requirements.txt
```

- Consistency distillation additionally requires the official Diffusion-QSM teacher code: clone <https://github.com/AMRI-Lab/Diffusion-QSM>, install its own dependencies as documented there, and either pass `--diffusion-root /path/to/Diffusion-QSM` or set `export QSM_DIFFUSION_ROOT=/path/to/Diffusion-QSM`.

## Path setup

The code keeps the flat-module layout of the original development tree. From the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/qsm_stage4"
```

(The entry scripts under `qsm_stage4/` can also be invoked directly by path, e.g. `python qsm_stage4/stage4_infer_diffusionqsm.py ...`.)

## Inference

Input data follow the `data_DiffusionQSM.mat` convention of the official Diffusion-QSM repository (`phi_use`, `mask`, `D2`, `pad_size`).

```bash
python qsm_stage4/stage4_infer_diffusionqsm.py \
    --ckpt checkpoints/stage4_v3_epoch_004.pth \
    --data-root /path/to/test_data \
    --output-dir ./runs/inference
```

Paper settings: `--infer-nfe 2` with EMA weights (both default), DC policy `fixed` with the protocol-specific `--lambda-dc` reported in the paper (see also `configs/run_config.json`).

## Training

The full pipeline is (1) consistency distillation, (2) training-cache generation, (3) physics-guided fine-tuning:

```bash
# 1. Consistency distillation (requires the Diffusion-QSM teacher)
torchrun --nproc_per_node=<num_gpus> qsm_cm/consistency_distill_ddp.py \
    --data-dir /path/to/training_data --teacher-ckpt /path/to/teacher.pth ...

# 2. Training-cache generation and teacher-endpoint augmentation
python qsm_stage4/stage4_bank_gen_gt_core.py ...
python scripts/augment_v3_cache_with_teacher_endpoint.py ...

# 3. Physics-guided fine-tuning (paper configuration: configs/run_config.json)
python qsm_stage4/stage4_pcct_cg_lossplacement_experimental.py \
    --cache-dir cache/gt_core_15k_teacher_endpoint \
    --init-ckpt checkpoints/cd_ddp_epoch_018.pth \
    --output-dir runs/stage4_finetune ...
```

`configs/run_config.json` records every hyperparameter of the paper's fine-tuning run (initialization, cache layout, loss weights, learning-rate schedule, DC settings). Run any entry point with `--help` for the full argument list.

## Checkpoints

See `checkpoints/README.md`. Trained weights will be released upon acceptance of the paper.

## Data

All evaluation datasets are publicly available from their original releases (2016 and 2019 QSM Reconstruction Challenge, in vivo 1-mm isotropic and 1x1x2-mm anisotropic datasets, the 7T cohort, and the ex vivo mouse brain from the msQSM dataset); please see the paper for citations. No data are redistributed in this repository.

## License

MIT (see `LICENSE`).

## Citation

If you find this code useful, please cite the preprint:

> J. Fan, A. Jiang, C. Wu, and Y.-X. Ren, "Consistency-QSM: Few-Step Quantitative Susceptibility Mapping with a Physics-Guided Generative Prior," Zenodo (preprint), 2026, doi:10.5281/zenodo.24549979. Under review at *IEEE Transactions on Biomedical Engineering*.

```bibtex
@misc{fan2026consistencyqsm,
  author       = {Fan, Junxin and Jiang, Anhong and Wu, Chenhuan and Ren, Yu-Xuan},
  title        = {Consistency-QSM: Few-Step Quantitative Susceptibility
                  Mapping with a Physics-Guided Generative Prior},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24549979},
  url          = {https://doi.org/10.5281/zenodo.24549979},
  note         = {Preprint. Under review at IEEE Transactions on Biomedical Engineering}
}
```
