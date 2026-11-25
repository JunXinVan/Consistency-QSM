# Checkpoints

Two checkpoints are used in the paper:

| File | Role |
|---|---|
| `cd_ddp_epoch_018.pth` | Distilled consistency student (initialization for fine-tuning) |
| `stage4_v3_epoch_004.pth` | Final fine-tuned model used for all reported results |

**Release status:** the trained weights will be uploaded here upon acceptance of the paper. The released files will contain model/EMA weights only (optimizer states trimmed); md5 checksums will be listed here at that time.

**Loading:** use the provided loader (`load_student_components` in `qsm_stage4/qsm_bank_utils.py`), which handles the DDP/`base_model.` key prefix and first-convolution channel expansion automatically.
