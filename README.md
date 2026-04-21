# Pose Workspace

This repository is the top-level workspace for the SoccerNet / SpiideoSynLoc experiments in this project.

It contains three main components:

- [`ViTPose/`](./ViTPose): the main top-down pose pipeline, training code, Slurm jobs, and LocSim evaluation code.
- [`yolo26det/`](./yolo26det): Ultralytics detector-side scripts used as a lightweight detector baseline workspace.
- [`RT_DETR/`](./RT_DETR): RT-DETR weights used by the config-driven detector-to-pose evaluation flow.

## Where To Start

If you are new to the repo, the best entry points are:

- [`ViTPose/README_PROJECT.md`](./ViTPose/README_PROJECT.md): project-specific guide for training, evaluation, detector integration, and current experiment layout.
- [`ViTPose/README.md`](./ViTPose/README.md): upstream ViTPose / MMPose documentation.
- [`yolo26det/README.md`](./yolo26det/README.md): detector-side notes for the Ultralytics scripts copied into this workspace.

## Recommended Navigation

### 1. Main project code

Most of the active work in this workspace lives under [`ViTPose/`](./ViTPose).

Useful places inside it:

- [`ViTPose/configs/`](./ViTPose/configs): model, dataset, and evaluation configs.
- [`ViTPose/tools/`](./ViTPose/tools): training, dataset conversion, evaluation, and analysis scripts.
- [`ViTPose/train_soccernet_fhd.sbatch`](./ViTPose/train_soccernet_fhd.sbatch): main training batch job.
- [`ViTPose/eval_soccernet_yolo_test_fixed_threshold.sbatch`](./ViTPose/eval_soccernet_yolo_test_fixed_threshold.sbatch): fixed-threshold YOLO test job.
- [`ViTPose/eval_soccernet_rtdetr_test_fixed_threshold.sbatch`](./ViTPose/eval_soccernet_rtdetr_test_fixed_threshold.sbatch): fixed-threshold RT-DETR test job.
- [`ViTPose/eval_soccernet_oracle_test_fixed_threshold.sbatch`](./ViTPose/eval_soccernet_oracle_test_fixed_threshold.sbatch): fixed-threshold oracle test job.

### 2. Detector-side scripts

The folder [`yolo26det/`](./yolo26det) contains compact Ultralytics scripts that were copied into this workspace for detector training/evaluation experiments.

Start with:

- [`yolo26det/README.md`](./yolo26det/README.md)
- [`yolo26det/train.py`](./yolo26det/train.py)
- [`yolo26det/evaluate_spiideo_val.py`](./yolo26det/evaluate_spiideo_val.py)

### 3. Evaluation outputs

Generated evaluation reports and metrics are usually written under:

- [`ViTPose/evaluation/`](./ViTPose/evaluation)

Common subfolders:

- [`ViTPose/evaluation/oracle_detection/`](./ViTPose/evaluation/oracle_detection)
- [`ViTPose/evaluation/yolo_detection/`](./ViTPose/evaluation/yolo_detection)
- [`ViTPose/evaluation/rtdetr_detection/`](./ViTPose/evaluation/rtdetr_detection)

### 4. Model assets

- [`RT_DETR/`](./RT_DETR): RT-DETR checkpoint storage.
- `ViTPose/YOLO_weights/`: YOLO detector checkpoints used by the ViTPose evaluation scripts.

## Notes On Large Artifacts

This repo intentionally ignores:

- datasets
- model weights
- Slurm logs
- generated reports
- training outputs

Those files are expected to live locally in the workspace but not be versioned in Git.

## Practical Workflow

For most tasks, follow this path:

1. Read [`ViTPose/README_PROJECT.md`](./ViTPose/README_PROJECT.md).
2. Choose the relevant config under [`ViTPose/configs/`](./ViTPose/configs).
3. Launch training or evaluation using the matching script under [`ViTPose/`](./ViTPose).
4. Inspect outputs under [`ViTPose/evaluation/`](./ViTPose/evaluation) or local Slurm logs.

If you specifically need the detector scripts rather than the pose pipeline, use [`yolo26det/README.md`](./yolo26det/README.md).
