# Pose Workspace

This repository is the top-level workspace for the SoccerNet / SpiideoSynLoc experiments in this project.

It contains two main methodological tracks plus the detector weights used by the top-down track:

- [`yolo26det/`](./yolo26det): the bottom-up / end-to-end Ultralytics pose-estimation line of work developed by my colleague. This path predicts pose directly in one model family and serves as one of the two main techniques in the workspace.
- [`ViTPose/`](./ViTPose): the top-down pipeline, where person boxes are provided by a detector and then refined with ViTPose for keypoint prediction and downstream localization.

## Where To Start

If you are new to the repo, the best entry points are:

- [`yolo26det/README.md`](./yolo26det/README.md): overview of the bottom-up / end-to-end Ultralytics branch.
- [`ViTPose/README_PROJECT.md`](./ViTPose/README_PROJECT.md): project-specific guide for training, evaluation, detector integration, and current experiment layout.
- [`ViTPose/README.md`](./ViTPose/README.md): upstream ViTPose / MMPose documentation.

## Recommended Navigation

### 1. Bottom-up / End-to-end branch

The folder [`yolo26det/`](./yolo26det) contains the Ultralytics-based end-to-end pose baseline.

This branch is useful if you want to:

- inspect the bottom-up / end-to-end approach in this workspace
- see the earlier Ultralytics training and evaluation scripts
- compare that approach against the top-down `ViTPose` pipeline

Start with:

- [`yolo26det/README.md`](./yolo26det/README.md)
- [`yolo26det/train.py`](./yolo26det/train.py)
- [`yolo26det/evaluate_spiideo_val.py`](./yolo26det/evaluate_spiideo_val.py)

### 2. Top-down branch

The folder [`ViTPose/`](./ViTPose) contains the detector-plus-pose pipeline.

Useful places inside it:

- [`ViTPose/configs/`](./ViTPose/configs): model, dataset, and evaluation configs.
- [`ViTPose/tools/`](./ViTPose/tools): training, dataset conversion, evaluation, and analysis scripts.
- [`ViTPose/train_soccernet_fhd.sbatch`](./ViTPose/train_soccernet_fhd.sbatch): main training batch job.
- [`ViTPose/eval_soccernet_yolo_test_fixed_threshold.sbatch`](./ViTPose/eval_soccernet_yolo_test_fixed_threshold.sbatch): fixed-threshold YOLO test job.
- [`ViTPose/eval_soccernet_rtdetr_test_fixed_threshold.sbatch`](./ViTPose/eval_soccernet_rtdetr_test_fixed_threshold.sbatch): fixed-threshold RT-DETR test job.
- [`ViTPose/eval_soccernet_oracle_test_fixed_threshold.sbatch`](./ViTPose/eval_soccernet_oracle_test_fixed_threshold.sbatch): fixed-threshold oracle test job.

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

For most tasks, choose the branch you care about first:

1. If you want the bottom-up / end-to-end Ultralytics approach, start with [`yolo26det/README.md`](./yolo26det/README.md).
2. If you want the top-down detector-plus-ViTPose approach, start with [`ViTPose/README_PROJECT.md`](./ViTPose/README_PROJECT.md).
3. Use the matching scripts and configs for that branch.
4. Inspect outputs under [`ViTPose/evaluation/`](./ViTPose/evaluation) or the relevant local logs.

The repo is meant to document both techniques, not only `ViTPose`, so both entrypoints are first-class depending on which method you want to work with.
