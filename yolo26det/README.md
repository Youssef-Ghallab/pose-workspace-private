# yolo26det

This directory contains the Ultralytics-based bottom-up / end-to-end pose-estimation branch used in this workspace.

This branch is different from the top-down `ViTPose` track in [`../ViTPose/`](../ViTPose):

- `yolo26det` represents the end-to-end Ultralytics approach
- `ViTPose` represents the detector-plus-pose top-down approach

This code is my colleague's work and should be read as one of the two implemented techniques in the project, not as a minor side folder.

If you want the workspace-level overview, start from:

- [`../README.md`](../README.md)

## Files

- [`train.py`](./train.py): Ultralytics training launcher from this branch. It currently contains hard-coded paths/settings from the source environment and should be adapted before reuse.
- [`evaluate_spiideo_val.py`](./evaluate_spiideo_val.py): validation-time end-to-end Ultralytics pose evaluation script with LocSim scoring.

## What This Folder Is For

Use this directory if you want to:

- inspect the bottom-up / end-to-end technique
- run or adapt the Ultralytics-side training/evaluation code
- compare this branch against the top-down `ViTPose` branch

## Relationship To ViTPose

The top-down branch lives under [`../ViTPose/`](../ViTPose), especially:

- [`../ViTPose/README_PROJECT.md`](../ViTPose/README_PROJECT.md)
- [`../ViTPose/tools/analysis/eval_soccernet_from_config.py`](../ViTPose/tools/analysis/eval_soccernet_from_config.py)
- [`../ViTPose/configs/eval/`](../ViTPose/configs/eval)

That branch is the one used for:

- oracle-box evaluation
- YOLO-box evaluation
- RT-DETR-box evaluation
- fixed-threshold validation/test runs

So the two folders should be read as two separate methodological directions in the same workspace:

- [`./`](./): bottom-up / end-to-end Ultralytics branch
- [`../ViTPose/`](../ViTPose): top-down detector-plus-pose branch

## Caution

Some scripts here contain hard-coded paths and training settings from the original source environment. Before reusing them directly, check:

- model checkpoint paths
- dataset paths
- output locations
- W\&B login / experiment naming

For reproducibility, document any local path changes you make before rerunning this branch.
