# ViTPose SoccerNet Pipeline

This repository contains a ViTPose-based top-down keypoint pipeline adapted for the SoccerNet SpiideoSynLoc Full HD dataset.

The main custom pieces in this workspace are:

- Full HD data preparation from the original 4K annotations
- a 2-keypoint SoccerNet configuration
- Slurm training and evaluation entrypoints
- oracle-box and YOLO-box LocSim evaluation flows
- fixed-threshold evaluation utilities

This document explains how to train, evaluate, and reuse the pipeline.

## 1. Environment

The project has been tested with:

- Python `3.8`
- PyTorch `1.9.0`
- torchvision `0.10.0`
- mmcv-full `1.3.9`
- timm `0.4.9`
- yapf `0.31.0`
- ultralytics `8.4.38`

Create and activate the conda environment:

```bash
source /home/youssef.ghallab/miniforge3/etc/profile.d/conda.sh
conda create -y -n vitpose python=3.8 pip
conda activate vitpose
```

Install GPU-compatible packages:

```bash
pip install "torch==1.9.0+cu111" "torchvision==0.10.0+cu111" -f https://download.pytorch.org/whl/torch_stable.html
pip install "mmcv-full==1.3.9" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -r requirements.txt
pip install timm==0.4.9 einops
pip install yapf==0.31.0
pip install ultralytics==8.4.38 sskit
pip install -v -e .
```

Sanity check:

```bash
pip check
python - <<'PY'
import torch, mmcv, mmpose, timm, ultralytics
print(torch.__version__, torch.cuda.is_available())
print(mmcv.__version__, mmpose.__version__, timm.__version__, ultralytics.__version__)
PY
```

## 2. Dataset Layout

Expected raw data layout:

```text
/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/
  4K/
    annotations.zip
  fullhd/
    train.zip
    val.zip
    test.zip
```

The project converts the 4K annotations to Full HD and writes:

```text
data/soccernet_fhd/
  annotations/
    train.json
    val.json
```

Prepare the Full HD annotations:

```bash
conda activate vitpose
cd /nfs-stor/youssef.ghallab/pose/ViTPose
python tools/dataset/prepare_soccernet_fullhd.py
```

If the images are already extracted, use:

```bash
python tools/dataset/prepare_soccernet_fullhd.py --skip-extract
```

## 3. Main Config

The main SoccerNet training config is:

```bash
configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py
```

This config uses:

- `ViTPose-Small`
- top-down single-person pose
- input size `256x192`
- 2 keypoints:
  - `body_anchor`
  - `ground_contact`

## 4. Training

### Local training

```bash
conda activate vitpose
cd /nfs-stor/youssef.ghallab/pose/ViTPose
export NO_ALBUMENTATIONS_UPDATE=1
python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py \
  --work-dir work_dirs/soccernet_fhd_nomosaic \
  --seed 0
```

### Slurm training

Use the provided batch script:

```bash
sbatch train_soccernet_fhd.sbatch
```

This script:

- prepares the converted annotations
- trains the no-mosaic SoccerNet config
- writes logs under `slurm_logs/`
- writes checkpoints under `work_dirs/soccernet_fhd_nomosaic/`

Monitor the latest log:

```bash
./monitor_log.sh
```

Or monitor a specific log:

```bash
./monitor_log.sh slurm_logs/<log_name>.log
```

## 5. Best Checkpoint

The best checkpoint from the current oracle-box LocSim evaluation is:

```bash
work_dirs/soccernet_fhd/epoch_30.pth
```

## 6. Evaluation Pipelines

### Oracle-box evaluation

This mode uses ground-truth bounding boxes as top-down pose inputs.

Run:

```bash
sbatch eval_soccernet_oracle_locsim.sbatch
```

Outputs are written under:

```text
evaluation/oracle_detection/
```

### YOLO-box evaluation

This mode uses the detector weights in:

```bash
YOLO_weights/best.pt
```

Run:

```bash
sbatch eval_soccernet_yolo_locsim.sbatch
```

Outputs are written under:

```text
evaluation/yolo_detection/
```

### Config-driven detector evaluation

For detector backends that do not share the same runtime as `vitpose`, use the config-driven launcher:

```bash
conda activate vitpose
python tools/analysis/eval_soccernet_from_config.py \
  --config configs/eval/soccernet_rtdetr_locsim.yaml
```

This launcher separates:

- detector box generation
- pose inference
- LocSim evaluation

The config file controls:

- detector head or backend via `detector.type`
  - `oracle`
  - `yolo`
  - `rtdetr`
- detector weights
- detector conda env
- pose checkpoint
- dataset paths
- runtime batch sizes and output directory

Example RT-DETR config:

```bash
configs/eval/soccernet_rtdetr_locsim.yaml
```

Current stable RT-DETR validation settings in that config are:

- `conf=0.2`
- `max_det=30`
- `batch_size=1`

Important note:

- official Ultralytics RT-DETR usage is through `from ultralytics import RTDETR`
- RT-DETR requires a newer PyTorch runtime than the `vitpose` env
- because of that, this launcher lets the detector run in a separate env such as `SpaTrack2`, while pose evaluation stays in `vitpose`

### Fixed-threshold evaluation

This protocol calibrates the LocSim score threshold on a random `10%` subset of the training set, then evaluates validation with that fixed threshold.

Run:

```bash
sbatch eval_locsim_train10_fixed_threshold.sbatch
```

This flow:

1. creates a random `10%` training subset
2. tunes an oracle threshold on that subset
3. tunes a YOLO threshold on that subset
4. reruns validation with those frozen thresholds
5. writes a comparison markdown

Outputs are written under:

```text
evaluation/fixed_threshold_train10/
```

## 7. Comparison Reports

Current comparison markdowns:

- `evaluation/locsim_comparison.md`
- `evaluation/fixed_threshold_train10/locsim_train10_fixed_threshold_comparison.md`

These summarize oracle-vs-YOLO behavior for the shared best checkpoint.

## 8. Test Split

The Full HD test images can be extracted from:

```bash
/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/fullhd/test.zip
```

At the moment, only test images are available locally. There is no local `test.json` annotation file yet, so full metric evaluation on test is not possible without either:

- test annotations, or
- an official hidden-test submission protocol

## 9. Useful Commands

Check a job:

```bash
squeue -j <job_id>
```

Watch a Slurm log:

```bash
tail -n 20 -f slurm_logs/<log_name>.log
```

Resume training from a checkpoint:

```bash
python tools/train.py \
  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py \
  --work-dir work_dirs/soccernet_fhd \
  --resume-from work_dirs/soccernet_fhd/epoch_30.pth \
  --seed 0
```

## 10. Key Custom Files

Dataset preparation:

- `tools/dataset/prepare_soccernet_fullhd.py`

Training:

- `train_soccernet_fhd.sbatch`
- `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py`

Evaluation:

- `eval_soccernet_oracle_locsim.sbatch`
- `eval_soccernet_yolo_locsim.sbatch`
- `eval_locsim_train10_fixed_threshold.sbatch`
- `tools/analysis/eval_soccernet_oracle_locsim.py`
- `tools/analysis/eval_soccernet_yolo_locsim.py`
- `tools/analysis/create_coco_image_subset.py`
- `tools/analysis/write_locsim_fixed_threshold_comparison.py`
