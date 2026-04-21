#!/bin/bash
#SBATCH --job-name=vitpose-locsim-fixed
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --output=/nfs-stor/youssef.ghallab/pose/ViTPose/slurm_logs/vitpose_locsim_fixed_%j.log

set -euo pipefail

REPO_ROOT="/nfs-stor/youssef.ghallab/pose/ViTPose"
CONFIG_PATH="${REPO_ROOT}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py"
CHECKPOINT_PATH="${REPO_ROOT}/work_dirs/soccernet_fhd/epoch_30.pth"
YOLO_WEIGHTS="${REPO_ROOT}/YOLO_weights/best.pt"

TRAIN_ANN="${REPO_ROOT}/data/soccernet_fhd/annotations/train.json"
VAL_ANN="${REPO_ROOT}/data/soccernet_fhd/annotations/val.json"
TRAIN_IMG_PREFIX="/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/fullhd/train/"
VAL_IMG_PREFIX="/nfs-stor/youssef.ghallab/pose/data/SoccerNet/SpiideoSynLoc/fullhd/val/"

OUTPUT_ROOT="${REPO_ROOT}/evaluation/fixed_threshold_train10"
TRAIN_SUBSET_ANN="${OUTPUT_ROOT}/train10_seed0.json"
ORACLE_TUNE_DIR="${OUTPUT_ROOT}/oracle_tune"
ORACLE_VAL_DIR="${OUTPUT_ROOT}/oracle_val_fixed"
YOLO_TUNE_DIR="${OUTPUT_ROOT}/yolo_tune"
YOLO_VAL_DIR="${OUTPUT_ROOT}/yolo_val_fixed"
COMPARISON_MD="${OUTPUT_ROOT}/locsim_train10_fixed_threshold_comparison.md"

cd "${REPO_ROOT}"
mkdir -p slurm_logs "${OUTPUT_ROOT}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== ViTPose LocSim Fixed Threshold Evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vitpose --no-capture-output python tools/dataset/prepare_soccernet_fullhd.py --skip-extract

conda run -n vitpose --no-capture-output python tools/analysis/create_coco_image_subset.py \
  --input-ann "${TRAIN_ANN}" \
  --output-ann "${TRAIN_SUBSET_ANN}" \
  --ratio 0.1 \
  --seed 0

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_oracle_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --ann-file "${TRAIN_SUBSET_ANN}" \
  --img-prefix "${TRAIN_IMG_PREFIX}" \
  --output-dir "${ORACLE_TUNE_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 16 \
  --workers-per-gpu 4 \
  --overwrite

ORACLE_THRESH=$(conda run -n vitpose --no-capture-output python - <<PY
import json
from pathlib import Path
p = Path("${ORACLE_TUNE_DIR}") / "metrics" / "work_dirs__soccernet_fhd__epoch_30.json"
print(json.loads(p.read_text())["named_stats"]["score_threshold"])
PY
)
echo "Oracle threshold: ${ORACLE_THRESH}"

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_oracle_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --ann-file "${VAL_ANN}" \
  --img-prefix "${VAL_IMG_PREFIX}" \
  --output-dir "${ORACLE_VAL_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 16 \
  --workers-per-gpu 4 \
  --score-threshold "${ORACLE_THRESH}" \
  --overwrite

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_yolo_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --detector-weights "${YOLO_WEIGHTS}" \
  --ann-file "${TRAIN_SUBSET_ANN}" \
  --img-prefix "${TRAIN_IMG_PREFIX}" \
  --output-dir "${YOLO_TUNE_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 16 \
  --workers-per-gpu 4 \
  --detector-batch-size 16 \
  --threshold-tune-ratio 0 \
  --overwrite

YOLO_THRESH=$(conda run -n vitpose --no-capture-output python - <<PY
import json
from pathlib import Path
p = Path("${YOLO_TUNE_DIR}") / "metrics" / "epoch_30.json"
print(json.loads(p.read_text())["named_stats"]["score_threshold"])
PY
)
echo "YOLO threshold: ${YOLO_THRESH}"

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_yolo_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --detector-weights "${YOLO_WEIGHTS}" \
  --ann-file "${VAL_ANN}" \
  --img-prefix "${VAL_IMG_PREFIX}" \
  --output-dir "${YOLO_VAL_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 16 \
  --workers-per-gpu 4 \
  --detector-batch-size 16 \
  --threshold-tune-ratio 0 \
  --score-threshold "${YOLO_THRESH}" \
  --overwrite

conda run -n vitpose --no-capture-output python tools/analysis/write_locsim_fixed_threshold_comparison.py \
  --output-md "${COMPARISON_MD}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --train-subset-ann "${TRAIN_SUBSET_ANN}" \
  --oracle-tune-metrics "${ORACLE_TUNE_DIR}/metrics/work_dirs__soccernet_fhd__epoch_30.json" \
  --oracle-val-metrics "${ORACLE_VAL_DIR}/metrics/work_dirs__soccernet_fhd__epoch_30.json" \
  --yolo-tune-metrics "${YOLO_TUNE_DIR}/metrics/epoch_30.json" \
  --yolo-val-metrics "${YOLO_VAL_DIR}/metrics/epoch_30.json" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --seed 0 \
  --ratio 0.1
