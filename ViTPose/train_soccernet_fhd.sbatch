#!/bin/bash
#SBATCH --job-name=vitpose-sn-fhd-nomosaic
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --output=/nfs-stor/youssef.ghallab/pose/ViTPose/slurm_logs/vitpose_soccernet_fhd_nomosaic_%j.log

set -euo pipefail

REPO_ROOT="/nfs-stor/youssef.ghallab/pose/ViTPose"
CONFIG_PATH="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_fhd_256x192.py"
WORK_DIR="work_dirs/soccernet_fhd_nomosaic"

cd "${REPO_ROOT}"
mkdir -p slurm_logs "${WORK_DIR}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== ViTPose SoccerNet Full HD Training (No Mosaic) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vitpose --no-capture-output python tools/dataset/prepare_soccernet_fullhd.py --skip-extract
conda run -n vitpose --no-capture-output python -u tools/train.py "${CONFIG_PATH}" --work-dir "${WORK_DIR}" --seed 0
