#!/bin/bash
#SBATCH --job-name=vitpose-oracle-test-fixed
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --output=/nfs-stor/youssef.ghallab/pose/ViTPose/slurm_logs/vitpose_oracle_test_fixed_%j.log

set -euo pipefail

REPO_ROOT="/nfs-stor/youssef.ghallab/pose/ViTPose"
CONFIG_PATH="${REPO_ROOT}/configs/eval/soccernet_oracle_test_fixed_threshold.yaml"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== ViTPose SoccerNet Oracle Test Fixed-Threshold Evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_from_config.py \
  --config "${CONFIG_PATH}"
