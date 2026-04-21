#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/slurm_logs"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [log_file]"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    LOG_FILE="$1"
else
    if [[ ! -d "${LOG_DIR}" ]]; then
        echo "ERROR: Log directory not found: ${LOG_DIR}"
        exit 1
    fi

    LOG_FILE="$(find "${LOG_DIR}" -maxdepth 1 -type f | sort | tail -n 1)"
    if [[ -z "${LOG_FILE}" ]]; then
        echo "ERROR: No log files found in ${LOG_DIR}"
        exit 1
    fi
fi

if [[ ! -f "${LOG_FILE}" ]]; then
    echo "ERROR: Log file not found: ${LOG_FILE}"
    exit 1
fi

echo "Monitoring: ${LOG_FILE}"
tail -n 5 -f "${LOG_FILE}"
