#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
CONDA_SHELL="$(conda info --base)/etc/profile.d/conda.sh"
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate seed-env
 elif [[ "${CONDA_PREFIX}" != *"/seed-env" ]]; then
  conda deactivate
  conda activate seed-env
fi

# W&B / HF Setup
source "${HOME}/setup_seed.sh"
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${HF_HOME}/modules"

# Enforce verbose Hydra error logging
export HYDRA_FULL_ERROR=1

export NCCL_P2P_LEVEL=NVL
