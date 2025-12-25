#!/bin/bash
# Wrapper script for NL_julia_gpu.jl
# Automatically sets JULIA_NUM_THREADS based on --gpu parameter

# Default to 1 GPU
n_gpus=1

# Parse --gpu argument to count GPUs
for arg in "$@"; do
    if [[ "$arg" == --gpu=* ]]; then
        gpu_str="${arg#--gpu=}"
        n_gpus=${#gpu_str}  # Length of string like "01" = 2 GPUs
    fi
done

JULIA_NUM_THREADS=$n_gpus exec julia --project=. "$(dirname "$0")/NL_julia_gpu.jl" "$@"
