#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --time=00:15:00

set -e # Exit immediately if a pipeline returns a non-zero status.

# Usage: sbatch experiment_job.sh <experiment_dir> <start_iter> <communication_socket>
EXP_DIR=$1
ITER=$2
SOCK=$3

# CUDA_SOURCE_PATH=/path/to/cuda/setup/script.sh
# CONDA_PATH=/path/to/conda/directory

if [ -n "${CUDA_SOURCE_PATH:-}" ]; then
    source "$CUDA_SOURCE_PATH"
else
    echo "Using CUDA from the activated conda environment"
fi

if [ -v CONDA_PATH ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
else
    echo "CONDA_PATH not set, conda must already be active"
fi

conda activate "${CONDA_ENV:-nccl}"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Wait up to 60 seconds for NFS to sync the generated directory and source files
TIMEOUT=60
ELAPSED=0
while [ ! -f "$EXP_DIR/experiment.cu" ]; do
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Error: Timeout ($TIMEOUT seconds) waiting for NFS to sync $EXP_DIR/experiment.cu" >&2
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

cd "$EXP_DIR"

# Compile native GPU code to avoid requiring a newer-driver PTX JIT.
nvcc --version
"${CXX:-g++}" --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# Resolve development/runtime files from the selected nvcc.
CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
CUDA_INCLUDE=
for candidate in "$CUDA_ROOT/targets/x86_64-linux/include" "$CUDA_ROOT/include"; do
    if [ -f "$candidate/cuda_runtime.h" ]; then
        CUDA_INCLUDE="$candidate"
        break
    fi
done
CUDA_LIB=
for candidate in "$CUDA_ROOT/targets/x86_64-linux/lib" "$CUDA_ROOT/lib64" "$CUDA_ROOT/lib"; do
    if compgen -G "$candidate/libcudart.so*" >/dev/null; then
        CUDA_LIB="$candidate"
        break
    fi
done
if [ -z "$CUDA_INCLUDE" ] || [ -z "$CUDA_LIB" ]; then
    echo "CUDA development headers/runtime library missing beneath $CUDA_ROOT" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$CUDA_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvcc -std=c++20 -ccbin "${CXX:-g++}" -arch=native -DNCCL_KERNELS_ONLY \
    -c experiment.cu -o experiment_kernels.o
"${CXX:-g++}" -std=c++20 -I"$CUDA_INCLUDE" -I"$CONDA_PREFIX/include" \
    -x c++ -c experiment.cu -o experiment_host.o
"${CXX:-g++}" -std=c++20 -I"$CUDA_INCLUDE" -I"$CONDA_PREFIX/include" \
    experiment_host.o experiment_kernels.o experiment.pb.cc -o experiment.bin \
    -L"$CUDA_LIB" -L"$CONDA_PREFIX/lib" \
    -lcudart -lnccl -lzmq -lprotobuf \
    -labsl_log_internal_check_op -labsl_log_internal_message

srun --unbuffered -n 1 ./experiment.bin "$ITER" "$SOCK"
