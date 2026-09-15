#!/usr/bin/env bash
# 构建并运行核心案例 (nvcc 与 clang 两条路径都试)
set -uo pipefail

CUDA=/usr/local/cuda-12.8
LLVM=/usr/local/bin
export PATH=$CUDA/bin:$LLVM:$PATH

cd "$(dirname "$0")/.."
mkdir -p dumps
cd code

echo "===== [1] nvcc -arch=sm_89 -O3 ====="
nvcc -arch=sm_89 -O3 -o tc_mma.nvcc tc_mma.cu && ./tc_mma.nvcc
echo "nvcc 路径 rc=$?"

echo
echo "===== [2] clang++ 作为 CUDA 编译器 (host+device) ====="
clang++ -x cuda \
  --cuda-path=$CUDA \
  --cuda-gpu-arch=sm_89 \
  -O3 \
  -L$CUDA/lib64 -lcudart \
  -o tc_mma.clang tc_mma.cu && ./tc_mma.clang
echo "clang 路径 rc=$?"
