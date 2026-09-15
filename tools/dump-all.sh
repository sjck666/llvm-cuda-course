#!/usr/bin/env bash
# 把 .cu 的完整中间产物链 dump 到 dumps/
#   IR -> PTX -> cubin -> SASS, 以及 MIR / ptxas 资源用量 / verbose 日志
set -uo pipefail

CUDA=/usr/local/cuda-12.8
LLVM=/usr/local/bin
export PATH=$CUDA/bin:$LLVM:$PATH

cd "$(dirname "$0")/.."
mkdir -p dumps
cd code
OUT=../dumps

CUDA_FLAGS=(--cuda-path=$CUDA)

# ---------- 1. LLVM IR (device only) ----------
echo "[1] device IR -O0 / -O2"
clang++ -x cuda "${CUDA_FLAGS[@]}" --cuda-device-only --cuda-gpu-arch=sm_89 \
    -O0 -S -emit-llvm -o $OUT/01-device-O0.ll tc_mma.cu
clang++ -x cuda "${CUDA_FLAGS[@]}" --cuda-device-only --cuda-gpu-arch=sm_89 \
    -O2 -S -emit-llvm -o $OUT/02-device-O2.ll tc_mma.cu

# ---------- 2. PTX ----------
echo "[2] PTX (clang -S --cuda-device-only)"
clang++ -x cuda "${CUDA_FLAGS[@]}" --cuda-device-only --cuda-gpu-arch=sm_89 \
    -O2 -S -o $OUT/03-clang-O2.ptx tc_mma.cu
echo "[2b] PTX (nvcc -ptx) 用于对照"
nvcc -arch=sm_89 -O3 -ptx -o $OUT/04-nvcc-O3.ptx tc_mma.cu

# ---------- 3. cubin + SASS ----------
echo "[3] ptxas -> cubin, 以及 SASS"
ptxas -arch=sm_89 -v -o $OUT/05-clang.cubin $OUT/03-clang-O2.ptx 2> $OUT/05-ptxas-verbose.txt
ptxas -arch=sm_89 -v -o $OUT/05-nvcc.cubin  $OUT/04-nvcc-O3.ptx  2> $OUT/05b-ptxas-verbose.txt
cuobjdump -sass $OUT/05-clang.cubin > $OUT/06-sass-cuobjdump.txt
nvdisasm -c -hex $OUT/05-clang.cubin > $OUT/07-sass-nvdisasm.txt
cuobjdump -res-usage $OUT/05-clang.cubin > $OUT/08-res-usage.txt 2>&1

# ---------- 4. Machine IR ----------
echo "[4] MIR: isel 之后 / 寄存器分配之后"
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 \
    -stop-after=finalize-isel -o $OUT/09-mir-after-isel.mir $OUT/02-device-O2.ll
# 注意: NVPTX 用自己的 prolog-epilog pass (通用 PrologEpilogInserter 被禁用了),
# 所以这里要写 nvptx-prolog-epilog, 写 prologepilog 会报 pass is not registered
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 \
    -stop-after=nvptx-prolog-epilog -o $OUT/10-mir-after-regalloc.mir $OUT/02-device-O2.ll

# ---------- 5. 完整日志 ----------
echo "[5] clang -v 完整驱动日志"
clang++ -x cuda "${CUDA_FLAGS[@]}" --cuda-device-only --cuda-gpu-arch=sm_89 \
    -O2 -S -o /dev/null -v tc_mma.cu > $OUT/11-clang-verbose.log 2>&1

# ---------- 6. host 端 IR (看 <<<>>> 怎么落成什么) ----------
echo "[6] host 端 IR"
clang++ -x cuda "${CUDA_FLAGS[@]}" --cuda-host-only -O0 -S -emit-llvm \
    -o $OUT/12-host-O0.ll tc_mma.cu

echo
ls -l $OUT | tail -20
