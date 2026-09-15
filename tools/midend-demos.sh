#!/usr/bin/env bash
# 第 5 章的中端 pass 实验: 流水线清单 / SROA+InstCombine / LoopUnroll 对照 / NVVMReflect
set -uo pipefail

CUDA=/usr/local/cuda-12.8
LLVM=/usr/local/bin
export PATH=$CUDA/bin:$LLVM:$PATH

cd "$(dirname "$0")/.."
OUT=dumps

CFLAGS=(--cuda-path=$CUDA --cuda-gpu-arch=sm_89)

echo "[1] 设备端完整 pipeline (clang -mllvm -print-pipeline-passes)"
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 -S -emit-llvm \
    -o /dev/null -mllvm -print-pipeline-passes code/tc_mma.cu \
    2>/dev/null | tr ',' '\n' > $OUT/14-device-pipeline.txt
wc -l < $OUT/14-device-pipeline.txt

echo "[2] 对未优化 IR 单独跑 sroa / mem2reg / instcombine"
# -O0 的 clang 会给每个函数打上 optnone, opt 会直接跳过它们 —— 所以先用
# -Xclang -disable-O0-optnone 生成一份"可优化"的未优化 IR
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O0 -Xclang -disable-O0-optnone \
    -S -emit-llvm -o $OUT/01b-device-O0-nooptnone.ll code/tc_mma.cu 2>/dev/null
opt -passes='sroa,mem2reg' -S $OUT/01b-device-O0-nooptnone.ll \
    -o $OUT/15-after-sroa-mem2reg.ll
opt -passes='sroa,mem2reg,instcombine' -S $OUT/01b-device-O0-nooptnone.ll \
    -o $OUT/16-after-instcombine.ll
for f in 01-device-O0.ll 01b-device-O0-nooptnone.ll 15-after-sroa-mem2reg.ll 16-after-instcombine.ll 02-device-O2.ll; do
  printf '%-32s %s 行\n' "$f" "$(wc -l < $OUT/$f)"
done

echo "[3] 汇编调用数量变化"
for f in 01-device-O0.ll 15-after-sroa-mem2reg.ll 16-after-instcombine.ll 02-device-O2.ll; do
  printf '%-24s asm 调用 = %s\n' "$f" "$(grep -c 'asm sideeffect' $OUT/$f)"
done

echo "[4] K 为编译期常量的对照实验 (unroll_demo.cu)"
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 -S -emit-llvm \
    -o $OUT/17-kconst-O2.ll code/unroll_demo.cu
printf '%-24s mma asm 调用 = %s\n' "17-kconst-O2.ll" \
       "$(grep -c 'mma\.sync' $OUT/17-kconst-O2.ll)"
printf '%-24s 循环分支 br = %s\n' "17-kconst-O2.ll" \
       "$(grep -c '^\s*br i1' $OUT/17-kconst-O2.ll)"

echo "[4b] 同一份代码, 关掉 convergent 标记后再看"
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 -fno-convergent-functions \
    -S -emit-llvm -o $OUT/20-kconst-noconvergent.ll code/unroll_demo.cu
printf '%-24s mma asm 调用 = %s\n' "20-kconst-noconvergent.ll" \
       "$(grep -c 'mma\.sync' $OUT/20-kconst-noconvergent.ll)"
printf '%-24s 循环分支 br = %s\n' "20-kconst-noconvergent.ll" \
       "$(grep -c '^\s*br i1' $OUT/20-kconst-noconvergent.ll)"
printf '%-24s convergent 出现次数 = %s\n' "20-kconst-noconvergent.ll" \
       "$(grep -c 'convergent' $OUT/20-kconst-noconvergent.ll)"

echo "[5] NVVMReflect: opt 单独跑 vs 真实 NVPTX 流水线"
opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -passes=nvvm-reflect \
    -S code/reflect_demo.ll -o $OUT/18-reflect-opt.ll
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -o $OUT/19-reflect-llc.ptx \
    code/reflect_demo.ll
grep -B2 -A8 '^arch_id' $OUT/19-reflect-llc.ptx | head -14

echo
echo "产物: 14-device-pipeline.txt 15-after-sroa-mem2reg.ll 16-after-instcombine.ll"
echo "      17-kconst-O2.ll 18-reflect-opt.ll 19-reflect-llc.ptx"
