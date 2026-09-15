#!/usr/bin/env bash
# 第 6/7 章实验: 后端流水线 stage 列表, 各阶段 MIR, PTX 发射结果
set -uo pipefail

CUDA=/usr/local/cuda-12.8
LLVM=/usr/local/bin
export PATH=$CUDA/bin:$LLVM:$PATH

cd "$(dirname "$0")/.."
OUT=dumps

LLC="llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2"

echo "[1] CodeGen 流水线里跑过的 pass (从 -print-after-all 里提取)"
$LLC -print-after-all -o /dev/null $OUT/02-device-O2.ll 2>&1 \
  | awk '/IR Dump After/ { sub(/^.*IR Dump After /, ""); sub(/ \(.*$/, ""); print }' \
  | uniq > $OUT/25-codegen-passes.txt
cat -n $OUT/25-codegen-passes.txt

echo
echo "[2] 各阶段 MIR: isel 之后 / 寄存器分配之后 / prologue-epilogue 之后"
$LLC -stop-after=finalize-isel   -o $OUT/26-isel.mir      $OUT/02-device-O2.ll
$LLC -stop-after=greedy          -o $OUT/27-regalloc.mir  $OUT/02-device-O2.ll
# NVPTX 禁用了通用的 PrologEpilogInserter, 用自己的 "NVPTX Prolog Epilog Pass",
# 所以管线里的名字是 nvptx-prolog-epilog, 不是 prologepilog
$LLC -stop-after=nvptx-prolog-epilog -o $OUT/28-prologepilog.mir $OUT/02-device-O2.ll
ls -l $OUT/26-isel.mir $OUT/27-regalloc.mir $OUT/28-prologepilog.mir

echo
echo "[3] NVPTX 专用 opcode 统计 (isel 之后)"
grep -oE 'INT_PTX_[A-Za-z_0-9]+|BARRIER_[A-Za-z_0-9]+|LD_GLOBAL_NC_[A-Za-z0-9_]+|LD_i(32|64)|ST_i32|cvta_to_global_64|MUL_WIDE[A-Za-z0-9_]+|SHL32_ri|SRL32_ri|AND_b32ri|CVT_[a-z0-9_]+|ADD(32|64)(rr|ri)|MULT32rr|SETP_i32[a-z]+' \
     $OUT/26-isel.mir | sort | uniq -c | sort -rn

echo
echo "[4] PTX 里的虚拟寄存器声明 (clang -O2 出的 PTX)"
grep -n '\.reg\|\.shared\|\.param' $OUT/03-clang-O2.ptx | head -20
