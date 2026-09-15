#!/usr/bin/env bash
# 第 5 章核心实验: convergent 到底挡住了哪些优化
#   A) 原样: K 编译期常量, 循环仍不展开
#   B) 用 pass remark 问编译器"为什么不展开"
#   C) 手工把 convergent 从 IR 里去掉, 再跑一遍 loop-unroll, 看是否展开
set -uo pipefail

CUDA=/usr/local/cuda-12.8
LLVM=/usr/local/bin
export PATH=$CUDA/bin:$LLVM:$PATH

cd "$(dirname "$0")/.."
OUT=dumps
CFLAGS=(--cuda-path=$CUDA --cuda-gpu-arch=sm_89)

echo "===== A) K 编译期常量, 默认(CUDA 设备端 asm 带 convergent) ====="
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 -S -emit-llvm \
    -o $OUT/17-kconst-O2.ll code/unroll_demo.cu 2>/dev/null
echo "mma asm 调用 = $(grep -c 'mma\.sync' $OUT/17-kconst-O2.ll)"
echo "循环 br i1  = $(grep -c 'br i1' $OUT/17-kconst-O2.ll)"

echo
echo "===== B) pass remark: 让 loop-unroll 自己说他为什么不展开 ====="
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 \
    -mllvm -pass-remarks-missed=loop-unroll \
    -mllvm -pass-remarks-analysis=loop-unroll \
    -S -emit-llvm -o /dev/null code/unroll_demo.cu 2>&1 \
  | grep -i 'unroll' | head -10

echo
echo "===== C) 把 convergent 从 IR 里抹掉, 再让 opt 跑 loop-unroll ====="
grep -v '^declare.*llvm.nvvm' $OUT/17-kconst-O2.ll \
  | sed -e 's/^\(attributes #[0-9]* = {\) convergent /\1 /' \
        -e 's/, convergent /, /' -e 's/ convergent / /' \
  > $OUT/21-kconst-noconvergent-ir.ll
echo "-- 抹掉后仍有 convergent 的行数: $(grep -c 'convergent' $OUT/21-kconst-noconvergent-ir.ll)"
opt -passes='loop-unroll' -S $OUT/21-kconst-noconvergent-ir.ll \
    -o $OUT/22-unrolled.ll 2>&1 | head -5
echo "-- 原文件跑 loop-unroll 之后: mma = $(opt -passes='loop-unroll' -S $OUT/17-kconst-O2.ll -o - 2>/dev/null | grep -c 'mma\.sync')"
echo "-- 去掉 convergent 之后:      mma = $(grep -c 'mma\.sync' $OUT/22-unrolled.ll)"
echo "-- 去掉 convergent 之后:      br i1 = $(grep -c 'br i1' $OUT/22-unrolled.ll)"

echo
echo "===== D) 区分'合法性'与'收益': 放开代价模型再试 ====="
for f in 17-kconst-O2.ll 21-kconst-noconvergent-ir.ll; do
  full=$(opt -passes='loop-unroll-full' -S $OUT/$f -o - 2>/dev/null | grep -c 'mma\.sync')
  big=$(opt -passes='loop-unroll' -unroll-threshold=1000000 -S $OUT/$f -o - 2>/dev/null | grep -c 'mma\.sync')
  cnt=$(opt -passes='loop-unroll' -unroll-count=4 -unroll-threshold=1000000 -S $OUT/$f -o - 2>/dev/null | grep -c 'mma\.sync')
  c4=$(opt -passes='loop-unroll' -unroll-count=4 -S $OUT/$f -o - 2>/dev/null | grep -c 'mma\.sync')
  printf '%-34s full=%-3s count=4(默认阈值)=%-3s threshold=1e6 -> %-3s count=4+阈值 -> %s\n' \
         "$f" "$full" "$c4" "$big" "$cnt"
done

echo
echo "===== E) loop-unroll-full 对含 convergent 循环的处理 (pass remark) ====="
opt -passes='loop-unroll-full' -pass-remarks-missed=loop-unroll \
    -pass-remarks-analysis=loop-unroll -S $OUT/17-kconst-O2.ll -o /dev/null 2>&1 \
  | grep -i 'unroll' | head -8

echo
echo "===== F) #pragma unroll 4 能不能让 IR 层展开 ====="
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 -S -emit-llvm \
    -o $OUT/23-unroll-pragma.ll code/unroll_pragma_demo.cu 2>/dev/null
echo "unroll_demo.cu        (无 pragma): mma 条数 = $(grep -c 'mma\.sync' $OUT/17-kconst-O2.ll)"
echo "unroll_pragma_demo.cu (有 pragma): mma 条数 = $(grep -c 'mma\.sync' $OUT/23-unroll-pragma.ll)"
echo "unroll_pragma_demo.cu             循环 br = $(grep -c 'br i1' $OUT/23-unroll-pragma.ll)"

echo
echo "===== G) 三条摘掉 convergent 的路, 到底哪条有效 ====="
echo "-- (1) __attribute__((noconvergent)) 放在 asm 前 (预期: 语法报错)"
printf '%s\n' \
  '#include <cuda_runtime.h>' \
  'extern "C" __global__ void k(float *o, float *i) {' \
  '  float x = i[threadIdx.x];' \
  '  __attribute__((noconvergent))' \
  '  asm volatile("add.f32 %0, %1, %1;" : "+f"(x) : "f"(x));' \
  '  o[threadIdx.x] = x;' \
  '}' > /tmp/nc_attr.cu
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 -S -emit-llvm \
    -o /dev/null /tmp/nc_attr.cu 2>&1 | head -4

echo
echo "-- (2)(3) 见 code/noconvergent_demo.cu 里两个 kernel:"
clang++ -x cuda "${CFLAGS[@]}" --cuda-device-only -O2 -S -emit-llvm \
    -o $OUT/24-noconvergent.ll code/noconvergent_demo.cu 2>&1 \
  | grep -i 'ignored' | head -2
echo
echo "-- 每个 kernel 里那条 add.f32 的 asm, 用的是哪个属性组:"
for fn in with_convergent with_noconvergent with_noconvergent_stmt; do
  attr=$(sed -n "/@$fn(/,/^}/p" $OUT/24-noconvergent.ll \
         | grep -o 'add.f32[^#]*#[0-9]*' | grep -o '#[0-9]*$')
  has=$(grep "^attributes $attr = " $OUT/24-noconvergent.ll | grep -c convergent)
  printf '  %-24s 属性组 %-3s 含 convergent: %s\n' "$fn" "$attr" \
         "$([ "$has" = 1 ] && echo 是 || echo 否)"
done
echo
echo "-- 属性组定义:"
grep '^attributes #' $OUT/24-noconvergent.ll
