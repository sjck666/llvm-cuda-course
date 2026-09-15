#!/usr/bin/env bash
# 第 10 章实验: 探测手上这套 LLVM 里 NVVM intrinsic 的真实名字
#   - 工具链和源码树是同一份 (llvm-project 源码就地构建), 名字直接以编译器为准
set -uo pipefail
LLVM=/usr/local/bin
SRC=/root/llvm-project
export PATH=$LLVM:$PATH

cd "$(dirname "$0")/.."
OUT=dumps

probe() {   # probe <intrinsic 名> <返回类型> <参数列表>
  local name=$1 ret=$2 args=$3 f ptx
  f=$(mktemp /tmp/probeXXXX.ll)
  # 注意: 参数类型必须和 declare 里写的一致(用同一个 %s 填进去), 否则是类型不匹配,
  #       跟"名字对不对"这件事就混在一起了
  printf 'target triple = "nvptx64-nvidia-cuda"\ndeclare %s @%s(%s)\ndefine %s @use(%s %%p) {\n  %%r = call %s @%s(%s %%p)\n  ret %s %%r\n}\n' \
    "$ret" "$name" "$args" "$ret" "$args" "$ret" "$name" "$args" "$ret" > "$f"
  ptx=$(llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 -o - "$f" 2>&1)
  # 真正的指令行长这样: 前导空白 + ldmatrix.sync.aligned... ；被当成普通调用时是
  # ".extern .func llvm.nvvm.ldmatrix..." 和一条 call
  if echo "$ptx" | grep -qE '^[[:space:]]+(ldmatrix|mma)\.sync\.aligned'; then
    echo "OK    $name"
    echo "$ptx" | grep -E '^[[:space:]]+(ldmatrix|mma)\.sync\.aligned' | head -2 | sed 's/^/        /'
  else
    echo "FAIL  $name"
    echo "$ptx" | grep -iE 'error' | head -1 | sed 's/^/        /'
  fi
  rm -f "$f"
}

echo "===== ldmatrix.x4.b16: 三组对照(名字/签名 对与不对) ====="
# A) 和 IntrinsicsNVVM.td 里的定义完全一致 -> 应该出指令
probe 'llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16'        '{i32, i32, i32, i32}' 'ptr addrspace(3)'
# B) 名字照 PTX 猜, 多了 .shared -> 后端不认
probe 'llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.shared.b16' '{i32, i32, i32, i32}' 'ptr addrspace(3)'
# C) 名字对, 但返回类型写成 <2 x half> 聚合 -> IR 校验不过
probe 'llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16'        '{<2 x half>, <2 x half>, <2 x half>, <2 x half>}' 'ptr addrspace(3)'

echo
echo "===== mma.m16n8k16.row.col.f32.f32 是否被识别 ====="
MARG='<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float'
RETM='{float, float, float, float}'
f=$(mktemp /tmp/mmaXXXX.ll)
printf 'target triple = "nvptx64-nvidia-cuda"\ndeclare %s @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(%s)\ndefine %s @use() {\n  %%r = call %s @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(<2 x half> undef, <2 x half> undef, <2 x half> undef, <2 x half> undef, <2 x half> undef, <2 x half> undef, float 0.0, float 0.0, float 0.0, float 0.0)\n  ret %s %%r\n}\n' \
  "$RETM" "$MARG" "$RETM" "$RETM" "$RETM" > "$f"
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 -o - "$f" 2>&1 \
  | grep -E '^[[:space:]]+(ldmatrix|mma)\.sync\.aligned' | head -2
rm -f "$f"

echo
echo "===== 源码树里那份 mma 测试直接在 llc 上跑 ====="
llc "$SRC/llvm/test/CodeGen/NVPTX/mma-no-sink-after-laneid-check.ll" \
    -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx81 -o - 2>&1 \
  | grep -E 'mma.sync.aligned|error' | head -3

echo
echo "===== 对照组: 我们自己那份 intrinsic demo 的 PTX 头 ====="
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 \
    -o $OUT/29-nvvm-intrinsic-demo.ptx code/nvvm_intrinsic_demo.ll
grep -n 'extern .func\|ldmatrix.sync\|mma.sync' $OUT/29-nvvm-intrinsic-demo.ptx | head -8
