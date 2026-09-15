#!/usr/bin/env bash
# 把 SASS 洗干净(去掉编码注释), 统计关键指令分布, 并打印带行号的控制流骨架
set -uo pipefail

CUDA=/usr/local/cuda-12.8
export PATH=$CUDA/bin:$PATH

cd "$(dirname "$0")/.."
mkdir -p dumps

CUBIN=${1:-dumps/05-clang.cubin}

# 1) 洗掉编码注释, 得到一行一条指令的 SASS
#    cuobjdump -sass 每条指令后面跟一个 /* 0x.... */ 编码, 再跟一行续行注释
cuobjdump -sass "$CUBIN" \
  | sed -e '/^[[:space:]]*\/\* 0x/d' \
        -e 's@[[:space:]]*/\* 0x[0-9a-f]* \*/[[:space:]]*$@@' \
  > dumps/13-sass-clean.txt

echo "===== 每个函数的指令分布 ====="
awk '
  /Function :/            { fn=$3 }
  /HMMA/                  { hmma[fn]++ }
  /LDSM/                  { ldsm[fn]++ }
  /LDGSTS/                { ldgsts[fn]++ }
  /BAR\.SYNC/             { bar[fn]++ }
  /LDS[^M]/               { lds[fn]++ }
  /STG/                   { stg[fn]++ }
  /LDG[^S]/               { ldg[fn]++ }
  END {
    for (f in hmma) {
      printf "%-16s HMMA=%-3d LDSM=%-3d LDGSTS=%-3d BAR=%-2d LDS=%-3d LDG=%-2d STG=%d\n",
             f, hmma[f], ldsm[f], ldgsts[f], bar[f], lds[f], ldg[f], stg[f]
    }
  }' dumps/13-sass-clean.txt

echo
echo "===== mma_tc_manual 的控制流骨架 (只留分支/谓词/MMA/访存/EXIT) ====="
awk '
  /Function :/ { fn=$3 }
  fn=="mma_tc_manual" && /HMMA|BRA|BSSY|BSYNC|EXIT|ISETP|LDS|LDG|STG|IMAD\.WIDE|DEPBAR/ { print }
' dumps/13-sass-clean.txt | head -60
