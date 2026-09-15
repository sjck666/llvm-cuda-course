# 08-1 · 拿到 SASS 与资源用量

这一讲先把工具链跑起来，并解释那几个数字。

## 一、三条命令

```bash
CUDA=/usr/local/cuda-12.8
export PATH=$CUDA/bin:$PATH

# 1) PTX → cubin，顺便看资源用量
ptxas -arch=sm_89 -v dumps/03-clang-O2.ptx -o dumps/05-clang.cubin

# 2) 从 cubin 里读出 SASS（两种视图）
cuobjdump -sass dumps/05-clang.cubin > dumps/06-sass-cuobjdump.txt
nvdisasm -c -hex dumps/05-clang.cubin > dumps/07-sass-nvdisasm.txt

# 3) 洗干净并统计（我们的脚本）
bash tools/analyze-sass.sh
```

## 二、资源用量：四个关键数字

```
ptxas info    : Compiling entry function 'mma_tc_ldmatrix' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 768 bytes smem, 388 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function 'mma_tc_manual' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers, 388 bytes cmem[0]
```

| 数字 | 谁决定的 | 含义 |
| --- | --- | --- |
| **40 registers** | **ptxas** | 真正的物理寄存器数量，**决定 occupancy** |
| 0 spill | ptxas | 没有寄存器溢出（SASS 里不会有 `LDL`/`STL`） |
| 768 bytes smem | LLVM | 共享内存用量（两个 `__shared__` 数组：512 + 256） |
| 1 barrier | LLVM（`__syncthreads`） | barrier 0 被用到了 |
| 388 bytes cmem[0] | LLVM/驱动 | kernel 参数区（常量内存 0 号 bank） |

### 为什么"寄存器数"是最重要的那个

因为 **occupancy**（每个 SM 上能同时跑多少 warp）直接由它决定：

```
寄存器越多 → 每个 warp 占的寄存器文件越多 → 能同时驻留的 warp 越少 → 延迟越难藏住
```

而**这个数字是 ptxas 报的，不是 LLVM 报的**——因为寄存器分配是 ptxas 做的（第 05-10 讲）。

**所以调优时你要关注的是"ptxas 能看到的东西"**：

```
PTX 里虚拟寄存器的数量（由 LLVM 决定）
.maxnreg 指令（可以在源码/PTX 里指定）
__launch_bounds__（告诉编译器期望的 occupancy）
```

### "0 spill" 为什么值得庆幸

一旦 spill，SASS 里会插一堆 `LDL`（local load）、`STL`（local store），
而且 local memory 是在显存里的——延迟高得离谱。

**对我们这个教学案例来说，"0 spill"还有个额外意义**：
数据流没有被 spill 打断，我们读 `HMMA` 周围的寄存器传递才能读得清楚。

## 三、指令分布：一眼看出两个 kernel 的差别

```bash
bash tools/analyze-sass.sh
```

```
===== 每个函数的指令分布 =====
mma_tc_manual    HMMA=7   LDSM=0   LDGSTS=0   BAR=0  LDS=0   LDG=56 STG=4
mma_tc_ldmatrix  HMMA=7   LDSM=14  LDGSTS=14  BAR=14 LDS=21  LDG=7  STG=4
```

**这张表就是 v1 和 v2 的"体检报告"**：

| 指令 | v1（手动） | v2（ldmatrix） | 说明 |
| --- | --- | --- | --- |
| `HMMA` | 7 | 7 | 两个 kernel 都是 7 个矩阵乘站点（4 主 + 2 中 + 1 收尾） |
| `LDG`（全局加载） | **56** | **7** | v1 全靠 `LDG` 读 A/B；v2 只在装载片段时读 D 之类 |
| `LDSM`（矩阵加载） | 0 | **14** | v2 用 `LDSM` 装 fragment |
| `LDGSTS`（cp.async） | 0 | **14** | v2 用异步拷贝搬 tile |
| `BAR.SYNC` | 0 | **14** | v2 需要同步（每个 k 步两次） |
| `STG`（全局存储） | 4 | 4 | 写回 D |

**一句话总结这张表：**

> **v1 用 56 条全局加载"硬凑" fragment；v2 用 14 条异步拷贝 + 14 条矩阵加载
> 完成了同样的事，而且数据走的是共享内存。**

这就是"手写 fragment"和"用 Tensor Core 正规打法"的差别——
在 SASS 层面是**指令数量和访存路径的差别**。

## 四、为什么两个 kernel 都是 7 个 HMMA

因为它们做的是同一个矩阵乘（`K = 64`，每次前进 16）。
7 这个数字来自 ptxas 的展开策略：

```
主循环 4 个 + 中间循环 2 个 + 收尾 1 个 = 7
```

第 08-6 讲会详细讲这个 4/2/1 分解。

**注意：这不是"代码里有 7 个 mma"，而是"ptxas 把它们展开成了 7 个站点"。**
源码里只有一个 mma（在循环里）。

## 五、两种 SASS 视图的差别

我们生成了两份：

| 文件 | 命令 | 特点 |
| --- | --- | --- |
| `06-sass-cuobjdump.txt` | `cuobjdump -sass` | 干净，一行一条指令 |
| `07-sass-nvdisasm.txt` | `nvdisasm -c -hex` | 带机器码字（两行十六进制）和更多注释 |

**读代码用前一份，需要确认编码时用后一份。**

## 六、小结

1. 三条命令：`ptxas -v`（汇编 + 资源用量）→ `cuobjdump -sass`（SASS）→
   `analyze-sass.sh`（清洗 + 统计）。
2. 资源用量里**"40 registers"是 ptxas 决定的**，它是 occupancy 的关键；
   `0 spill` 说明 SASS 没有被 spill 打断。
3. 指令分布表一眼看出 v1/v2 的差别：**v1 靠 56 条 `LDG`，v2 靠 14 条 `LDGSTS` +
   14 条 `LDSM`**；两个 kernel 都是 7 个 `HMMA` 站点（那是 ptxas 展开出来的）。

## 七、动手题

1. 跑一遍 `bash tools/analyze-sass.sh`，确认你看到的数据和上面一致。
2. 试试用 `-maxrregcount=32` 重新汇编：

   ```bash
   ptxas -arch=sm_89 -v -maxrregcount=32 dumps/03-clang-O2.ptx -o /tmp/small.cubin
   ```

   我们实测的结果很有意思：

   ```
   mma_tc_ldmatrix:  32 registers, 0 bytes spill            ← 压得下
   mma_tc_manual:    32 registers, 8 bytes spill stores,    ← 压不下了！
                                  4 bytes spill loads
   ```

   **同样的限制下，v2 压得下、v1 压不下。** 这就是"寄存器压力"最直观的实测，
   也说明 v2 的写法（共享内存 + `ldmatrix`）在寄存器上更省。

下一讲我们学怎么读 SASS 的格式。
