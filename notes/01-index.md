# 第 1 章 · 总目录与依赖图

## 先说我们这门课要干什么

我们这门课只干一件事：把一个用 inline PTX 手写的 Tensor Core 矩阵乘 kernel，从 `.cu` 源码一路追到 SASS 指令，中间的每一站都停下来看清楚。

不是"介绍 LLVM 架构"，也不是"CUDA 编程入门"。我们盯的是一条具体的、可复现的流水线：

```
tc_mma.cu  ──clang──►  .ll  ──opt──►  优化后 .ll  ──llc──►  PTX  ──ptxas──►  cubin ──►  SASS
             (前端)                (中端)         (NVPTX 后端)      (汇编器)
```

每一站我都会给你**能直接粘贴执行的命令**，以及**这台机器上真实跑出来的输出**。凡是"我觉得应该是这样"的地方，我都会标出来，并告诉你怎么自己验证。

## 环境（这台机器上实测的版本）

后面所有命令的输出都来自下面这套环境，你在别的机器上跑，版本差一点结果可能就不一样，所以先对一下：

| 项 | 实测值 | 说明 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4070, `compute_cap = 8.9` | Ada Lovelace, sm_89 |
| 驱动 | NVIDIA-SMI 610.43.02 (KMD 610.47, CUDA UMD 13.3) | WSL2 里能直接跑 kernel |
| CUDA Toolkit | 12.8 (`/usr/local/cuda-12.8`, V12.8.93) | 提供 `nvcc`/`ptxas`/`cuobjdump`/`nvdisasm` |
| Clang / LLVM | 24.0.0git (`/usr/local/bin`，源码在 `/root/llvm-project`) | 带 `nvptx64` 后端；Release 构建（无断言） |
| 操作系统 | WSL2 Ubuntu 24.04 (kernel 6.18.33.2-microsoft-standard-WSL2) | 教程里的命令都在 Linux shell 里跑 |
| 目标 | `sm_89`，PTX ISA 8.7（IR 里是 `+ptx87,+sm_89`） | 由 `--cuda-gpu-arch=sm_89` 决定 |

两个必须提前知道的现实问题：

1. **这套 LLVM 是 `LLVM_ENABLE_ASSERTIONS=OFF` 的 Release 构建**。像 `-debug-only=isel`
   这类只在断言构建里注册的选项**不存在**，直接跑会得到
   `llc: Unknown command line argument '-debug-only=isel'`。
   本教程凡是需要看"后端内部状态"的地方，都换成 Release 构建里也能用的替代品（见下文）。
2. **`-arch=sm_89` 是 nvcc 的写法，不是 clang 的**。clang 要写
   `--cuda-gpu-arch=sm_89`（配合 `-x cuda`）。这点在原始需求里给的命令清单里是混的，第 2 章会给出 clang/nvcc 两套能跑的版本。

## 章节地图与依赖关系

```
                 ┌──────────────────────────────┐
                 │ 01 总目录与依赖图（本章）      │
                 └───────────────┬──────────────┘
                                 │ 先有地图
                                 ▼
                 ┌──────────────────────────────┐
                 │ 02 案例代码与编译命令         │  ← 唯一"必须有"的前置
                 │    tc_mma.cu + 全部 dump 命令 │     后面每一章都拿它当输入
                 └───────────────┬──────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │ 前端视角                │ 产物视角                │ 命令/环境视角
        ▼                        ▼                        ▼
┌───────────────┐      ┌──────────────────┐      （第 2 章已覆盖）
│ 03 Clang 前端 │      │ 04 LLVM IR 精讲  │
│ .cu → .ll     │─────►│ 逐段读 .ll       │
└───────────────┘      └────────┬─────────┘
                                │ 有了 IR 才能谈优化
                                ▼
                     ┌──────────────────────┐
                     │ 05 中端 Pass 优化     │
                     │ SROA/InstCombine/     │
                     │ NVVMReflect/Unroll/   │
                     │ convergent 的约束      │
                     └──────────┬───────────┘
                                │ IR 定型，交给后端
                                ▼
                     ┌──────────────────────┐
                     │ 06 NVPTX 后端（上）   │
                     │ ISel: DAG → MachineIR │
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │ 07 NVPTX 后端（下）   │
                     │ 寄存器分配 → PTX 发射  │
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │ 08 PTX 指令精讲       │
                     │ mma/ldmatrix/cp.async │
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │ 09 ptxas → SASS       │
                     │ HMMA.16816 逐字段拆解  │
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │ 10 反推练习           │
                     │ 新 ISA 指令怎么落地    │
                     └──────────────────────┘
```

几条捷径，按你的目的挑：

- 只想搞懂 **SASS 里 HMMA 的操作数**：02 → 09，中间可以跳。
- 想搞懂 **LLVM IR 长什么样**：02 → 03 → 04 → 05。
- 想自己做 **后端移植 / 加新指令**：02 → 04 → 05 → 06 → 07 → 10。
- 想搞懂 **为什么我的 kernel 没打满 Tensor Core**：02 → 08 → 09，回头再看 05（unroll）和 07（寄存器分配）。

## 每章交付什么（对照表）

| 章 | 主题 | 本章的输入 | 本章要产出的东西 |
| --- | --- | --- | --- |
| 01 | 总目录与依赖图 | 无 | 地图、环境基线、产物—命令对照 |
| 02 | 案例代码与编译命令 | 无 | `tc_mma.cu`（可编译可跑）、全部 dump 命令、GPU 上 PASS 的证据 |
| 03 | Clang 前端 lowering | `.cu` | `.cu → .ll` 的规则：kernel 约定、地址空间、内联汇编、`__syncthreads` |
| 04 | LLVM IR 结构精讲 | `.ll`（O0/O2） | 逐段读 IR：fragment 变量怎么变成向量、asm 调用怎么带 convergent |
| 05 | 中端 Pass 优化 | 未优化 `.ll` | SROA/InstCombine/NVVMReflect/LoopUnroll 在本例中的实际效果与边界 |
| 06 | NVPTX 后端（上） | O2 `.ll` | SelectionDAG → Machine IR 的形态，TargetLowering 在做什么 |
| 07 | NVPTX 后端（下） | MIR | 寄存器分配结果、PTX 发射路径、NVPTX 自己的 Pass |
| 08 | PTX 指令精讲 | `.ptx` | `mma.sync`/`ldmatrix`/`cp.async` 的语义、操作数、约束 |
| 09 | ptxas → SASS | `.ptx` | `ptxas -v` 资源用量、`HMMA.16816.F32` 的寄存器拆解 |
| 10 | 反推练习 | 全部 | 从一条 ISA 指令倒推：IR intrinsic → 前端识别 → 后端 pattern → 测试 |

## 中间产物—文件—命令 对照表

这张表你先扫一眼，第 2 章会逐条展开。所有产物都落在 `llvm-cuda-course/dumps/` 下，由 `tools/dump-all.sh` 一键生成。

| 产物 | 文件 | 生成命令（核心部分） |
| --- | --- | --- |
| 设备端 IR（未优化） | `dumps/01-device-O0.ll` | `clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_89 -O0 -S -emit-llvm` |
| 设备端 IR（优化后） | `dumps/02-device-O2.ll` | 同上，`-O2` |
| PTX（clang 出） | `dumps/03-clang-O2.ptx` | `clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_89 -O2 -S` |
| PTX（nvcc 出，对照） | `dumps/04-nvcc-O3.ptx` | `nvcc -arch=sm_89 -O3 -ptx` |
| 汇编后 cubin | `dumps/05-clang.cubin` | `ptxas -arch=sm_89 -v` |
| SASS（带编码） | `dumps/06-sass-cuobjdump.txt` | `cuobjdump -sass` |
| SASS（nvdisasm） | `dumps/07-sass-nvdisasm.txt` | `nvdisasm -c -hex` |
| 寄存器/共享内存用量 | `dumps/08-res-usage.txt` | `cuobjdump -res-usage` |
| 指令选择后的 MIR | `dumps/09-mir-after-isel.mir` | `llc -stop-after=finalize-isel` |
| 编码前 MIR（前导/后记之后） | `dumps/10-mir-after-regalloc.mir` | `llc -stop-after=nvptx-prolog-epilog` |
| 驱动完整日志 | `dumps/11-clang-verbose.log` | `clang++ ... -v` |
| NVVMReflect 演示 | `code/reflect_demo.ll` | `opt -passes=nvvm-reflect` |

## 关于 `-debug-only=isel` 这件事（提前说清楚）

需求清单里提到用 `-debug-only=isel` 抓 SelectionDAG。这条命令**只在开启了断言的 LLVM 构建里存在**，本机这套是 Release 构建，直接跑会得到：

```
$ llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -debug-only=isel ...
llc: Unknown command line argument '-debug-only=isel'.  Try: 'llc --help'
llc: Did you mean '--debug-pass=isel'?
```

所以本教程里，凡是需要看"后端内部状态"的地方，我都换成这套**在 Release 构建里也能用的替代品**，而且都在真机上跑过：

- `-print-after-all`：打印每个 pass 之后的 IR / MIR；
- `-stop-after=<pass>`：把流水线停在某一站，dump 成 `.mir` 文本；
- `cuobjdump -sass` / `nvdisasm -c -hex`：直接看最终机器码；
- 源码对照：真要确认某个 lowering 细节，去读 `llvm/lib/Target/NVPTX/` 里对应的 `.td` / `.cpp`。

如果你自己构建了 `LLVM_ENABLE_ASSERTIONS=ON` 的 LLVM，`-debug-only=isel` 就能用了，第 6 章会把原始 DAG 的样子和替代方案放在一起讲。

## 复现这套教材的全部命令

```bash
# 0) 工具链（本机已就绪）
#    clang / llc / opt / llvm-config 在 /usr/local/bin
#    CUDA 12.8 在 /usr/local/cuda-12.8
clang --version                            # 24.0.0git

# 1) 编译 + 在本机 GPU 上验证数值正确性
bash llvm-cuda-course/tools/build-and-run.sh

# 2) 一次性 dump 出全部中间产物
bash llvm-cuda-course/tools/dump-all.sh
```

两条脚本的实测输出都在第 2 章里。

## 下一章要做什么

第 2 章先把"靶子"立起来：完整写出 `tc_mma.cu`，讲清楚 A/B/C/D 四个 fragment 每个线程到底持有哪几个元素（这是后面所有章节的地基，fragment 布局错了，SASS 章节你会看不懂 `HMMA` 的操作数），然后给出两条编译路径和全部 dump 命令，最后在 GPU 上跑出 `PASS`。
