# 从一行 CUDA 到一条 HMMA：LLVM 编译 Tensor Core kernel 的全链路课

这套课只做一件事：**把 `.cu` 里那几行手写 Tensor Core 代码，一路追到 SASS 指令，
中间每一站都停下来，用讲课的方式讲清楚"为什么是这个样子"。**

不是"LLVM 架构介绍"，也不是"CUDA 入门"，而是一条具体的、可复现的流水线：

```
tc_mma.cu ──clang──► .ll ──opt──► 优化后 .ll ──llc──► PTX ──ptxas──► cubin ──► SASS
           (前端)              (中端)        (NVPTX 后端)      (汇编器)
```

## 这套课有两套文本，各有用处

| 目录 | 是什么 | 什么时候看 |
| --- | --- | --- |
| `docs/` | **讲义（逐字稿）**。按讲拆开，一讲只解决一两个问题，讲原理、讲推导、讲踩坑 | 第一次学、想搞懂"为什么" |
| `notes/` | **浓缩笔记**。原来的 10 章，一页一张表，信息密度极高 | 复习、查某个具体数字/命令 |

建议的用法是：**先看 `docs/` 的一讲，跑一遍那一讲里的命令，再翻 `notes/` 对应章节当速查表。**

## 讲义目录（逐字稿）

| 部分 | 主题 | 讲数 | 状态 |
| --- | --- | --- | --- |
| [第 0 部分 · 导览](docs/00-intro/index.md) | 这套课怎么上、环境怎么对 | 2 | ✅ |
| [第 1 部分 · 案例与 fragment](docs/01-case/index.md) | 靶子立起来：kernel、fragment 布局、编译命令 | 5 | ✅ |
| [第 2 部分 · Clang 前端](docs/02-frontend/index.md) | CUDA 的东西怎么变成 LLVM IR | 8 | ✅ |
| [第 3 部分 · LLVM IR](docs/03-ir/index.md) | 逐行读懂 `-O2` 之后的 IR | 5 | ✅ |
| [第 4 部分 · 中端优化](docs/04-midend/index.md) | SROA / InstCombine / NVVMReflect / unroll / convergent | 6 | ✅ |
| [第 5 部分 · LLVM 后端设计](docs/05-backend-design/index.md) | **后端到底是怎么搭起来的**（通用设计，逐层拆） | 12 | ✅ |
| [第 6 部分 · NVPTX 后端](docs/06-nvptx/index.md) | 拿 NVPTX 当活标本，看上面的设计怎么落地 | 5 | ✅ |
| [第 7 部分 · PTX](docs/07-ptx/index.md) | `mma` / `ldmatrix` / `cp.async` 逐条讲 | 5 | ✅ |
| [第 8 部分 · SASS](docs/08-sass/index.md) | ptxas 干了什么、`HMMA.16816.F32` 怎么读 | 6 | ✅ |
| [第 9 部分 · 反推与实战](docs/09-reverse/index.md) | 给一条新指令做端到端接入 | 4 | ✅ |

## 快速开始

```bash
# 1) 编译 + 在本机 GPU 上验证数值正确性（应该看到两个 PASS）
bash tools/build-and-run.sh

# 2) 一次性生成全部中间产物到 dumps/
bash tools/dump-all.sh
```

`dumps/` 里的每一个文件，讲义里都会讲到；每讲讲到的命令都能直接粘贴执行。

## 这台机器上的环境基线

讲义里所有输出都是在这台机器上真实跑出来的，版本对不上，结果可能就不一样：

| 项 | 实测值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070，`compute_cap = 8.9`（Ada Lovelace, sm_89） |
| 驱动 | NVIDIA-SMI 610.43.02（KMD 610.47，CUDA UMD 13.3），WSL2 |
| CUDA Toolkit | 12.8（`/usr/local/cuda-12.8`，V12.8.93） |
| Clang / LLVM | 24.0.0git（`/usr/local/bin`，源码在 `/root/llvm-project`，Release 构建） |
| 操作系统 | WSL2 Ubuntu 24.04（kernel 6.18.33.2-microsoft-standard-WSL2） |
| 编译目标 | `sm_89`，PTX ISA 8.7（IR 里是 `+ptx87`） |

## 仓库里还有什么

```
llvm-cuda-course/
├── docs/           讲义（逐字稿），本课主线
├── notes/          10 章浓缩笔记，速查用
├── code/           案例源码与几个对照实验的源码
├── tools/          一键脚本：编译、验证、dump、探测 intrinsic
└── dumps/          全部中间产物（脚本生成）
```

`code/` 里的文件：

| 文件 | 用途 |
| --- | --- |
| `tc_mma.cu` | 核心案例，两个 kernel（手装 fragment / ldmatrix），可编译可跑 |
| `unroll_demo.cu` | K 是编译期常量的对照实验 |
| `unroll_pragma_demo.cu` | 加了 `#pragma unroll 4` 的对照 |
| `noconvergent_demo.cu` | 摘掉 `convergent` 的三条路对比 |
| `reflect_demo.ll` | NVVMReflect 演示 IR |
| `nvvm_intrinsic_demo.ll` | 纯 intrinsic（不用内联汇编）生成 ldmatrix+mma 的 demo |
