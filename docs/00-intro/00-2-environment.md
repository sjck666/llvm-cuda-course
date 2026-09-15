# 00-2 · 环境与工具链：先把你手上的版本钉死

上一讲说了这门课是干什么的。这一讲我们要做一件看起来很枯燥、但**后面每一讲都要靠它**的事：
把环境钉死。

为什么这么重要？我举个上一讲提过的例子。你可能记得这么一句话：

> "K 是编译期常量时，k 循环会被展开。"

这句话在 LLVM 18 上是**错的**，在 LLVM 24 上是对的。同一个文件、同一条 `-O2` 命令，
只是换了个编译器版本，结论就翻过来了。如果你不知道自己是哪一版，你就没法判断
"是我读错了"还是"是版本不一样"。

所以这一讲我们先把版本和路径全部打印出来对一遍，再讲几个会影响结论的版本差异。

## 一、四个必须记住的版本

我们这条链路上一共站着四个角色，各管一段：

| 角色 | 负责哪一段 | 本机版本 | 装在哪 |
| --- | --- | --- | --- |
| **Clang / LLVM** | `.cu → IR`、`IR → PTX` | 24.0.0git | `/usr/local/bin` |
| **CUDA Toolkit** | 提供头文件、libdevice、汇编器 | 12.8 (V12.8.93) | `/usr/local/cuda-12.8` |
| **GPU 与驱动** | 也就是最终的硬件 | RTX 4070 / sm_89，驱动 610.47 | WSL2 直通 |
| **操作系统** | 提供 shell 和文件系统 | WSL2 Ubuntu 24.04 | — |

把它们一条命令打出来（这也是你换机器时应该跑的第一条命令）：

```bash
clang --version | head -1
llc --version | head -3
nvcc --version | grep release
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

这台机器上的输出：

```
clang version 24.0.0git (ssh://git@ssh.github.com:443/llvm/llvm-project.git 677a4c33ba942fe7aec6a1be15ba388f1b74d855)
LLVM (http://llvm.org/):
  LLVM version 24.0.0git
  Optimized build.
Cuda compilation tools, release 12.8, V12.8.93
name, compute_cap, driver_version
NVIDIA GeForce RTX 4070, 8.9, 610.47
```

## 二、逐条解释：这四个版本各自会影响什么

### 1）LLVM 24.0.0git：决定 IR、PTX、以及"哪条优化做得出来"

这是这套课里**唯一一个换掉就会让结论翻车的组件**。它的影响面超出你的想象：

**第一，它决定 IR 长什么样。** 比如 kernel 函数的签名。老版本这样写：

```llvm
define dso_local void @mma_tc_manual(...)      ; 老版本：就是个普通函数
!nvvm.annotations = !{...}                     ; kernel 的身份藏在元数据里
```

本机这一版是：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(...)   ; kernel 的身份写在 calling convention 上
```

**第二，它决定 PTX 的起始版本。** 同一个 `--cuda-gpu-arch=sm_89`，老版本的 clang 会给你
`+ptx83`（PTX ISA 8.3），这一版给的是 `+ptx87`（PTX ISA 8.7）。PTX 版本一变，
有些指令的写法就不一样了（比如这一版里有 `.param .u64 .ptr .align 1` 这种更精确的参数修饰）。

**第三，它决定优化器有多"聪明"。** 我们这门课最有意思的一次翻车就在这儿：
同一份 `unroll_demo.cu`（K 是 `constexpr`），LLVM 18 编出来循环还在，LLVM 24 编出来
循环被完全展开成 4 条 `mma`。

本机的这个 LLVM 是从源码构建的，源码就在 `/root/llvm-project`。**这一点很重要**，
因为后面第 5 部分讲"后端设计"时，我们会直接翻它的源码；而且既然编译器就是从这棵树构建的，
**源码里的行号和你手上的编译器是对得上的**——这在读 LLVM 时是个奢侈品，值得利用。

### 2）CUDA 12.8：提供"外部世界"的三样东西

CUDA Toolkit 在这条链路上提供三样东西，缺一不可：

| 提供什么 | 具体路径 | 谁在用 |
| --- | --- | --- |
| 头文件 | `/usr/local/cuda-12.8/include` | clang 前端（`cuda_fp16.h`、`cuda_runtime.h`） |
| libdevice（数学库的 bitcode） | `/usr/local/cuda-12.8/nvvm/libdevice/libdevice.10.bc` | clang 前端，链接进 IR |
| 汇编器与反汇编器 | `ptxas` / `cuobjdump` / `nvdisasm` | 链路后半段（PTX → SASS） |

注意 **libdevice 是一份 LLVM bitcode**。这件事本身就说明：NVIDIA 自己也用 LLVM。
第 2 部分我们会看到 clang 命令行里那行 `-mlink-builtin-bitcode .../libdevice.10.bc`，
它就是把这份 bitcode 链进你的 IR。

顺带说一个会踩到的坑：**clang 对 CUDA 版本是有"认不认识"的问题的**。
老版本的 clang 只认到 CUDA 12.3，遇到 12.8 会警告可能不兼容。本机这版 clang 直接认识 12.8，
所以你在这门课里**不需要** `--no-cuda-version-check` 这类开关——这一点和很多流传的
CUDA + clang 命令清单不一样，别照抄。

### 3）RTX 4070 / sm_89：决定后端生成的指令集合

GPU 这一端我们只关心一个数字：`compute_cap = 8.9`，也就是 `sm_89`。

它为什么重要？因为**不同的 SM 版本支持的指令集不一样**。我们的案例用的是
`mma.sync.aligned.m16n8k16`，这条指令从 sm_80 就有；而 `cp.async` 也是 sm_80 引入的；
再往上的 sm_90 是全新的 `wgmma`/`tcgen05` 体系，编程模型都不一样了。

你在 IR 里会看到这个数字被写进函数属性：

```llvm
"target-cpu"="sm_89" "target-features"="+ptx87"
```

这一串东西是**从命令行一路传下来的**：`--cuda-gpu-arch=sm_89` → `-target-cpu sm_89`
→ 函数属性 → PTX 的 `.target sm_89`。第 2、7 部分会各讲一次。

### 4）WSL2：只影响一件事——`nvidia-smi` 不在 PATH 里

你在 WSL2 里可能发现 `nvidia-smi` 命令找不到。这不是没装驱动，而是 WSL 把 NVIDIA 的用户态
库放在另一个路径下：

```bash
ls /usr/lib/wsl/lib/          # 这里有 libcuda.so、nvidia-smi 等
/usr/lib/wsl/lib/nvidia-smi   # 直接用全路径调用
```

编译和跑 kernel 都不受影响（WSL 通过 `/dev/dxg` 直通 GPU），只有你想看 GPU 状态时
需要注意这个路径。这门课里其它命令都在普通 PATH 上，不用特殊处理。

## 三、这条链路上的六个可执行文件

后面每一讲都会用到下面这几个命令，先认个脸：

```
/usr/local/bin/clang            前端 + 驱动：.cu → IR → PTX，也能直接编出可执行文件
/usr/local/bin/llc              后端：IR → PTX（也可以只跑到中间某一站 dump 出来）
/usr/local/bin/opt              中端：单独跑 pass，做对照实验
/usr/local/cuda-12.8/bin/ptxas  PTX → SASS（cubin），并报寄存器/共享内存用量
/usr/local/cuda-12.8/bin/cuobjdump  从 cubin 里读出 SASS、资源用量
/usr/local/cuda-12.8/bin/nvdisasm   另一种反汇编视图（带机器码字）
```

它们的分工，用一句话概括就是：

> **clang 和 llc 是"从 IR 往下走"，ptxas 及之后是"完全离开 LLVM 的世界"。**

最后这句话不是修辞。`ptxas` 是 NVIDIA 的闭源汇编器，它**不是 LLVM 的一部分**，
它有自己的寄存器分配器、自己的调度器。第 8 部分我们会看到它替 LLVM 干了多少活。

## 四、一个必须提前知道的限制：`-debug-only` 用不了

你上网搜 LLVM 后端调试，十篇文章里有八篇会告诉你用 `-debug-only=isel`、`-mllvm -debug`
看 SelectionDAG。我们这台机器上跑一下：

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -debug-only=isel -o /dev/null dumps/02-device-O2.ll
```

```
llc: Unknown command line argument '-debug-only=isel'.  Try: 'llc --help'
llc: Did you mean '--debug-pass=isel'?
```

为什么？因为这些选项只在**开启了断言的构建**（`LLVM_ENABLE_ASSERTIONS=ON`）里注册。
发行版自带的 LLVM、以及我们这份为了编译速度做的 Release 构建，都没有它们。

所以这门课里凡是需要"看后端内部状态"的地方，我们都换成 Release 构建里也能用的替代品：

| 想看的 | 用这个 |
| --- | --- |
| 每个 pass 之后的 IR/MIR | `llc -print-after-all` |
| 停在某一站 dump | `llc -stop-after=<pass>` |
| 指令选择的结果 | `llc -stop-after=finalize-isel` |
| 最终的机器码 | `ptxas` + `cuobjdump -sass` |

这些够用了——事实上第 5 部分讲后端设计时，我们主要靠这四条命令加源码对照，
就能把 SelectionDAG 的行为推得清清楚楚。

## 五、环境自检：一次性把该看的都看了

把前面几段拼起来，就是一条可以随手跑的自检命令。**换机器、换版本、或者"结果和教材对不上"时，
先跑它。**

```bash
echo "== LLVM =="; clang --version | head -1; llc --version | sed -n 2p
echo "== CUDA =="; nvcc --version | grep release
echo "== GPU  =="; /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv | tail -1
echo "== 工具 =="; command -v clang llc opt ptxas cuobjdump nvdisasm | tr '\n' ' '; echo
```

本机输出（也就是这门课所有结论成立的前提）：

```
== LLVM ==
clang version 24.0.0git (ssh://git@ssh.github.com:443/llvm/llvm-project.git 677a4c33ba942fe7aec6a1be15ba388f1b74d855)
  LLVM version 24.0.0git
== CUDA ==
Cuda compilation tools, release 12.8, V12.8.93
== GPU  ==
NVIDIA GeForce RTX 4070, 8.9, 610.47
== 工具 ==
/usr/local/bin/clang /usr/local/bin/llc /usr/local/bin/opt /usr/local/cuda-12.8/bin/ptxas /usr/local/cuda-12.8/bin/cuobjdump /usr/local/cuda-12.8/bin/nvdisasm
```

## 六、小结与预告

这一讲你要带走的是三句话：

1. 这门课的所有结论**绑定在 LLVM 24.0.0git + CUDA 12.8 + sm_89** 上；
   版本不同，尤其 LLVM 大版本不同，结论可能翻车（我们已经见过一次）。
2. LLVM 是从 `/root/llvm-project` 源码构建的，所以**源码行号和你手上的编译器一致**——
   这是后面读后端源码的基础。
3. 没有 `-debug-only`，我们用 `-print-after-all` / `-stop-after` / `ptxas+cuobjdump` 顶替。

## 七、动手题

1. 跑一遍上面那条自检命令，把输出和教材里贴的对一下。如果 LLVM 版本不同，
   把后面每一讲里出现的行号、寄存器号、pass 名字都打一个问号——它们可能都变了。
2. 试着把 `clang --version` 里那一长串东西拆开看：`24.0.0git`、
   括号里的 `ssh://...llvm-project.git 677a4c3...`。括号里那个 `677a4c3`
   是**构建这个 clang 时的 git commit**，你也可以在 `/root/llvm-project` 里
   用 `git log -1` 找到它——这就是"源码和工具链对得上"的证据。

下一讲开始，我们正式入场：先看案例本身，把 fragment 布局钉死。
