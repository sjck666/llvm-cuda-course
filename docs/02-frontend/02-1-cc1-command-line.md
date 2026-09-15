# 02-1 · 一次编译、两套前端

这一讲我们不写代码，只干一件事：**让 clang 把它自己的命令行招供出来，然后逐段读。**

为什么值得花一整讲？因为后面所有"为什么 IR 长这样"的答案，一半都能在这条命令里找到。

## 一、clang 是个"驱动"，不是"编译器"

很多人以为 `clang++ tc_mma.cu` 就是"clang 编译了这个文件"。其实严格说：

**`clang++` 是一个驱动程序（driver）**，它自己不做编译，只负责：

1. 解析你给的参数；
2. 决定要跑几遍编译、每遍带什么参数；
3. 调用真正干活的 `clang -cc1`（前端）、`ptxas`（外部汇编器）、`ld`（链接器）。

在 CUDA 这种"一份源码、两个目标"的场景里，这个"跑几遍"就变得很有戏了。

## 二、让它招供

加上 `-v`（verbose）就能看到它到底调了什么。我们用**只编设备端**的模式，
这样输出里就只有一次 `-cc1`：

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 \
        --cuda-device-only --cuda-gpu-arch=sm_89 -O2 -S -o /dev/null -v \
        code/tc_mma.cu
```

完整输出在 `dumps/11-clang-verbose.log`。前面几行是环境探测：

```
clang version 24.0.0git (ssh://git@ssh.github.com:443/llvm/llvm-project.git 677a4c33...)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/local/bin
Found candidate GCC installation: /usr/lib/gcc/x86_64-linux-gnu/13
Found CUDA installation: /usr/local/cuda-12.8, version 12.8
 (in-process)
```

注意 `Found CUDA installation: ... version 12.8`：clang 自己去探测了 CUDA 的版本。
第 00-2 讲说过，老版本的 clang 探测能力只到 CUDA 12.3，遇到 12.8 会警告；
这一版认识 12.8，所以命令行里**不需要 `--no-cuda-version-check`**。

## 三、真正的那一行：`-cc1` 命令行

接下来这一行就是全部的秘密。我把它折成多行，逐段标了注释：

```
"/usr/local/bin/clang-24" -cc1
   -triple nvptx64-nvidia-cuda            ← 设备端三元组：这台"虚拟机器"是 nvptx64
   -aux-triple x86_64-unknown-linux-gnu   ← host 端三元组
   -O2 -S
   -main-file-name tc_mma.cu
   -fcuda-is-device                       ← 这一遍是在编"设备端"
   -mlink-builtin-bitcode /usr/local/cuda-12.8/nvvm/libdevice/libdevice.10.bc
   -target-sdk-version=12.8
   -target-cpu sm_89                      ← 架构
   -target-feature +ptx87                 ← PTX ISA 版本
   -resource-dir /usr/local/lib/clang/24
   -internal-isystem /usr/local/lib/clang/24/include/cuda_wrappers
   -include __clang_cuda_runtime_wrapper.h
   -D__CUDA_ARCH_LIST__=890
   -internal-isystem /usr/local/cuda-12.8/include
   ...
   -x cuda tc_mma.cu
```

下面逐段说清楚。这一段读懂，你对"CUDA + clang"的理解就超过大多数人了。

### 1）`-triple nvptx64-nvidia-cuda`：这就是 IR 里的 `target triple`

LLVM 的一切都围绕"三元组"（target triple）展开。设备端的三元组是
`nvptx64-nvidia-cuda`——注意它描述的**不是物理 CPU，而是一个虚拟 ISA**（PTX）。

你在 `dumps/02-device-O2.ll` 第 4 行会看到同一串东西：

```llvm
target triple = "nvptx64-nvidia-cuda"
```

而 host 端的三元组是 `x86_64-unknown-linux-gnu`，它出现在 `-aux-triple` 里。
**一次编译里同时存在两个目标，这就是 CUDA 编译的基本形态。**

### 2）`-fcuda-is-device`：这一遍在编哪一半

clang 处理 `.cu` 的方式是：**同一份源码解析两遍**，一遍当 host 编、一遍当 device 编。

```
--cuda-device-only   →  只跑 device 那一遍（我们的主线）
--cuda-host-only     →  只跑 host 那一遍（第 08 讲要用）
两个都不加           →  两遍都跑，再把结果打包成 fatbinary
```

`-fcuda-is-device` 就是"device 那一遍"的标记。

### 3）`-mlink-builtin-bitcode ...libdevice.10.bc`：把数学库"链"进来

这一条最值得说。`libdevice.10.bc` 是 **NVIDIA 提供的一份 LLVM bitcode 文件**——
它里面是 `__nv_expf`、`__nv_sinf` 这类数学函数的实现。

clang 的做法不是"调用一个库"，而是**把这份 bitcode 直接链进你的 IR 模块**，
之后所有的优化和代码生成都在 LLVM 里完成。

**这件事说明什么？说明 NVIDIA 自己也用 LLVM。** PTX 只是他们选择的中间表示，
而这份 libdevice 就是"给 LLVM 用的运行时库"。

（我们这个案例里其实没调用数学函数，所以 libdevice 里真正被用到的东西不多，
但你可以在 IR 结尾看到它的痕迹——比如 `!llvm.ident` 里那条
`clang version 3.8.0 (tags/RELEASE_380/final)`，那是 libdevice 自带的编译器版本字符串。）

### 4）`-target-cpu sm_89` 和 `-target-feature +ptx87`

这两个值就是第 00-2 讲里 `--cuda-gpu-arch=sm_89` 传下来的结果：

```
--cuda-gpu-arch=sm_89  →  -target-cpu sm_89 + -target-feature +ptx87
```

它们最终会出现在 IR 的函数属性里：

```llvm
"target-cpu"="sm_89" "target-features"="+ptx87"
```

而 PTX 文件头的 `.target sm_89` / `.version 8.7` 也是从这儿来的。

**`+ptx87` 是这一版 clang 给的默认 PTX ISA 版本**（PTX ISA 8.7）。
老版本 clang 给的是 `+ptx83`。这个数字变了，PTX 里有些写法也会变，
第 7 部分会看到具体例子（比如 `.param .u64 .ptr .align 1`）。

### 5）`-resource-dir` 和那两个 `cuda_wrappers`

```
-resource-dir /usr/local/lib/clang/24
-internal-isystem /usr/local/lib/clang/24/include/cuda_wrappers
-include __clang_cuda_runtime_wrapper.h
```

`resource-dir` 是 clang 自己的"资源目录"，里面放着它的内建头文件
（`stddef.h`、`stdint.h` 之类）以及**CUDA 的包装头**。

`__clang_cuda_runtime_wrapper.h` 是个关键角色：**CUDA 官方的 `cuda_runtime.h`
在 clang 下不能直接用**，因为里面有一堆假设是给 nvcc 看的。
clang 用这一层包装头把官方的头文件包起来，然后：

- 把 `threadIdx`、`blockIdx` 这些内建变量挂上去；
- 把 `__device__`、`__global__` 这些属性映射到 clang 的属性；
- 把一些 nvcc 特有的东西屏蔽掉。

所以你的 `#include <cuda_runtime.h>` 实际上被换了层皮。

### 6）`-D__CUDA_ARCH_LIST__=890`

这是一版新加的：把"要编哪些架构"以宏的形式告诉源码。
`890` 就是 sm_89 去掉小数点。有些头文件会用这个宏做条件编译。

## 四、两条路径：device 那一遍和 host 那一遍

我们现在知道了 device 那一遍长什么样。那 host 那一遍呢？

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 --cuda-host-only \
        -O0 -S -emit-llvm -o dumps/12-host-O0.ll code/tc_mma.cu -v
```

它产出的是 `dumps/12-host-O0.ll`，里面：

- 三元组是 `x86_64-unknown-linux-gnu`；
- kernel 函数**不存在**（它们是设备代码），取而代之的是三个东西：
  1. `__device_stub__mma_tc_manual`（一个自动生成的 stub 函数）；
  2. 调用 `__cudaPushCallConfiguration` 来保存 `<<<grid, block>>>` 的配置；
  3. 调用 `cudaLaunchKernel` 真正启动。

第 02-8 讲会把 host 这一遍完整拆开。

## 五、小结

1. `clang++` 是**驱动程序**；真正干活的是它调起来的 `clang -cc1`（前端）。
2. CUDA 编译是**同一份源码解析两遍**：一遍 `-triple nvptx64-nvidia-cuda
   -fcuda-is-device`（设备端），一遍 host 三元组（host 端）。
3. `-cc1` 命令行里三样东西最值得记：
   - `-triple` / `-aux-triple`：两个目标；
   - `-mlink-builtin-bitcode .../libdevice.10.bc`：NVIDIA 的数学库是 LLVM bitcode，被"链"进 IR；
   - `-target-cpu sm_89 -target-feature +ptx87`：架构与 PTX 版本，会一路传到 PTX 文件头。

## 六、动手题

1. 跑一遍上面那条 `-v` 命令，在输出里找 `-cc1` 那一行，
   用眼睛数一数它有几个 `-internal-isystem`。
   **为什么同一个目录会出现两次？**（提示：看 clang 后面那句
   `ignoring duplicate directory`。）
2. 把 `--cuda-device-only` 换成 `--cuda-host-only`，再跑一遍 `-v`，
   看看 `-cc1` 那一行里 `-triple` 变成了什么、`-fcuda-is-device` 还在不在。
   这一条命令的差异，就是"两套前端"最直接的证据。

下一讲我们开始看设备端 IR 的第一件事：**`__global__` 函数变成了什么**。
