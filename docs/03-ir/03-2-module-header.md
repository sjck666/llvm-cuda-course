# 03-2 · Module 头部：datalayout 与全局变量

这一讲读一个文件最前面的那几行。它们看起来像"样板文件"，
但其实每一段都在告诉后端一些**生死攸关**的事。

## 一、我们先看这几行

`dumps/02-device-O2.ll` 的开头：

```llvm
; ModuleID = 'dumps/02-device-O2.ll'
source_filename = "tc_mma.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__half = type { i16 }
%struct.__half2 = type { %struct.__half, %struct.__half }

@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@blockIdx  = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t,  align 1
@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
@_ZZ15mma_tc_ldmatrixE2Bs = internal addrspace(3) global [16 x [8  x %struct.__half]] undef, align 16
```

前两行是注释和来源，跳过。有意思的是后面三块：**datalayout、triple、全局对象**。

## 二、`target triple`：这台"虚拟机器"叫什么

```llvm
target triple = "nvptx64-nvidia-cuda"
```

它来自第 02-1 讲那条 `-cc1` 命令行的 `-triple nvptx64-nvidia-cuda`。

一个三元组的三个部分：

```
nvptx64 - nvidia - cuda
   │         │       └── 操作系统/环境：cuda
   │         └────────── 厂商：nvidia
   └──────────────────── 架构：nvptx64（PTX 虚拟 ISA，64 位）
```

**注意这个架构是"虚拟"的**：`nvptx64` 不是一块真实芯片，它是 PTX 这个中间表示的
抽象机器。LLVM 为它生成 PTX 文本，真正的物理机器（sm_89）由 ptxas 和硬件负责。

## 三、`target datalayout`：一长串"数据布局"

```llvm
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
```

这一段是**用 `-` 分隔的一串约束**，告诉 LLVM"在这台机器上，类型怎么摆"。
逐段读：

| 段 | 含义 |
| --- | --- |
| `e` | little-endian（小端） |
| `p6:32:32` | 地址空间 6 的指针是 **32 位**，按 32 位对齐 |
| `i64:64` | `i64` 按 64 位（8 字节）对齐 |
| `i128:128` | `i128` 按 16 字节对齐 |
| `i256:256` | `i256` 按 32 字节对齐 |
| `v16:16` | 16 位向量（比如 `<2 x i8>`）按 16 位对齐 |
| `v32:32` | 32 位向量（比如 `<4 x i8>`、`<2 x i16>`）按 32 位对齐 |
| `n16:32:64` | "native" 整数宽度是 16 / 32 / 64 位 |

### 为什么 `p6:32:32` 很有意思

地址空间 6 是什么？看第 02-3 讲那张表：**`ADDRESS_SPACE_TENSOR`**，
sm_90 之后的 tensor memory。

我们这台机器是 sm_89，压根用不到 tensor memory——**但 datalayout 里已经有它的条目了**。
这说明 LLVM 的 NVPTX 后端是"一套后端支持所有 SM 版本"的，
datalayout 里会把所有可能的地址空间都写上；具体你这个 kernel 用不用得到，
由 target feature 决定（第 5 部分讲 SubtargetFeature 时展开）。

### `v32:32` 对我们很重要

回想第 01-2 讲的：A fragment 用"32 位整数视角"读两个 f16。
为什么这一手合法？因为 `<2 x i16>` 这类 32 位向量在这台机器上是
**32 位对齐**的——它们本来就住在 32 位寄存器里。`v32:32` 说的就是这件事。

## 四、类型定义：`%struct.__half`

```llvm
%struct.__half  = type { i16 }
%struct.__half2 = type { %struct.__half, %struct.__half }
```

这就是第 02-7 讲的结论：**`__half` 是个 16 位结构体**。

注意一个细节：`%struct.__half2` 不是 `{ i32 }`，而是**两个 `__half` 组成的结构体**。
为什么？因为它的语义是"两个 half"，而不是"一个 32 位整数"。
真正需要把它当 32 位用的时候，我们会走 `reinterpret_cast` 或者内联汇编
`mov.b32`——这就是第 02-7 讲那条胶水汇编的由来。

## 五、全局对象：四行代码，两类东西

```llvm
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@blockIdx  = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t,  align 1
@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
@_ZZ15mma_tc_ldmatrixE2Bs = internal addrspace(3) global [16 x [8  x %struct.__half]] undef, align 16
```

（第 02-3、02-4 讲详细讲过它们，这里只做对照。）

| 行 | 是什么 | 谁在用它 |
| --- | --- | --- |
| `@threadIdx` / `@blockIdx` | **占位符**：`extern_weak` 弱符号，没有任何指令引用 | 没人用，可以无视 |
| `@_ZZ...As` / `@_ZZ...Bs` | **真的共享内存数组**：`internal addrspace(3)`，`undef` 初值，`align 16` | `cp.async` / `ldmatrix` 的目标地址 |

最后两行还有个细节：类型是 `[16 x [16 x %struct.__half]]`，也就是**二维数组**。
而 `%struct.__half` 是 `{ i16 }`（2 字节），所以两个数组分别是
`16×16×2 = 512` 和 `16×8×2 = 256` 字节——和 PTX 里那两行声明对得上：

```
.shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2As[512];
.shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2Bs[256];
```

你可以自己算一遍验证。**这种"IR 里的类型 → 字节数 → PTX 声明"的三级对照，
是检查自己有没有读懂类型系统的最好练习。**

## 六、一个容易被忽略的点：全局变量的顺序

你可能会奇怪：为什么共享内存数组定义在**函数外面**？

因为 IR 的 Module 层是"全局对象"的住所。函数里**静态存储期**的对象
（比如 `__shared__` 数组）都会被提升到这一层——这正是第 02-3 讲的内容。

所以读 IR 里的全局变量时，问自己一句：

> **这是用户写的全局变量，还是某个函数里的静态对象被"提"上来的？**

如果是后者，名字里通常带着 Itanium ABI 的痕迹（`_ZZ<函数名长度><函数名>E...`），
一看就知道。

## 七、小结

1. `target triple = "nvptx64-nvidia-cuda"` 来自 `-triple`，它描述的是一台
   **虚拟 ISA**（PTX）的机器，不是物理 GPU。
2. `target datalayout` 是一串类型布局约束；其中 `v32:32` 解释了"为什么能用
   32 位视角读两个 half"，而 `p6:32:32` 说明后端连 sm_90 的地址空间都预留好了。
3. 开头的全局对象分两类：**占位符**（`@threadIdx`，没人用）和**真家伙**
   （`@_ZZ...As`，共享内存数组）。类型里的字节数和 PTX 的 `.shared` 声明能对上，
   这是检查自己读懂没有的好办法。

## 八、动手题

1. 算一遍字节数，验证 `[16 x [16 x %struct.__half]]` 为什么等于 512 字节，
   `[16 x [8 x %struct.__half]]` 为什么等于 256 字节。
2. 在 `dumps/01-device-O0.ll` 和 `dumps/02-device-O2.ll` 里各搜一次 `datalayout`，
   看看它们是否相同。**为什么同一个编译单元的两份 IR，datalayout 必须一致？**

下一讲我们看函数签名和属性组——那些 `noalias`、`readonly`、`convergent`
从哪来、有什么用。
