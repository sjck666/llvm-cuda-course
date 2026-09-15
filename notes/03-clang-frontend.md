# 第 3 章 · Clang 前端 lowering：.cu → LLVM IR

## 本章要回答的四个问题

上一章我们把靶子立起来了，这一章我们只问一个问题：**我写的那几行 CUDA 特有的东西，在 LLVM IR 里到底变成了什么？**

具体是四样：

1. `__global__` 这个修饰符，凭什么让一个函数变成 kernel？
2. `threadIdx.x` / `blockIdx.y` 是怎么读到硬件的？
3. `__shared__` 数组住在哪个地址空间？
4. `asm volatile("mma.sync...")` 变成 IR 之后，为什么带了 `convergent`？

顺便把 `__syncthreads()`、`__half`、`<<<grid, block>>>` 也一起收掉。

## 3.1 前端全景：一次编译，两套前端

先看 clang 自己招供的命令行。我们加上 `-v`：

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 \
        --cuda-device-only --cuda-gpu-arch=sm_89 -O2 -S -o /dev/null -v tc_mma.cu
```

（完整输出在 `dumps/11-clang-verbose.log`，下面是关键那一行，为了可读性我折了行。）

```
"/usr/local/bin/clang-24" -cc1 -triple nvptx64-nvidia-cuda
   -aux-triple x86_64-unknown-linux-gnu
   -main-file-name tc_mma.cu -fcuda-is-device
   -mlink-builtin-bitcode /usr/local/cuda-12.8/nvvm/libdevice/libdevice.10.bc
   -target-sdk-version=12.8 -target-cpu sm_89 -target-feature +ptx87
   -resource-dir /usr/local/lib/clang/24
   -internal-isystem /usr/local/lib/clang/24/include/cuda_wrappers
   -include __clang_cuda_runtime_wrapper.h
   ...
   -O2 -x cuda tc_mma.cu
```

这一行信息量很大，逐个说：

| 片段 | 含义 |
| --- | --- |
| `-triple nvptx64-nvidia-cuda` | **设备端**的目标三元组，IR 里的 `target triple` 就是它 |
| `-aux-triple x86_64-pc-linux-gnu` | host 侧的目标，CUDA 编译要同时知道两边 |
| `-fcuda-is-device` | 这次 cc1 进程是在编译**设备端**代码 |
| `-mlink-builtin-bitcode .../libdevice.10.bc` | **把 CUDA 的 libdevice 当成 bitcode "链接"进来**，`__nv_expf` 这类数学函数就住在这里面 |
| `-target-cpu sm_89` `-target-feature +ptx87` | 架构和 PTX 版本，对应第 1 章说的 PTX ISA 8.7 |
| `-include __clang_cuda_runtime_wrapper.h` | clang 自己的 CUDA 头文件包装层，`cuda_runtime.h` 被它包了一层，把 `__device__` 等属性和 builtin 变量挂上去 |
| `-internal-isystem .../cuda_wrappers` | 上面那个包装头的搜索路径 |

注意：**这里没有出现 `nvcc`**。设备端编译完全由 clang 自己完成，`libdevice.10.bc` 是 NVIDIA 提供的（预编译的 LLVM bitcode），clang 把它 link 进 IR，之后的优化和 codegen 都是 LLVM 自己做的。

### host 和 device 是两次独立的前端运行

`<<<grid, block>>>` 那行代码在 host 侧；`__global__` 函数体在 device 侧。clang 的做法是把同一份 `.cu` 解析两遍（两套 `-cc1` 调用，triple 不同）。所以：

- `--cuda-device-only` 只跑 device 那一遍，产出我们要研究的 `.ll` / `.ptx`；
- `--cuda-host-only` 只跑 host 那一遍，产出 `main()` 的 IR（本章 3.8 节用得上）；
- 都不加，就是两遍都跑再打包成 fatbinary。

下面的小节全部基于设备端 IR：`dumps/01-device-O0.ll`（未优化）和 `dumps/02-device-O2.ll`（优化后）。

## 3.2 `__global__` 变成了什么

这是 `mma_tc_manual` 在 IR 里的样子（`dumps/01-device-O0.ll`，为可读性折行）：

```llvm
; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local ptx_kernel void @mma_tc_manual(ptr noalias noundef %0,
                                               ptr noalias noundef %1,
                                               ptr noalias noundef %2,
                                               i32 noundef %3, i32 noundef %4,
                                               i32 noundef %5) #0 {
```

三件事值得停下来看：

**第一，名字没被 mangling。** 因为源码里写了 `extern "C"`。如果不写，你会看到 `_Z14mma_tc_manualPK6__halfS1_Pfiii` 这种东西——`.ll` 里难看，`cuobjdump` 里更难看。写 kernel 时建议一直带 `extern "C"`（除非你在做 C++ 模板 kernel）。

**第二，参数变成了 6 个标量。** `(const __half*, const __half*, float*, int, int, int)` 就是这么直白。

**第三，kernel 的身份写在 calling convention 上：`ptx_kernel`。** 这是 NVPTX 目标自己注册的 CC 编号，`~CUDAKernel` 那一档。老版本的 LLVM 用 `!nvvm.annotations` 里的 `!"kernel", i32 1` 来标 kernel，**现在没有了**——你在 `dumps/01-device-O0.ll` 里搜 `nvvm.annotations` 会一无所获，取而代之的就是签名上这个 `ptx_kernel`：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(...)   ; <- 这就是 kernel
define dso_local ptx_kernel void @mma_tc_ldmatrix(...)
```

它最终落到 cubin 的 `.nv.info` / 符号表，CUDA 运行时/驱动靠这个区分哪些 device 函数可以被 `<<<>>>` 启动。**你手工写 IR 做实验时如果忘了在签名上写 `ptx_kernel`，就算函数体完全正确，kernel 也起不来**——这是一个非常常见的坑。

顺带说，正因为用的是 CC 而不是元数据，`-O2` 之后它也不会像元数据那样被优化掉：`dumps/02-device-O2.ll` 里两个函数依然带着 `ptx_kernel`。

再往下翻还能看到版本信息：

```llvm
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 8]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = !{i32 2, i32 0}             ; !nvvmir.version
```

- `SDK Version` 里写的是 `12.8`，它跟着 `-target-sdk-version=12.8` 走，而这个值就是 clang 在 `--cuda-path=/usr/local/cuda-12.8` 里探测到的 toolkit 版本（`dumps/11-clang-verbose.log` 里那行 `Found CUDA installation: /usr/local/cuda-12.8, version 12.8`）。所以这个数字和你装的 CUDA 是一致的，**不需要 `--no-cuda-version-check` 之类的开关**。
- `nvvm-reflect-ftz` 是 FTZ 开关，第 5 章讲 NVVMReflect 时会用到它。

## 3.3 `threadIdx.x` / `blockIdx.y`：直接落到 sreg 读取 intrinsic

源码里的 `threadIdx.x`，在 IR 里长这样：

```llvm
%60 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%61 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
%63 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
```

对应关系是纯约定的：

| CUDA 里写的 | IR intrinsic | PTX 里最终变成 |
| --- | --- | --- |
| `threadIdx.x` | `llvm.nvvm.read.ptx.sreg.tid.x()` | `%tid.x` |
| `blockIdx.x` | `llvm.nvvm.read.ptx.sreg.ctaid.x()` | `%ctaid.x` |
| `blockIdx.y` | `llvm.nvvm.read.ptx.sreg.ctaid.y()` | `%ctaid.y` |
| `blockDim.x` | `llvm.nvvm.read.ptx.sreg.ntid.x()` | `%ntid.x` |
| `gridDim.x` | `llvm.nvvm.read.ptx.sreg.nctaid.x()` | `%nctaid.x` |

每个 intrinsic 都是 `readnone` 的纯值读取（`attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }`），所以后面的优化可以把它们当常数一样搬来搬去。

你可能会问：IR 里不是还有两个全局变量吗？那是给**不支持 sreg 的老路径**或者某些 `__device__` 变量访问留的：

```llvm
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@blockIdx  = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
```

`extern_weak` + `addrspace(1)`（全局内存空间）+ 空结构体，是一套"占位符号"，实际代码走的是上面的 intrinsic。**别被它误导**——你在 `.ll` 里搜 `@threadIdx` 大概率只会搜到声明本身。

## 3.4 地址空间：CUDA 的三种内存，各有各的编号

这是 CUDA 前端做过的最有实质影响的一件事。LLVM IR 里指针类型带地址空间编号，NVPTX 用的编号是：

| 地址空间 | 名字 | 什么时候出现 |
| --- | --- | --- |
| `0` | generic | **函数参数**、`malloc` 出来的指针、`new` 出来的对象 |
| `1` | global | `__device__` / `__constant__` 全局变量、`cudaMalloc` 传进来的指针（隐式） |
| `3` | shared | `__shared__` 变量、动态共享内存 |
| `4` | constant | `__constant__` |
| `5` | local | 寄存器溢出/局部数组 |

### `__shared__` 数组：从函数内的静态变量升成模块级 global

我们 v2 里写的是：

```cpp
__shared__ __align__(16) __half As[16][16];
__shared__ __align__(16) __half Bs[16][8];
```

它们变成了模块级的两个 global：

```llvm
@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
@_ZZ15mma_tc_ldmatrixE2Bs = internal addrspace(3) global [16 x [8 x %struct.__half]] undef, align 16
```

三处细节：

1. **名字里的 `_ZZ15mma_tc_ldmatrixE2As`** 是 Itanium ABI 的"函数内静态变量"写法：`_ZZ` + 函数名长度(`15`) + 函数名 + `E` + 变量名。编译器把它当"每个函数一份的静态对象"来命名，尽管它最终是模块级的。
2. **`undef` 初始化**——共享内存不需要初始化，`undef` 是最省的表示。
3. **`internal` + `addrspace(3)`**：模块私有、住在共享内存空间。`align 16` 是我们自己写的 `__align__(16)` 要求的，也是 `cp.async` 16 字节版本能用的前提。

访问它的时候，前端会先做一次 `addrspacecast`（下文 3.4.3 有实测 IR）。

### 函数参数是 generic 指针

再看 `define ... @mma_tc_manual(ptr noalias noundef %0, ...)`——注意参数是 **`ptr`（地址空间 0，generic）**，不是 `ptr addrspace(1)`。原因很实际：host 侧 `cudaMalloc` 拿到的指针在设备端是 global 地址，但 CUDA 允许你把 generic 指针传进来再自己 `__cvta_*` 转换。所以前端统一用 generic。

到了 MIR 阶段你会看到后端插了一条 `cvta_to_global_64`（第 6 章）。

### `__cvta_generic_to_shared` 的真身

我们 v2 里用 `__cvta_generic_to_shared(smem)` 把共享内存地址转成 32 位偏移，好喂给 `ldmatrix`/`cp.async`。这个 builtin 在 IR 里**不是 intrinsic，而是一个真实的函数**：

```llvm
$__nv_cvta_generic_to_shared_impl = comdat any

define linkonce_odr dso_local i64 @__nv_cvta_generic_to_shared_impl(ptr noundef %0) #5 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = addrspacecast ptr %3 to ptr addrspace(3)     ; generic -> shared
  %5 = ptrtoint ptr addrspace(3) %4 to i64           ; 变成整数偏移
  ret i64 %5
}
```

`linkonce_odr` + `comdat` 意味着"每个用到的编译单元各带一份，链接器去重"。所以你在 `.ll` 里搜 `cvta` 会同时看到这个函数定义**和**它的调用点。

有意思的是它在 PTX 里的落地方式：`-O2` 把这个 helper 内联、又把 `addrspacecast` 和 `ptrtoint` 一起吃掉了，最后剩下的就是**普通的 64 位地址算术 + 一次截断**（`dumps/03-clang-O2.ptx`）：

```
	mov.b64 	%rd9, _ZZ15mma_tc_ldmatrixE2As;   ; 共享内存符号的基址
	add.s64 	%rd10, %rd9, %rd8;                ; + 偏移
	add.s64 	%rd12, %rd10, %rd11;
	cvt.u32.u64 	%r19, %rd12;                 ; 64 位地址 -> 32 位共享内存偏移
	cp.async.cg.shared.global [%r19], [%rd43], 16;
```

中间那一步 `cvt.u32.u64` 就是 `__cvta_generic_to_shared` 的真身——它不需要任何专用指令，因为 sm_89 上共享内存本身就是"32 位偏移"寻址。MIR 里对应的是 `CVT_u32_u64`（`dumps/26-isel.mir`），再往下 `ptxas` 会把它化简成纯粹的地址算术（第 9 章能在 SASS 里看到 `R2`/`R4` 这类共享内存基址寄存器的来龙去脉）。

## 3.5 `__syncthreads()` → `llvm.nvvm.barrier.cta.sync.aligned.all`

这个最简单，也最没有歧义：

```llvm
call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3   ; #3 = { convergent nocallback nounwind }
```

参数那个 `0` 是 barrier 的编号（`bar.sync 0` 的那个 0），也是 `.all` 的含义：整个 CTA 的所有线程都到齐才放行。

两个细节值得注意：

1. **名字叫 `barrier.cta.sync.aligned.all`，不是老写法 `barrier0`。** LLVM 把 NVVM 里的 barrier 变体（`.cta`/`.sys`、`.aligned`、`.all`/`.count`）统一收进了一套命名规则，`__syncthreads()` 落到"CTA 级、全线程"那一格。老资料里的 `llvm.nvvm.barrier0` 已经不存在了——拿它写 IR 只会得到一条无名函数调用（第 10 章讲过这类"名字对不上不报错"的坑）。
2. **它带 `convergent`**。`bar.sync` 是**典型的收敛操作**：如果 warp 内部分线程走不到这个 barrier，就会死锁。所以这个 intrinsic 必须带 `convergent`，LLVM 才不允许把它搬到条件分支里或者复制成两份。

PTX 里最终是 `bar.sync 0;`（第 8 章），SASS 里是 `BAR.SYNC.DEFER_BLOCKING 0x0`（第 9 章）。

## 3.6 内联汇编：`asm volatile` 的三层信息

这一节是全章的核心。我们的 `mma_m16n8k16` 在 O0 的 IR 里长这样（`dumps/01-device-O0.ll:297`）：

```llvm
%225 = call contract { float, float, float, float } asm sideeffect
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
        (i32 %199, i32 %202, i32 %205, i32 %208, i32 %210, i32 %213,
         float %215, float %218, float %221, float %224) #7, !srcloc !6
```

把这条指令拆成五块看：

**1）返回类型是 `{ float, float, float, float }`** —— 四个输出寄存器组成了一个匿名结构体。所以后面紧跟四条 `extractvalue`：

```llvm
%226 = extractvalue { float, float, float, float } %225, 0
%227 = extractvalue { float, float, float, float } %225, 1
%228 = extractvalue { float, float, float, float } %225, 2
```

这是"内联汇编多返回值"的通用表示，跟 CUDA 无关。

**2）`asm sideeffect`** —— `volatile` 换来的。`sideeffect` 告诉 LLVM：这条汇编有副作用，不许因为"结果没人用"就删掉它，也不许随便复制。写 `asm` 而不写 `asm volatile` 时，这里就不会有 `sideeffect`。

**3）约束字符串 `"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"`** —— 和源码里那 14 个占位符一一对应：

| 约束 | 数量 | 含义 | 对应 |
| --- | --- | --- | --- |
| `=f` | 4 | 输出，浮点寄存器 | `d[0..3]`（C/D fragment） |
| `r` | 4 | 输入，32 位整型寄存器 | `a[0..3]`（A fragment，装的是 2 个 f16） |
| `r` | 2 | 输入，32 位整型寄存器 | `b[0..1]`（B fragment） |
| `f` | 4 | 输入，浮点寄存器 | `c[0..3]`（累加器输入） |

`=r` 后面没有数字，`=f` 也没有——说明都用了默认的 `i`（input 也是同一个）之外的普通整型/浮点寄存器类别。如果你想看"约束 → 寄存器类别"的完整映射，第 6 章的 MIR 会给答案：4 个输出是 `regdef:B32`，6 个 A/B 输入和 4 个 C 输入是 `reguse:B32`——在 NVPTX 里，`f` 和 `r` 落在同一个 32 位寄存器类上。

**4）`#7` = `{ convergent nounwind }`** —— 这条是本节的"戏肉"。为什么一条纯计算的 `mma` 会被标成 convergent？

去 clang 源码里找原因（`clang/lib/CodeGen/CGStmt.cpp:3328`）：

```cpp
if (!NoConvergent && getLangOpts().assumeFunctionsAreConvergent())
  // Conservatively, mark all inline asm blocks in CUDA or OpenCL as
  // convergent (meaning, they may call an intrinsically convergent op, such
  // as bar.sync, and so can't have certain optimizations applied around
  // them) unless it's explicitly marked 'noconvergent'.
  Result.addFnAttr(llvm::Attribute::Convergent);
```

而 `assumeFunctionsAreConvergent()` 什么时候为真？追到 `clang/lib/Frontend/CompilerInvocation.cpp:4290`：

```cpp
bool HasConvergentOperations = Opts.isTargetDevice() || Opts.OpenCL ||
                               Opts.HLSL || T.isAMDGPU() || T.isNVPTX();
Opts.ConvergentFunctions =
    Args.hasFlag(OPT_fconvergent_functions, OPT_fno_convergent_functions,
                 HasConvergentOperations);
```

编译 CUDA 设备端时 `isTargetDevice()` 为真 → `ConvergentFunctions` 默认**打开** → 你写的**每一条** `asm` 都被标 convergent。

这不是 clang 在偷懒。它没法知道你的汇编里有没有 `bar.sync`、`shfl`、`mma` 这些需要整个 warp 一起执行的指令，所以保守地全标上。代价是：**被标 convergent 的指令不能被复制、不能被随意移出/移入控制流**，某些优化会因此收手。第 5 章我们会具体看影响。

如果你很确定自己的汇编没有收敛语义（比如只有 `mov.b32` 打包这种），直觉上会想去掉这个标记。我们实测了四条路（`tools/convergent-experiment.sh` 的 G 段）：

| 尝试 | 结果 |
| --- | --- |
| `-fno-convergent-functions` | **有效**：驱动把它传给了设备端 `cc1`，IR 里所有 attribute 组都不再含 `convergent` |
| `__attribute__((noconvergent))` 放在 asm 前 | 直接语法报错：`error: an attribute list cannot appear here` |
| `[[clang::noconvergent]]` 直接放在 asm 前 | 编译能过，但会警告 `'clang::noconvergent' attribute ignored [-Wignored-attributes]`，asm 调用**仍然**引用 `{ convergent nounwind }`，等于没生效 |
| `[[clang::noconvergent]] { asm ...; }` 包一层语句块 | **有效**：属性挂到了 `AttributedStmt` 上，asm 调用点的属性组变成 `{ nounwind }` |

规矩很清楚：**属性必须挂在"语句"上，不能直接挂在 `asm` 语法元素上**。`CGStmt.cpp` 里那段 `case attr::NoConvergent:` 处理的是 `AttributedStmt`（第 809 行），而 `[[clang::noconvergent]]` 直接写在 `asm` 前面时，clang 根本没机会把它包成 `AttributedStmt`——这也是 `__attribute__` 那种写法直接语法报错的原因。

所以能用的两条路是：全局的 `-fno-convergent-functions`，或者给某几条 asm 外面包一层 `[[clang::noconvergent]] { ... }`。第 5 章会拿这两条路做对照实验。

**5）`!srcloc !6`** —— 指回源码位置的元数据，调试和 `-Rpass` 用。

### 其余三条汇编的 IR（供对照）

```llvm
; cp.async 16B
call void asm sideeffect "cp.async.cg.shared.global [$0], [$1], 16;\0A",
     "r,l"(i32 %101, ptr %102) #7, !srcloc !9

; ldmatrix.x4
%166 = call { i32, i32, i32, i32 } asm sideeffect
       "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A",
       "=r,=r,=r,=r,r"(i32 %165) #7, !srcloc !13

; ldmatrix.x2.trans  (dumps/01-device-O0.ll:605)
%185 = call { i32, i32 } asm sideeffect
       "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0A",
       "=r,=r,r"(i32 %184) #7, !srcloc !14
```

两个新约束类别出现了：`l` 表示 64 位整型（`l` = long），用在 `cp.async` 的全局地址上——所以 `ptr %102` 是 64 位的。共享内存那一侧是 `i32`（我们已经用 `cvta` 转成 32 位偏移了），对应 `r`。**这个"两边宽度不一样"的细节，是手写 `cp.async` 最容易写错的点**。

## 3.7 `__half` 和它带来的胶水代码

`__half` 在 IR 里是个结构体：

```llvm
%struct.__half  = type { i16 }
%struct.__half2 = type { %struct.__half, %struct.__half }
```

所以 `__halves2half2(lo, hi)` 这种操作在 IR 里不能白拿，它会变成一条 `mov.b32` 的汇编：

```llvm
; dumps/01-device-O0.ll:742
%7 = call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %5, i16 %6) #8, !srcloc !16
```

（这段来自 `cuda_fp16.hpp` 里的 `__halves2half2`。）注意约束里的 `h`——它表示 16 位整型寄存器，所以两个输入是 `i16`。到了 MIR，你能看得更清楚：

```
INLINEASM &"{  mov.b32 $0, {$1,$2};}\0A", isconvergent attdialect,
          regdef:B32, def %72, reguse:B16, %73, reguse:B16, %74, !17
```

注意这里的 `mov.b32` 汇编**也带 convergent**（`isconvergent attdialect`），虽然是纯数据打包。这正是 3.6 节那条"全标 convergent"规则的后果。

（顺带一提，新版 MIR 把寄存器类别打印成了 `regdef:B32` / `reguse:B16` 这种可读的名字，老版本打的是 `262154 /* regdef:Int32Regs */` 那样的数字 ID。第 6 章会用到这些类别名。）

对我们这门课更重要的是另一件事：**fragment 用的是 `.b32` 寄存器，不是 half 标量**。所以 A 的加载我们用

```cpp
return *reinterpret_cast<const uint32_t *>(p);   // pack_half2
```

一个 32 位 load 直接拿到两个 f16 —— 到了 PTX 就是一条 `ld.global.u32`，省了一条汇编打包。如果顺着 `__half` 结构体走，就得先 load 两个 i16、再 `mov.b32` 打包，多出两条指令。这类"用整数视角看 half"的技巧在 Tensor Core 编程里是标配。

## 3.8 `<<<grid, block>>>` 在 host 侧变成了什么

设备端的账算完了，顺手把 host 侧补上（`dumps/12-host-O0.ll`）。源码里这一行：

```cpp
mma_tc_manual<<<grid, block>>>(dA, dB, dD, M, N, K);
```

被拆成了三部分。**在调用点**：

```llvm
%151 = call i32 @__cudaPushCallConfiguration(i64 %144,  /* gridDim 打包 */
                                             i32 %146,
                                             i64 %148,  /* blockDim 打包 */
                                             i32 %150,
                                             i64 noundef 0,      /* sharedMem */
                                             ptr noundef null)   /* stream */
```

**在自动生成的 stub 里**（clang 为每个 kernel 造一个 `__device_stub__<名字>`）：

```llvm
%26 = call i32 @__cudaPopCallConfiguration(ptr %13, ptr %14, ptr %15, ptr %16)
...
%37 = call noundef i32 @cudaLaunchKernel(ptr noundef @__device_stub__mma_tc_manual,
                                         i64 %30, i32 %32,   ; gridDim
                                         i64 %34, i32 %36,   ; blockDim
                                         ptr noundef %19,    ; kernelParams
                                         i64 noundef %27,    ; sharedMemBytes
                                         ptr noundef %28)    ; stream
```

`declare i32 @cudaLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)` 是 CUDA runtime 的 API。也就是说：

- `<<<>>>` 语法 = `__cudaPushCallConfiguration` + 调 stub；
- stub 负责把参数一个个拷进 `kernelParams` 数组；
- 最后落到 `cudaLaunchKernel`。

**这就是"kernel 函数名必须保持可见"的原因**：host 代码里要拿 `@__device_stub__mma_tc_manual` 的地址去启动。你要是把 kernel 写成 `static`，或者用了奇怪的链接属性，这一步就会出问题。

## 3.9 本章小结与衔接

把本章的映射关系做成一张速查表：

| CUDA 源码 | LLVM IR | 后续章节 |
| --- | --- | --- |
| `__global__ void k(...)` | `define dso_local ptx_kernel void @k(...)` | 第 6 章（kernel 特殊处理）、第 9 章（`.text.<name>`） |
| `threadIdx.x` | `llvm.nvvm.read.ptx.sreg.tid.x()` | 第 6、8 章（`%tid.x`） |
| `__shared__ T a[N]` | `@a = internal addrspace(3) global [N x T] undef` | 第 6（ISel）、8（`.shared` 声明）、9（smem 基址） |
| 函数参数指针 | `ptr`（addrspace 0, generic） | 第 6 章（`cvta_to_global`） |
| `__syncthreads()` | `llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)` | 第 8（`bar.sync 0`）、9（`BAR.SYNC`） |
| `asm volatile("mma...")` | `call {...} asm sideeffect "...", "=f,...,r"` + `convergent` | 第 4（IR 形态）、5（优化约束）、6（INLINEASM）、7/8（原样进 PTX） |
| `__halves2half2` | `call i32 asm "{ mov.b32 $0, {$1,$2}; }"` | 第 9 章（被 ptxas 消掉） |
| `__cvta_generic_to_shared` | `linkonce_odr` 函数：`addrspacecast` + `ptrtoint` | 第 6 章（`cvta.to.shared`） |
| `<<<g,b>>>`（host） | `__cudaPushCallConfiguration` + stub + `cudaLaunchKernel` | 本课程不展开 |

一句话总结本章：**CUDA 前端做的事，本质上就是"把 CUDA 的语义映射成 LLVM 的基础构件"**——kernel 用元数据标、硬件寄存器用 intrinsic 读、共享内存用地址空间区分、同步和内联汇编用 `convergent` 保护。没有任何"魔法指令"进 IR，`mma.sync` 进来的时候就是一条普通的内联汇编调用。

下一章我们不再看"变成什么"，而是看"**这条 IR 具体长什么样、为什么长这样**"：把 `dumps/02-device-O2.ll` 从头到尾读一遍，讲清楚 `bitcast`、`getelementptr`、`extractvalue`、`!srcloc` 这些语法糖，以及 fragment 的 `uint32_t a[4]` 为什么在 O2 之后就从 IR 里"消失"了。
