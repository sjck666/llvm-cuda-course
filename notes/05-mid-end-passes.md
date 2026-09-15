# 第 5 章 · 中端 Pass 优化

## 5.0 本章要回答的问题

第 4 章我们读的是"结果"，这一章我们要读"过程"。四个具体问题：

1. `-O2` 到底调了哪些 pass、按什么顺序？（答案：121 项，头几项就很反直觉）
2. `float c[4]`、`uint32_t a[4]` 这些数组是谁消灭的？（SROA + mem2reg，796 行 → 353 行）
3. 循环为什么没展开？谁该为它负责？
4. 为什么 `mma` 这条 convergent 指令动不了？（不只是 MachineSink 一家的事）

所有实验都在 `tools/midend-demos.sh` 和 `tools/convergent-experiment.sh` 里，可以自己重跑。

## 5.1 先把流水线摊开看

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 \
        --cuda-device-only --cuda-gpu-arch=sm_89 -O2 -S -emit-llvm \
        -o /dev/null -mllvm -print-pipeline-passes tc_mma.cu
```

输出是一个逗号分隔的长字符串，`tr ',' '\n'` 拆开后一共 **121 项**（`dumps/14-device-pipeline.txt`）。开头这几十项值得完整看一眼：

```
memprof-remove-attributes
annotation2metadata
forceattrs
nvvm-reflect                       <-- 第 4 项：NVPTX 目标插进来的
function(nvvm-intr-range)          <-- 第 5 项：也是 NVPTX 专属
declare-to-assign
inferattrs
coro-early
function<eager-inv>(lower-expect
  simplifycfg<...>
  sroa<modify-cfg>
  early-cse<>)
openmp-opt
ipsccp
called-value-propagation
globalopt
function<eager-inv>(mem2reg
  instcombine<max-iterations=1;no-verify-fixpoint>
  simplifycfg<...>)
always-inline
require<globals-aa>
function(invalidate<aa>)
require<profile-summary>
cgscc(devirt<4>(inline
  function-attrs<...>
  ...
  function<eager-inv;no-rerun>(sroa<modify-cfg>
    early-cse<memssa>
    speculative-execution<only-if-divergent-target>
    jump-threading
    correlated-propagation
    simplifycfg<...>
    instcombine<...>
    aggressive-instcombine
    libcalls-shrinkwrap
    tailcallelim
    simplifycfg<...>
    reassociate
    constraint-elimination
    loop-mssa(loop-instsimplify
      loop-simplifycfg
      licm<no-allowspeculation>
      ...
      loop-unroll-full)
    ...
```

后面还有一大段（`loop-vectorize<...>`、`slp-vectorizer`、`loop-unroll<O2>`、`sroa<preserve-cfg>`、`gvn`、`memcpyopt`、`dse`、`adce`、`globaldce`、`constmerge`、`annotation-remarks`、`print`），我们不逐项背，按**阶段**归纳成一张表就够了：

| 阶段 | 代表 pass | 在本案例里干了什么 |
| --- | --- | --- |
| 模块级先手 | `memprof-remove-attributes`、`annotation2metadata`、`forceattrs`、**`nvvm-reflect`**、**`nvvm-intr-range`**、`inferattrs` | 把 `__nvvm_reflect` 折成常数、给 NVVM intrinsic 补范围信息；再补全属性 |
| 早期简化 | `sroa`、`early-cse`、`mem2reg`、`instcombine`、`simplifycfg` | **消灭 alloca**、做局部折叠 |
| 过程间 | `inline`、`function-attrs`、`ipsccp`、`globalopt` | 本例的 kernel 没被内联（`__global__` 不能内联），但 helper 全被内联 |
| 循环层 | `licm`、`loop-unroll-full`、`loop-unroll<O2>`、`loop-vectorize`、`loop-sink` | K 是编译期常量时**循环会被展开**；K 是运行期参数时不会（5.5 节） |
| 向量化 | `loop-vectorize`、`slp-vectorizer` | 对 `<4 x i8>` 之外的标量代码有机会；本例没触发 |
| 收尾 | `gvn`、`dse`、`adce`、`globaldce`、`constmerge` | 清死代码 |

### 第 4 项 `nvvm-reflect` 为什么排这么靠前？

因为它**不是** clang 中端 pipeline 自带的，而是 **NVPTX 目标自己插进来的**。看源码（`llvm/lib/Target/NVPTX/NVPTXCodeGenPassBuilder.cpp:321`）：

```cpp
void NVPTXTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  ...
  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &PM, OptimizationLevel Level) {
        // We do not want to fold out calls to nvvm.reflect early if the user
        // has not provided a target architecture just yet.
        if (Subtarget.hasTargetName())
          PM.addPass(NVVMReflectPass(Subtarget.getSmVersion()));

        FunctionPassManager FPM;
        FPM.addPass(NVVMIntrRangePass());
        ...
      });
}
```

这是新 PM 的"扩展点"机制：目标机注册一个回调，PassBuilder 在建 pipeline 时把它挂在**最前面**（`registerPipelineStartEPCallback`）。所以 `nvvm-reflect` 和 `nvvm-intr-range` 才会出现在通用 O2 流程的前几步里。

同一个文件里，传统 codegen 路径还有第二处（`NVPTXTargetMachine.cpp:244` 的 `NVPTXPassConfig::addIRPasses`，里面第 270 行注册）：

```cpp
addPass(createNVVMReflectPass(ST.getSmVersion()));
```

记住这个机制，第 10 章我们给新指令加 pass 时会再用到它。

## 5.2 SROA + mem2reg：796 行 → 353 行

### 一个必须先说的坑：`optnone`

你如果直接拿 `dumps/01-device-O0.ll` 去跑 `opt`，会发现**什么都没变**。原因不是 pass 失效，而是 `-O0` 的 clang 会给每个函数打上 `optnone` 属性：

```llvm
attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone ... }
                                                             ^^^^^^^
```

凡是带 `optnone` 的函数，绝大多数 pass 会直接跳过（这是 LLVM 的硬规矩：`-O0` 的语义就是"不要优化"）。所以做实验时要先去掉它——这是 LLVM 生态里很常见的操作，测试用例里到处都是：

```bash
clang++ -x cuda ... --cuda-device-only -O0 -Xclang -disable-O0-optnone \
        -S -emit-llvm -o dumps/01b-device-O0-nooptnone.ll tc_mma.cu
```

### 实验结果

```bash
opt -passes='sroa,mem2reg'                  -S 01b-device-O0-nooptnone.ll -o 15-after-sroa-mem2reg.ll
opt -passes='sroa,mem2reg,instcombine'      -S 01b-device-O0-nooptnone.ll -o 16-after-instcombine.ll
```

```
01-device-O0.ll                  796 行      <- clang -O0 原样（带 optnone）
01b-device-O0-nooptnone.ll       796 行      <- 去掉 optnone，给 opt 用
15-after-sroa-mem2reg.ll         353 行      <- 少了一半多
16-after-instcombine.ll          338 行
02-device-O2.ll                  291 行      <- clang -O2 的最终形态
```

**796 → 353 这一步就是 SROA 和 mem2reg 的功劳。** 我们看具体变化。优化前（`01b`）：

```llvm
  %11 = alloca %struct.__half2, align 4
  %10 = alloca %struct.__half2, align 4
  ...
  %222 = load ptr, ptr %10, align 8
  %223 = getelementptr inbounds float, ptr %222, i64 3
  %224 = load float, ptr %223, align 4
  %225 = call contract { float, float, float, float } asm sideeffect "mma.sync..."(i32 %199, ...) #7, !srcloc !6
```

优化后（`15`）：

```llvm
  %80 = call contract { float, float, float, float } asm sideeffect "mma.sync..."(
         i32 %46, i32 %49, i32 %52, i32 %55, i32 %75, i32 %79,
         float %.sroa.097.0, float %.sroa.499.0, float %.sroa.7.0, float %.sroa.10.0) #7, !srcloc !6
```

**`alloca` + `load`/`store` 的那一圈全部消失**：fragment 的输入直接以 SSA 值的形式进 asm 调用（`float %.sroa.097.0` 这种带 `.sroa.` 前缀的名字正是 SROA 拆数组留下的痕迹——它把 `float c[4]` 拆成了 4 个独立标量）。这就是第 4 章说的"从内存世界搬到值世界"。

顺带说一个细节：`opt -passes='sroa,mem2reg'` 之所以"够用"，是因为我们的数组都是**常量下标**（`a[0]`、`a[1]`、…）。如果下标是运行期变量，SROA 就不能拆分，那就得靠后续的 `gvn`/`dse` 或者干脆真的落到局部内存（SASS 里会出现 `LDL/STL`，还会占 local memory）——**Tensor Core 代码里"不要让 fragment 数组被动态下标访问"是个硬性经验**，Intel/AMD/NVIDIA 的手册都这么说，原因就在这里。

## 5.3 InstCombine：353 行 → 338 行

`instcombine` 是 LLVM 里最"碎"的 pass，它专门做 peephole 级别的代数化简。我们案例里最典型的几处，在最终的 O2 IR 里都能指出来：

```llvm
  %12 = lshr i32 %7, 2               ; gid = lane >> 2
  %14 = shl nuw nsw i32 %7, 1        ; lane * 2
  %15 = and i32 %14, 6               ; (lane * 2) & 6
  %16 = or disjoint i32 %15, 8       ; (lane * 2) & 6 | 8
```

你写的是 `gid = lane >> 2`、`tig = lane & 3` 和 `tig * 2`。两处改写值得看：

- `(lane & 3) * 2` → `(lane * 2) & 6`：乘 2 就是左移 1 位，低位一定是 0，所以掩码可以从 `& 3` 提到乘法之后变成 `& 6`。**结果是少了一条指令**（原来要 `and` + `shl`，现在是 `shl` + `and`……数目一样，但掩码常量变小了，后面更容易和别的地址计算合并）。
- `lane >> 2` 从 `ashr` 变成 `lshr`：源码里 `lane` 是 `int`，而 `threadIdx.x` 的取值范围被 `ValueTracking` 推出来了（非负），于是"算术右移"这种带符号语义的操作可以降级成逻辑右移。

这种"看起来没省什么、但在 SASS 上能少一条 `LOP3`"的微调，就是 `instcombine` 存在的意义。

`disjoint` 标记是 `or disjoint i32 %15, 8`：它告诉后端"这两个操作数的位不重叠"，于是这条 `or` 可以当成加法来处理，在地址计算里能直接折进 `IADD3` 或寻址模式。

想自己观察这些折叠，用下面这条命令最方便（只跑 instcombine，不跑别的）：

```bash
opt -passes='instcombine' -S 15-after-sroa-mem2reg.ll -o - | diff - 15-after-sroa-mem2reg.ll | head -40
```

## 5.4 NVVMReflect：`opt` 里是 0，真流水线里是 890

`__nvvm_reflect` 是 CUDA 生态里一个很特别的东西：libdevice（NVIDIA 提供的数学库 bitcode）里到处在用

```c
if (__nvvm_reflect("__CUDA_FTZ")) { /* FTZ 版本 */ } else { /* 非 FTZ 版本 */ }
```

来选择实现。`NVVMReflect` 这个 pass 的活就是把这类调用折成常数，然后把死掉的分支删掉。

我们准备了一份最小 IR（`code/reflect_demo.ll`），里面有两个 kernel：一个用 `__CUDA_FTZ` 选路径，一个直接返回 `__CUDA_ARCH`。先单独用 `opt` 跑：

```bash
opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -passes=nvvm-reflect -S code/reflect_demo.ll
```

```llvm
define float @fmul_ftz(float %a, float %b, float %c) #0 {
entry:
  br label %ftz                       ; <- 调用被折成常数 1，条件跳转随即被折掉

ftz:
  %m1 = fmul float %a, %b
  %s1 = fadd float %m1, %c
  ret float %s1

nftz:                                 ; No predecessors!
  ...
define i32 @arch_id() #0 {
entry:
  ret i32 0                            ; <- __CUDA_ARCH 折成了 0 !?
}
```

注意 `nftz` 那个块被标上了 `No predecessors!`：`br i1` 在 `__CUDA_ARCH` 折成常数之后被判死，非 FTZ 那条路径虽然还在文本里，却已经没有任何前驱了。**这就是"死分支"在 IR 里的样子**——它得等后面某个 CFG 清理 pass 才被真正删掉。

那个 `0` 不是笔误：**单独跑 `opt` 时，pass 拿不到 SM 版本**（它是由 `NVVMReflectPass(Subtarget.getSmVersion())` 构造进来的，`opt -passes=nvvm-reflect` 用的是默认值）。而在真实的 NVPTX 流水线里（`llc` 或者 clang 编译设备代码），SM 版本是已知的：

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -o - code/reflect_demo.ll
```

```
        .visible .func  (.param .b32 func_retval0) arch_id()
        {
        // %bb.0:
                st.param.b32    [func_retval0], 890;   ; sm_89 -> 89 * 10 = 890
                ret;
        }
```

那个 `890` 被直接折进了 `st.param` 的立即数里，连 `mov.b32` 都省了——这是指令选择阶段就把常数传播做完的结果。

再把折完之后的两条分支过一次 `simplifycfg + instcombine`，`%is_ftz` 的 `icmp ne i32 1, 0` 变 true，非 FTZ 那条分支就被删掉了：

```bash
opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 \
    -passes='nvvm-reflect,function(simplifycfg,instcombine)' -S code/reflect_demo.ll
```

```llvm
define float @fmul_ftz(float %a, float %b, float %c) #0 {
entry:
  %m1 = fmul float %a, %b
  %s1 = fadd float %m1, %c
  ret float %s1
}
```

### FTZ 从哪来？

还有一个开关：`nvvm-reflect-ftz` 是**模块级 flag**。在 `01-device-O0.ll` 末尾能看到：

```llvm
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
```

`NVVMReflect::populateReflectMap` 就是读这个 flag 来决定 `__CUDA_FTZ` 折成 0 还是 1；命令行也留了口子：

```bash
opt -passes=nvvm-reflect -nvvm-reflect-add='__CUDA_FTZ=1' ...
```

（源码：`llvm/lib/Target/NVPTX/NVVMReflect.cpp` 的 `populateReflectMap` 与 `-nvvm-reflect-add` 定义。）

**这一节的教学价值在于**：它让你看到"中端 pipeline"和"目标机扩展点"是混在一起的。你看到的那 121 项里，有一部分是 target 自己插进来的，读 pipeline 时别以为全是通用的 O2 流程。

## 5.5 LoopUnroll：四个实验，"convergent 挡展开"只对一半情形成立

我们的 k 循环长这样（`02-device-O2.ll`）：

```llvm
58:  %59 = phi i32 [ 0, %23 ], [ %96, %58 ]
     ...
     %91 = tail call contract {...} asm sideeffect "mma.sync..."(...) #4
     %96 = add nuw nsw i32 %59, 16
     %97 = icmp slt i32 %96, %5           ; 和运行期参数 K 比较
     br i1 %97, label %58, label %38, !llvm.loop !18
```

### 实验一：K 是运行期参数 → 不可能展开

`%5` 是函数参数（`K`），SCEV 拿不到 trip count，**任何 unroll 都得放弃**。这没什么好说的。

### 实验二：把 K 变成编译期常量，循环就被展开了

`code/unroll_demo.cu` 是同一份代码，但 `K` 是 `constexpr int K = 64`。编译出来：

```
17-kconst-O2.ll     mma asm 调用 = 4     循环 br i1 = 0
```

**循环没了，4 条 `mma` 平铺在基本块里。** 也就是说：只要 trip count 编译期可见，`-O2` 的流水线（`loop-unroll-full`）会直接把圈数展开完，**convergent 拦不住它**。

这条结论很值得停一下：**"convergent 挡住展开"这个说法，只对一半情形成立**。展开分成两种：

| 展开方式 | 需要 remainder 循环吗 | 含 convergent 时能不能做 |
| --- | --- | --- |
| **完全展开**（trip count 已知，一次铺开） | 不需要 | **能**——没有新增控制流依赖，warp 的收敛性不受影响 |
| **运行时展开**（trip count 未知，按 4 份 + 1 份余数） | 需要 | **不能**——余数循环本身就是"给 convergent 操作套了一层额外控制流依赖" |

### 实验三：运行期 K 的循环，convergent 真的挡住了展开

拿我们的主案例（`K` 是函数参数）做对照。它是标准的 runtime unroll 场景，我们手工把 unroller 打开（`tools/convergent-experiment.sh` 里那条命令的等价写法）：

```bash
# A) 原样（asm 带 convergent）
opt -passes='loop-unroll' -unroll-runtime -unroll-count=4 -S dumps/02-device-O2.ll -o a.ll

# B) 先把 convergent 从 IR 里抹掉，再跑同一条命令
opt -passes='loop-unroll' -unroll-runtime -unroll-count=4 -S noconvergent.ll -o b.ll
```

```
mma_tc_manual    A) mma 条数 = 1    <- 一点没展开
mma_tc_manual    B) mma 条数 = 4    <- 展开成 4 份
```

**同一份 IR、同一条命令，唯一的差别就是 `convergent`。** 这就是那条规则的实际后果：运行期展开要造余数循环，而余数循环会给 convergent 操作新增一个控制流依赖，LLVM 直接放弃。

### 实验四：`#pragma unroll 4`

`code/unroll_pragma_demo.cu` 只是多了一行 `#pragma unroll 4`：

```
unroll_demo.cu        (无 pragma): mma 条数 = 4
unroll_pragma_demo.cu (有 pragma): mma 条数 = 4
unroll_pragma_demo.cu             循环 br = 0     <- 循环彻底消失
```

这份代码里两种写法的结果一样——因为 trip count 已知时 `loop-unroll-full` 本来就会展开。`#pragma unroll N` 的价值在另外两种场合：trip count 已知但**代价模型认为不划算**时，它会在循环的 `!llvm.loop` 元数据里挂上 `llvm.loop.unroll.count`，明确要求 unroller 照做；或者 trip count 有上界但编译器证明不了精确值时，帮它选定展开因子。**它是"我要你展开"的显式指令，代价模型管不了它。**

### 那 convergent 在这件事上到底管什么？

LLVM 对"循环里有 convergent 操作"确实有额外限制，源码写得很清楚（`llvm/lib/Transforms/Scalar/LoopUnrollPass.cpp:1372`）：

```cpp
  // If the loop contains a convergent operation, the prelude we'd add
  // to do the first few instructions before we hit the unrolled loop
  // is unsafe -- it adds a control-flow dependency to the convergent
  // operation. Therefore restrict remainder loop (try unrolling without).
  UP.AllowRemainder &= UCE.ConvergenceAllowsRuntime;
```

以及同文件 720 行：

```cpp
  if (Convergence == ConvergenceKind::ExtendedLoop) {
    ReportCannotUnroll("contains convergent operations");
    return false;
  }
```

它的意思是：**展开往往需要造一个"remainder 循环"（处理不足一整轮的尾巴），这会引入额外的控制流分支，而 convergent 操作不能被随便放到新的控制流依赖下面**。所以含 convergent 的循环，unroller 会更保守——它可能放弃 runtime unroll，只留下完全展开这条路（第 1401 行还有一条 `UP.Runtime &= UCE.ConvergenceAllowsRuntime;`）。

对照实验二和实验三：**trip count 已知时 convergent 不挡路（完全展开没有新增控制流依赖），trip count 未知时 convergent 就是那道墙**。至于"不挡路时展开多少"，才是代价模型说了算的。

### 最后，ptxas 在背后做了补偿

别忘了 SASS。虽然运行期 K 的循环在 IR 层没展开，但 `ptxas` 在生成 SASS 时把循环拆成了 **4x + 2x + 1x** 三级结构（`tools/analyze-sass.sh` 的输出）：

```
mma_tc_manual    HMMA=7   LDG=56  STG=4
```

7 个 HMMA 站点 = 主循环里 4 个（`0x4b0/0x5f0/0x730/0x820`）+ 中间循环 2 个（`0xa50/0xbb0`）+ 收尾 1 个（`0xd60`）。这是标准的"按 4/2/1 分解任意 trip count"展开。**所以"IR 没展开"不等于"机器码没展开"**——这条经验在你调优时非常值钱。

## 5.6 convergent 的第二道防线：为什么 `mma` 挪不动

第 3 章我们知道：CUDA 里的每条内联汇编都被标了 `convergent`。这一节看它在后端造成的后果。

### 从 IR 到 MIR：标记是怎么传下去的

`llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp:10210`：

```cpp
    if (Call.isConvergent())
      Flags |= InlineAsm::Extra_IsConvergent;
```

于是 MIR 里那条 `INLINEASM` 长这样（`dumps/09-mir-after-isel.mir`）：

```
INLINEASM &"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};\0A",
          sideeffect isconvergent attdialect,
          regdef:B32, def %78, regdef:B32, def %79, ...
```

两个标志：`sideeffect`（来自 `asm volatile`）+ `isconvergent`。老版本这里打印的是个数字（`33`），新版直接打名字，读起来省事多了。

### 后端里"看见 convergent 就收手"的 pass

`MachineInstr::isConvergent()`（`llvm/include/llvm/CodeGen/MachineInstr.h:1082`）就是读上面那个标志位：

```cpp
  bool isConvergent(QueryType Type = AnyInBundle) const {
    if (isInlineAsm()) {
      unsigned ExtraInfo = getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
      if (ExtraInfo & InlineAsm::Extra_IsConvergent)
        return true;
    }
    ...
```

而 MachineSink 见到它就直接返回（`llvm/lib/CodeGen/MachineSink.cpp:424`）：

```cpp
  // Convergent operations may not be made control-dependent on additional
  // values.
  if (MI.isConvergent())
    return false;
```

同一份代码里还有另外两处同样的检查（第 752、1857 行），分别属于不同的下沉路径。

**"不能被 MachineSink"这句话的完整含义是**：不能让这条指令变成"只在某个条件成立时才执行"。因为 `mma.sync` 是 warp 级同步操作——32 个线程必须一起到达这条指令；如果它被下沉到一个只有部分线程会走的路径里，硬件行为就未定义了（轻则结果错，重则死锁）。同一条理由适用于各种 `bar.sync`、`shfl`、`redux`。

### 受影响的 pass 不止 MachineSink

搜一下谁检查 `isConvergent()`：

```
lib/CodeGen/MachineSink.cpp
lib/CodeGen/MachineLICM.cpp          <- 不能提升出循环
lib/CodeGen/MachineCSE.cpp           <- 不能做公共子表达式合并
lib/CodeGen/IfConversion.cpp         <- 不能做 if-conversion（谓词化）
lib/CodeGen/TailDuplicator.cpp       <- 不能复制到尾部
lib/CodeGen/MachineConvergenceVerifier.cpp   <- 专门验证收敛性
lib/CodeGen/GlobalISel/InlineAsmLowering.cpp
```

也就是说，**一条 `mma` 汇编进来，等于给一整族机器级优化按了暂停键**。这解释了你在 CUDA 里可能见过的现象：把某个操作写成内联汇编之后，周围代码的优化质量会突然下降。这不是编译器变笨了，是它必须保守。

### 能不能把这个标记摘掉？

我们试了三条路（`tools/convergent-experiment.sh` 的 G 段）：

| 做法 | 结果 |
| --- | --- |
| `-fno-convergent-functions` | **生效**，整个设备端 IR 里不再有 `convergent`（驱动确实把它传给了设备端 cc1） |
| `__attribute__((noconvergent))` | 语法错误：`an attribute list cannot appear here` |
| `[[clang::noconvergent]]` 直接写在 asm 前 | 编译能过，但警告 `'clang::noconvergent' attribute ignored [-Wignored-attributes]`；asm 用到的属性组仍是 `{ convergent nounwind }` |
| `[[clang::noconvergent]] { asm ...; }` | **生效**：asm 用到的属性组变成 `{ nounwind }` |

`tools/convergent-experiment.sh` 的 G 段把这三条并排跑了一遍，实测输出：

```
  with_convergent          属性组 #3  含 convergent: 是
  with_noconvergent        属性组 #3  含 convergent: 是     <- 直接挂属性，无效
  with_noconvergent_stmt   属性组 #4  含 convergent: 否     <- 挂语句块，有效
```

还有一个"暴力"办法：直接改 IR 把 `convergent` 抹掉，再喂给 `llc`。我们在实验三里就是这么做的（`dumps/21-kconst-noconvergent-ir.ll`），用来证明 runtime unroll 被拒绝确实是因为这个标记。**但请不要在生产代码里这么干**——除非你百分百确定那段汇编里没有 warp 级同步语义。

## 5.7 本章小结与衔接

把本章结论压缩成五条：

1. `-O2` 的设备端 pipeline 有 121 项，**其中 `nvvm-reflect`、`nvvm-intr-range` 这类 NVPTX 专属 pass 是目标机用扩展点插进来的**，不是通用 O2 的一部分。
2. `SROA + mem2reg` 是"数组消失"的元凶：796 → 353 行；做实验时记得先 `-Xclang -disable-O0-optnone`。
3. `InstCombine` 干的是小账本：`(lane&3)*2` → `(lane*2)&6`，以及 `or disjoint` 这类提示。
4. **循环展不展开要分两种**：trip count 编译期已知时（`K` 是 `constexpr`），`-O2` 会直接完全展开，convergent 拦不住；trip count 未知时（`K` 是函数参数），**convergent 会挡住 runtime unroll**——要展开就得先摘掉这个标记。而 `ptxas` 在 SASS 层已经帮你做了 4x/2x/1x 展开，所以"IR 没展开"不等于"机器码没展开"。
5. `convergent` 标记会一路传到 MIR，让 MachineSink / MachineLICM / MachineCSE / IfConversion / TailDuplicator 全部收手。这是"mma 不能动"的完整理由。

到这里，IR 层的账算完了。下一章我们进入后端：**这份 `-O2` 的 IR 是怎么变成 `MachineFunction` 的**——`getelementptr` 变成什么、`load i32` 变成什么、`call asm` 又是怎么进 MIR 的。
