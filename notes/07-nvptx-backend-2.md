# 第 7 章 · NVPTX 后端 lowering（下）：Machine IR → PTX

## 7.0 本章目标

上一章我们走到了 Machine IR。这一章把它推到 PTX 文本，并且要打碎一个几乎所有人都有的误解：

> "LLVM 后端会做寄存器分配，把虚拟寄存器映射到物理寄存器。"

在 NVPTX 上，**这句话是错的**。本章会给你源码和 dump 两方面的证据。

## 7.1 机器层 pass 逐段看

从上一章那份 165 项的清单里，把机器层的部分单独拎出来（`dumps/25-codegen-passes.txt`）：

```
61  NVPTX DAG->DAG Pattern Instruction Selection        <- ISel
62  Finalize ISel and expand pseudo-instructions
63  Early Tail Duplication
64  Optimize machine instruction PHIs
65  Slot index numbering
66  Merge disjoint stack slots
67  Local Stack Slot Allocation
68  Remove dead machine instructions
69  Early Machine Loop Invariant Code Motion
70  Machine Common Subexpression Elimination
71  Machine code sinking
72  Peephole Optimizations
73  Remove dead machine instructions
74  NVPTX Forward Params                                  <- NVPTX 专属
75  NVPTX Address Folder                                  <- NVPTX 专属
76  NVPTX Proxy Register Instruction Erasure              <- NVPTX 专属
77  Process Implicit Definitions
78  Remove unreachable machine basic blocks
79  Live Variable Analysis
80  Eliminate PHI nodes for register allocation
81  Two-Address instruction pass
82  Slot index numbering
83  Register Coalescer
84  Machine Instruction Scheduler
85  Stack Slot Coloring
86  NVPTX Prolog Epilog Pass                              <- NVPTX 专属
87  NVPTX optimize redundant cvta.to.local instruction    <- NVPTX 专属
90  Control Flow Optimizer
91  Post-RA pseudo instruction expansion pass
93  Branch Probability Basic Block Placement
```

我们挑三个讲。

### `NVPTX Proxy Register Instruction Erasure`（第 76 项）

名字很唬人，干的事很简单：删掉一类叫"proxy register"的伪指令。

背景：NVPTX 在生成 kernel 调用序列（`callseq_start` / `callseq_end`）时，需要保证这些伪指令不会被优化掉，于是用一个"代理寄存器"把它们拴在一起。等机器层优化都跑完，这个代理就没有存在的必要了，这个 pass 负责清理。注册点是 `NVPTXPassConfig::addPreRegAlloc()`（`NVPTXTargetMachine.cpp:340`）：

```cpp
void NVPTXPassConfig::addPreRegAlloc() {
  addPass(createNVPTXForwardParamsLegacyPass());
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createNVPTXAddressFolderLegacyPass());
  // Remove Proxy Register pseudo instructions used to keep `callseq_end` alive.
  addPass(createNVPTXProxyRegErasureLegacyPass());
}
```

注意这一版在它前面还插了 `NVPTXForwardParams` 和 `NVPTXAddressFolder`（后者只在优化构建里加）。我们这份代码里没有 kernel 调用，所以 proxy reg 那个 pass 空转——但它的存在说明了一件事：**NVPTX 有一套自己的伪指令体系，需要在机器层专门维护。**

### `Machine code sinking` / `Machine CSE`（第 73、74 项）

第 5 章讲过：`mma` 是 `convergent`，MachineSink 见到它就返回 false。这里你在真实流水线里看到这个 pass 排在 PHI 消除之前，可以对照 `dumps/26-isel.mir` 观察：`mma` 那条 `INLINEASM` 从头到尾**没有跨基本块移动过**。

`Machine CSE` 同理——它不会把两条一模一样的 `mma` 合并成一条（哪怕操作数完全相同），因为合并会改变"warp 每个线程执行多少次"这个语义。

### `Register Coalescer`（第 83 项）

这个 pass 在 NVPTX 上**特别重要**，因为它承担了"减少寄存器数量"的全部责任。它的活是：把 `COPY` 链合并掉，让 `%a = COPY %b; use %a` 变成 `use %b`。

在上层 IR 里，`uint32_t a[4]`、`b[2]`、`float c[4]` 这些变量在 SSA 形式下会产生大量 `COPY`；到了这里，coalescer 把它们尽量并到同一个虚拟寄存器上。**这是 NVPTX 路径下唯一一次"降低寄存器压力"的机会**——LLVM 侧报出来的 PTX 虚拟寄存器数量，直接决定 ptxas 的压力。

## 7.2 反直觉的核心事实：NVPTX 不做寄存器分配

先看源码（`llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp:358`）：

```cpp
FunctionPass *NVPTXPassConfig::createTargetRegisterAllocator(bool) {
  return nullptr; // No reg alloc
}

void NVPTXPassConfig::addFastRegAlloc() {
  addPass(&PHIEliminationID);
  addPass(&TwoAddressInstructionPassID);
}

void NVPTXPassConfig::addOptimizedRegAlloc() {
  addPass(&ProcessImplicitDefsID);
  addPass(&LiveVariablesID);
  addPass(&MachineLoopInfoID);
  addPass(&PHIEliminationID);

  addPass(&TwoAddressInstructionPassID);
  addPass(&RegisterCoalescerID);

  // PreRA instruction scheduling.
  if (addPass(&MachineSchedulerID))
    printAndVerify("After Machine Scheduling");

  addPass(&StackSlotColoringID);

  // FIXME: Needs physical registers
  // addPass(&MachineLICMID);

  printAndVerify("After StackSlotColoring");
}
```

**`return nullptr; // No reg alloc`** —— 目标机自己说"我不要寄存器分配器"。

那 `addOptimizedRegAlloc()` 里这一串在干什么？它在做**编码前的准备**：消除 PHI、处理两地址指令、合并 COPY、预分配指令调度、合并栈槽。**唯独没有"把虚拟寄存器映射到物理寄存器"这一步。**

### 三份证据

**证据一：pass 列表里没有分配器。** 对比我们在 MIPS 上看到的流程（那里有 `Greedy Register Allocator`），NVPTX 这份 165 项清单的第 79–85 项只有 PHI 消除、Two-Address、Coalescer、Scheduler、Stack Slot Coloring，**没有 Greedy/RegAllocBasic/RegAllocFast**。

**证据二：流水线最后一步的 MIR 里还是虚拟寄存器。** 看 `dumps/28-prologepilog.mir`（`-stop-after=nvptx-prolog-epilog`，编码前最后一站）：

```
    INLINEASM &"{  mov.b32 $0, {$1,$2};}\0A", isconvergent attdialect,
              regdef:B32, def %72, reguse:B16, %73, reguse:B16, %74, !17
    INLINEASM &"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};\0A",
              sideeffect isconvergent attdialect,
              regdef:B32, def %125, regdef:B32, def %126, regdef:B32, def %127, regdef:B32, def %128,
              reguse:B32, %82, ... , reguse:B32, %125, reguse:B32, %126, reguse:B32, %127, reguse:B32, %128, !18
```

`%125`、`%128`、`%72` 全是**虚拟寄存器编号**（`def %125` / `reguse %125` 里那个数字），一直活到最后一站。

顺带注意 `mma` 的 C 输入（`reguse` 的最后 4 个）就是它自己的 D 输出 `%125..%128`——`D = A*B + C` 里的"原地累加"在这个层级还看得见。

**证据三：PTX 里它们还是"虚拟"的。** 发射出来的 PTX：

```
	.reg .pred 	%p<3>;
	.reg .b16 	%rs<5>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<40>;
```

PTX 是**虚拟 ISA**：`%r31`、`%rs2` 这些名字对硬件没有意义，它们只是给 ptxas 看的符号。真正的物理寄存器由 **ptxas** 分配。

还有个细节值得圈一下：**声明里没有 `.reg .f32`**。我们的累加器明明是 `float c[4]`，但它们全程住在 `.b32` 寄存器里（第 6.4 节说过，`=f` 约束在 NVPTX 里也归 `B32` 类）。**PTX 的虚拟寄存器是按"宽度"分类的，不是按"类型"。**

### 为什么这么设计？

因为 PTX 的抽象层次就是"无限寄存器的 RISC 汇编"。把寄存器分配推给 ptxas 有几个好处：

- ptxas 掌握真实硬件信息（bank 冲突、dual-issue、`.maxnreg`、occupancy 目标），能做出比 LLVM 更好的决策；
- 同一份 PTX 可以被不同版本的 ptxas、在 JIT 场景下重新分配寄存器；
- LLVM 侧只需要保证"能用、别太浪费"，不必实现一套完整的 SM 寄存器模型。

代价也很明确：**你在 LLVM 侧看到的"寄存器数"其实毫无意义**。第 2 章 `ptxas -v` 报的 40 个寄存器是 ptxas 的数。你调优时唯一该看的也是它。

## 7.3 MIR → PTX：AsmPrinter 干了什么

PTX 文本由 `NVPTXAsmPrinter` 生成。它做四件事，我们逐个看。

### 1）给虚拟寄存器编号：per-class 编号

源码在 `llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp` 的 `setAndEmitFunctionVirtualRegisters()`：

```cpp
  // Go through all virtual registers to establish the mapping between the
  // global virtual register number and the per class virtual register number.
  // We use the per class virtual register number in the ptx output.
  for (unsigned I : llvm::seq(MRI->getNumVirtRegs())) {
    Register VR = Register::index2VirtReg(I);
    if (MRI->use_empty(VR) && MRI->def_empty(VR))
      continue;
    auto &RCRegMap = VRegMapping[MRI->getRegClass(VR)];
    RCRegMap[VR] = RCRegMap.size() + 1;
  }

  // Emit declaration of the virtual registers or 'physical' registers for
  // each register class
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (const TargetRegisterClass &RC : TRI->regclasses()) {
    // Only declare those registers that may be used.
    const auto It = VRegMapping.find(&RC);
    if (It == VRegMapping.end() || It->second.empty())
      continue;

    TS->emitRegDirective(
        TRI->getRegSizeInBits(RC).getFixedValue(),
        NVPTX::getVirtualRegisterPrefix(getVirtualRegisterKind(&RC)),
        It->second.size() + 1);
  }
```

关键在注释那句：**"把全局虚拟寄存器编号映射成 per-class 的编号，从 1 开始"**。所以：

- MIR 里的 `%125`（全局编号，`B32` 类）→ PTX 里的某个 `%r`（该类里第 N 个）；
- MIR 里的 `%72`（`B32` 类）→ 同理；
- 声明写成 `.reg .b32 %r<37>;` 表示"这个函数里 b32 类用到了 %r1..%r36"，`<N+1>` 是因为编号从 1 开始、0 号留空。**这也是为什么 PTX 里的寄存器编号看起来总是比 MIR 里的小**：它们换了一套编号空间。

**这就是 MIR 虚拟寄存器编号和 PTX 寄存器名之间的唯一关系——它只是个重命名，不涉及任何分配。**

### 2）函数入口：`.visible .entry`

```
.version 8.7
.target sm_89
.address_size 64

.visible .entry mma_tc_manual(
	.param .u64 .ptr .align 1 mma_tc_manual_param_0,
	.param .u64 .ptr .align 1 mma_tc_manual_param_1,
	.param .u64 .ptr .align 1 mma_tc_manual_param_2,
	.param .u32 mma_tc_manual_param_3,
	.param .u32 mma_tc_manual_param_4,
	.param .u32 mma_tc_manual_param_5
)
```

`__global__` 函数变成 `.visible .entry`（可被 host 启动），`__device__` 函数会变成 `.func`。参数类型直接从 IR 签名来：三个 `ptr` → `.param .u64 .ptr .align 1`（新版的参数会带上 `.ptr` 和 `.align` 修饰，语义更准确），三个 `int` → `.param .u32`。

```
	.shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2As[512];
	.shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2Bs[256];
```

共享内存的 `addrspace(3) global` 变成 `.shared` 声明，512 = 16×16×2 字节，256 = 16×8×2 字节，`.align 16` 来自 IR 的 `align 16`。

### 3）内联汇编：原文照抄

```
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r33,%r34,%r35,%r36}, {%r22,%r23,%r24,%r25}, {%r20,%r21}, {%r33,%r34,%r35,%r36};
```

**和源码里写的一模一样**，只是 `%0..%13` 被替换成了具体寄存器名。这就是内联汇编的本质：LLVM 只负责"把操作数放进寄存器、把占位符换成寄存器名"，指令本身的语义它完全不理解（所以它只能保守地标 convergent）。

对照三层的操作数，你能看到整条链严丝合缝：

| 层 | D/C（4 个 f32） | A（4 个 b32） | B（2 个 b32） |
| --- | --- | --- | --- |
| 源码约束 | `"=f"` × 4 | `"r"` × 4 | `"r"` × 2 |
| MIR 类别 | `regdef:B32` × 4 | `reguse:B32` × 4 | `reguse:B32` × 2 |
| PTX 寄存器 | `%r33,%r34,%r35,%r36` | `%r22,%r23,%r24,%r25` | `%r20,%r21` |

### 4）普通指令：直译

```
	mov.u32 	%r1, %tid.x;                        <- INT_PTX_SREG_TID_x
	mov.u32 	%r11, %ctaid.y;                     <- INT_PTX_SREG_CTAID_y
	shl.b32 	%r12, %r11, 4;                      <- SHL32_ri
	setp.gt.s32 	%p1, %r10, 0;                   <- SETP_i32ri
	@!%p1 bra 	$L__BB0_1;                      <- CBranch
	ld.param::entry.b64 %rd8, [mma_tc_manual_param_2]; <- LD_i64
	cvta.to.global.u64 	%rd3, %rd8;            <- cvta_to_global_64
	ld.global.nc.b32 	%r22, [%rd38+-16];     <- LD_GLOBAL_NC_i32
	st.global.b32 	[%rd31], %r33;              <- ST_i32
```

`ld.global.nc` 里的 `nc` 是 **non-coherent**（只读数据缓存路径，等价于 `__ldg`），它从哪来？来自函数参数的 `readonly` 属性（第 4 章讲的 `readonly` + `noalias`）——第 6.4 节那个 `LD_GLOBAL_NC_i32` 的 opcode 名字里就写着这件事。**一条前端属性，最终决定了走哪条缓存路径，还顺着改了机器指令的名字**——这是"属性不是注释"的最好例证。

## 7.4 完整的产物对照表（本案例）

| 概念 | LLVM IR | Machine IR | PTX |
| --- | --- | --- | --- |
| 线程号 | `llvm.nvvm.read.ptx.sreg.tid.x()` | `INT_PTX_SREG_TID_x` | `mov.u32 %r1, %tid.x;` |
| 块号 | `...ctaid.y()` | `INT_PTX_SREG_CTAID_y` | `mov.u32 %r11, %ctaid.y;` |
| A 的读 | `load i32, ptr addrspace(1) ...` | `LD_GLOBAL_NC_i32` | `ld.global.nc.b32 %r22, [%rd38+-16];` |
| B 的读 | `load i16, ptr addrspace(1)` ×2 + `mov.b32` | 2×`LD_GLOBAL_NC_i16` + `INLINEASM` | `ld.global.nc.b16` ×2（打包成 32 位由 `PRMT` 完成，见第 9 章） |
| mma | `call asm sideeffect "mma.sync..."` | `INLINEASM sideeffect isconvergent` | `mma.sync.aligned...` |
| 同步 | `llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)` | `BARRIER_CTA_SYNC_ALIGNED_ALL_i 0` | `bar.sync 0;` |
| D 的写 | `store float, ptr addrspace(1)` | `ST_i32` | `st.global.b32 [%rd31], %r33;` |

## 7.5 资源用量到底属于谁

```bash
$ ptxas -arch=sm_89 -v dumps/03-clang-O2.ptx -o dumps/05-clang.cubin
ptxas info    : Compiling entry function 'mma_tc_ldmatrix' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 768 bytes smem, 388 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function 'mma_tc_manual' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers, 388 bytes cmem[0]
```

| 数字 | 谁决定的 | 说明 |
| --- | --- | --- |
| 40 / 40 registers | **ptxas** | 真正的物理寄存器数量，决定 occupancy |
| 0 spill | ptxas | 没有寄存器溢出，SASS 里不会有 LDL/STL |
| 768 bytes smem | LLVM（`__shared__` 数组大小） | 与 `.shared` 声明一致 |
| 1 barrier | LLVM（`llvm.nvvm.barrier.cta.sync.aligned.all`） | `__syncthreads` 带来的 |
| 388 bytes cmem[0] | LLVM/驱动 | kernel 参数区 |

**这一节最想让你记住的是**：想调 occupancy，改的是 ptxas 能看到的东西（PTX 的寄存器数量、`.maxnreg`、`__launch_bounds__`），而不是在 LLVM 侧找"寄存器分配器"。

## 7.6 小结与衔接

1. **NVPTX 后端没有寄存器分配器**（`createTargetRegisterAllocator` 直接 `return nullptr`）。LLVM 侧只做 PHI 消除、COPY 合并（coalescing）、预分配调度；真正的分配是 ptxas 干的。
2. 机器层 pass 里有四类 NVPTX 专属动作：参数转发（`NVPTXForwardParams`）、地址折叠（`NVPTXAddressFolder`）、伪指令清理（`NVPTXProxyRegErasure`）、前导/后记代码（`NVPTXPrologEpilog`，**通用的 `PrologEpilogInserter` 被它禁用了**）。这也解释了为什么 `-stop-after=prologepilog` 会在 NVPTX 上报错，得写 `-stop-after=nvptx-prolog-epilog`。
3. PTX 文本的生成规则很简单：**虚拟寄存器按类重新编号**（MIR 的 `%125` → PTX 的某个 `%rNN`），**内联汇编原文照抄**，**其他指令一一对应**。`ld.global.nc` 这种细节则由前端属性（`readonly`）一路传导而来，连机器 opcode 的名字都叫 `LD_GLOBAL_NC_i32`。

到这里，"IR → 机器码"这条链在 LLVM 内部就走完了。剩下两章把镜头拉近到 PTX 和 SASS 本身：**第 8 章逐条读 PTX（`mma.sync` / `ldmatrix` / `cp.async` 的操作数怎么算）**，**第 9 章拆 SASS（`HMMA.16816.F32` 的操作数到底是什么，`LDSM.16.M88.4` 里的 `M88` 是什么意思）**。
