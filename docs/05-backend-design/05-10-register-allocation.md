# 05-10 · 寄存器分配：通用算法，与 NVPTX 的"不分配"

这一讲讲后端最经典的问题：**怎么把无限多的虚拟寄存器塞进有限的物理寄存器里。**

顺便讲一个反直觉的事实：**NVPTX 后端压根不做这件事**。

## 一、问题的本质：不是"分配"，是"复用"

虚拟寄存器的数量不受限制（SSA 里每个值一个）。物理寄存器只有几十个。

所以问题的本质是：

> **怎么让多个虚拟寄存器复用同一个物理寄存器？**

约束只有一条**核心规则**：

```
如果两个虚拟寄存器的"活跃区间"有重叠，它们不能共用同一个物理寄存器。
否则后写的人会覆盖先写的人，程序就错了。
```

"活跃区间"（live interval）就是"这个值从被定义到最后一次被使用之间的那段代码"。

于是整个问题变成了一张图着色问题：

```
每个虚拟寄存器 = 图上一个点
两个虚拟寄存器的活跃区间有重叠 = 两点之间连一条边
物理寄存器 = 颜色
目标：给每个点涂色，使相邻的点颜色不同，且颜色数不超过物理寄存器数
```

**"寄存器分配"这个名字其实不太准确，叫"寄存器复用"更贴切**——
它不是在"分配资源"，而是在"安排复用"。

## 二、配套的三件事

真实实现还要处理三件事：

| 问题 | 处理方式 |
| --- | --- |
| **溢出（spill）** | 颜色不够时，把某些值放到栈上（local memory），用的时候再读回来 |
| **拆区间（splitting）** | 一个值活得很久，可以拆成几段，让它在不同段用不同寄存器 |
| **合并（coalescing）** | 把 `COPY` 的两个端点合并到同一个寄存器，减少指令 |

**注意"溢出"的代价非常大**：本来寄存器够用的时候零成本，溢出之后就变成访存了。
你在 `ptxas -v` 的输出里看到的那行：

```
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

就是在说"没有溢出"。**如果这里出现非零值，性能会明显下降**。

## 三、LLVM 的分配器

LLVM 里有几个寄存器分配器可选：

| 分配器 | 代码 | 特点 |
| --- | --- | --- |
| **Greedy**（默认） | `lib/CodeGen/RegAllocGreedy.cpp` | 贪心 + 拆区间，效果最好，最常用 |
| Basic | `RegAllocBasic.cpp` | 简单版，教学用 |
| Fast | `RegAllocFast.cpp` | 只在 `-O0` 用，不做活跃分析 |
| PBQP | `RegAllocPBQP.cpp` | 用"分区布尔二次规划"建模 |

Greedy 的大致流程（了解就好，不必背）：

```
1. 计算每个虚拟寄存器的活跃区间（LiveIntervals 分析）
2. 先尝试把 COPY 相关的区间合并掉（coalescing）
3. 按"优先级"排序（溢出代价高的先分配）
4. 逐个尝试分配：能放进已有寄存器就放
5. 放不下就尝试"拆区间"或"挤走优先级更低的邻居"
6. 实在不行就 spill（放到栈上）
```

**"贪心 + 拆分 + 溢出"这三招的组合，就是 Greedy 这个名字的来源。**

## 四、NVPTX 的答案：不分配

现在是这一讲最有意思的部分。看 NVPTX 的源码
（`NVPTXCodeGenPassBuilder.cpp:115`）：

```cpp
  // NVPTX has no register allocation; virtual registers are emitted directly.
  void addTargetRegisterAllocator(PassManagerWrapper &PMW, bool) override {}
```

**方法体是空的**，注释写得明明白白：

> *"NVPTX has no register allocation; virtual registers are emitted directly."*
> （NVPTX 不做寄存器分配；虚拟寄存器被直接发射出去。）

旧路径里也有对应的证据（`NVPTXTargetMachine.cpp:358`）：

```cpp
FunctionPass *NVPTXPassConfig::createTargetRegisterAllocator(bool) {
  return nullptr; // No reg alloc
}
```

**`return nullptr; // No reg alloc`** ——目标机自己说"我不要分配器"。

### 那它都做了什么

NVPTX 的 `addOptimizedRegAlloc` 里仍然有一堆 pass：

```cpp
Error NVPTXCodeGenPassBuilder::addOptimizedRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(ProcessImplicitDefsPass(), PMW);
  addMachineFunctionPass(UnreachableMachineBlockElimPass(), PMW);
  addMachineFunctionPass(RequireAnalysisPass<LiveVariablesAnalysis, ...>(), PMW);
  addMachineFunctionPass(RequireAnalysisPass<MachineLoopAnalysis, ...>(), PMW);
  addMachineFunctionPass(PHIEliminationPass(), PMW);
  addMachineFunctionPass(TwoAddressInstructionPass(), PMW);
  addMachineFunctionPass(RegisterCoalescerPass(), PMW);
  addMachineFunctionPass(MachineSchedulerPass(&TM), PMW);
  addMachineFunctionPass(StackSlotColoringPass(), PMW);
  // FIXME: Needs physical registers
  // addMachineFunctionPass(MachineLICMPass(), PMW);
  return Error::success();
}
```

**注意最后那两行注释：**

```cpp
  // FIXME: Needs physical registers
  // addMachineFunctionPass(MachineLICMPass(), PMW);
```

**MachineLICM 被注释掉了，理由是"它需要物理寄存器"**——因为 NVPTX 根本没分配物理寄存器。
这是"设计取舍留下的痕迹"，读源码时这类注释特别有价值。

### 怎么验证

三条独立的证据：

**证据一：pass 清单里没有分配器。** 用 `-print-after-all` 打印全部 pass 名字，
然后筛一下"名字里带 register 的"：

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 -print-after-all \
    -o /dev/null dumps/02-device-O2.ll 2>&1 \
  | grep -iE "register|regalloc|greedy" | sort -u
```

实测输出**只有三行**：

```
# *** IR Dump After Eliminate PHI nodes for register allocation (phi-node-elimination) ***:
# *** IR Dump After NVPTX Proxy Register Instruction Erasure (nvptx-proxyreg-erasure) ***:
# *** IR Dump After Register Coalescer (register-coalescer) ***:
```

**没有任何 Greedy/Basic/Fast RegAlloc。** PHI 消除的标题里虽然有
"for register allocation"（这是历史遗留的名字），但分配器本身不在。

**证据二：编码前的 MIR 里还是虚拟寄存器。** 看 `dumps/28-prologepilog.mir`：

```
INLINEASM &"mma.sync...", sideeffect isconvergent attdialect,
          regdef:B32, def %125, regdef:B32, def %126, ..., !18
```

`%125`、`%126` 这些是**虚拟寄存器编号**，一直活到最后一站。

**证据三：PTX 里它们还是"虚拟"的。**

```
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<40>;
```

**`%r1`、`%rd2` 这些名字对硬件没有意义**——它们只是给 ptxas 看的符号。
真正的物理寄存器，由 ptxas 分配。

## 五、为什么可以这么设计

因为 **PTX 的抽象层次是"无限寄存器的 RISC 汇编"**。

把寄存器分配推给下游（ptxas）有几个好处：

1. **ptxas 知道真实硬件**：bank 冲突、dual-issue、`.maxnreg`、occupancy 目标……
   这些信息 LLVM 这一侧拿不到（或者要靠额外的数据建模）；
2. **同一份 PTX 可以适配不同代际**：JIT 场景下，PTX 可以被不同版本的 ptxas
   重新编译和重新分配寄存器；
3. **LLVM 侧只需要保证"能用、别太浪费"**，不必实现一整套 SM 寄存器模型。

代价也很明确：

> **你在 LLVM 侧看到的"寄存器数"其实没有意义。**

第 7 部分我们会看到，`ptxas -v` 报的"40 个寄存器"才是真实数字，
而它是由 ptxas 分配出来的。你想调 occupancy，要改的是"ptxas 能看到的东西"
（PTX 的虚拟寄存器数量、`.maxnreg`、`__launch_bounds__`），
而不是去 LLVM 里找"寄存器分配器"。

## 六、连带影响：合并这一步变得特别重要

既然没有分配器，那么"减少寄存器数量"的唯一机会就是 **Register Coalescer**（合并）。

在别的后端上，合并只是"省几条 mov"；在 NVPTX 上，它**直接决定 PTX 里虚拟寄存器的数量**，
而这个数量又会影响 ptxas 的分配压力和最终的 occupancy。

所以你会看到 NVPTX 的流水线里，合并是被认真安排过的（就在 PHI 消除和 Two-Address 之后）。

## 七、小结

1. 寄存器分配的本质是**图着色式的"复用安排"**，核心规则是"活跃区间重叠就不能共用"；
   配套三件事：合并、拆区间、溢出（spill）。
2. **NVPTX 不做寄存器分配**（`return nullptr; // No reg alloc`，
   新路径里注释写着 "virtual registers are emitted directly"），
   因为 PTX 本身就是"无限寄存器"的虚拟 ISA，这件事它推给了 ptxas。
3. 三条验证路径：pass 清单里没有分配器、MIR 里还是虚拟寄存器、PTX 里也是 `%rN` 符号。
   **代价是 LLVM 侧的"寄存器数量"没有意义，调优要看 `ptxas -v`。**

## 八、动手题

1. 用 `-print-after-all` 找出所有和寄存器有关的 pass，确认没有分配器：

   ```bash
   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 -print-after-all \
       -o /dev/null dumps/02-device-O2.ll 2>&1 \
     | grep -iE "register|regalloc|greedy" | sort -u
   ```
2. 对比一下"谁在分配寄存器"：找一个 x86 的例子（`clang -S -emit-llvm` 然后 `llc`），
   用同样的命令看它的 pass 清单里有没有 Greedy Register Allocator。
   （提示：你手边这台机器的 host 就是 x86-64，用来对比正合适。）

下一讲我们看机器层的另外三件事：调度、帧降低、调用约定。
