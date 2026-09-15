# 06-1 · 活标本：NVPTX 的整体结构与取舍

第 5 部分最后那张"零件清单"，我们这一讲把它**一格一格对上 NVPTX 的真实文件**。
对完之后你会发现：NVPTX 是个"薄一半、厚一半"的后端。

## 一、先对清单：每个零件都在

第 05-1 讲那张表，右列就是 NVPTX 的实体。放在一起看：

| 零件 | NVPTX 的文件 |
| --- | --- |
| `TargetMachine` | `NVPTXTargetMachine.h/.cpp` |
| `TargetSubtargetInfo` | `NVPTXSubtarget.h/.cpp` |
| `TargetLowering` | `NVPTXISelLowering.h/.cpp` |
| `TargetInstrInfo` | `NVPTXInstrInfo.h/.cpp` |
| `TargetRegisterInfo` | `NVPTXRegisterInfo.h/.cpp` |
| `TargetFrameLowering` | `NVPTXFrameLowering.h/.cpp` |
| 指令描述（`.td`） | `NVPTX.td`、`NVPTXInstrInfo.td`、`NVPTXIntrinsics.td`、`NVPTXRegisterInfo.td` |
| 指令选择器 | `NVPTXISelDAGToDAG.cpp` + 生成的 `NVPTXGenDAGISel.inc` |
| `AsmPrinter` | `NVPTXAsmPrinter.h/.cpp` |
| MC 层 | `MCTargetDesc/`（6 个文件） |
| `PassConfig` / PassBuilder | `NVPTXTargetMachine.cpp`（旧）、`NVPTXCodeGenPassBuilder.cpp`（新） |

自己验证一下：

```bash
ls /root/llvm-project/llvm/lib/Target/NVPTX/
```

**每个 LLVM 后端都是这个形状。** 你以后看 RISC-V、AMDGPU、WebAssembly，
都能用这张表当目录。

## 二、"薄"在哪儿：把活推给下游

NVPTX 后端有两处明显的"薄"，都来自同一个原因——**PTX 是虚拟 ISA**。

### 薄处一：不做寄存器分配

第 05-10 讲讲过，这里只放证据：

```cpp
  // NVPTX has no register allocation; virtual registers are emitted directly.
  void addTargetRegisterAllocator(PassManagerWrapper &PMW, bool) override {}
```

**别的后端最复杂的那一块，它直接空着。**

### 薄处二：MC 层没有二进制编码器

第 05-12 讲讲过：NVPTX 的 MC 层只负责"打印文本"，
没有 `MCCodeEmitter`（因为 PTX 没有机器码）。

```bash
ls /root/llvm-project/llvm/lib/Target/NVPTX/MCTargetDesc/
```

```
NVPTXBaseInfo.h        NVPTXInstPrinter.cpp/h     NVPTXMCAsmInfo.cpp/h
NVPTXMCTargetDesc.cpp/h  NVPTXTargetStreamer.cpp/h
```

对照 x86 的 `MCTargetDesc`，你会看到 x86 有一堆 `*MCCodeEmitter.cpp`、
`*AsmBackend.cpp`、`*ObjectWriter.cpp`——那些 NVPTX 全都不需要。

## 三、"厚"在哪儿：IR 层的 pass 特别多

反过来，NVPTX 在 **IR 层**做了大量工作。
把后端流水线打印出来（`-print-after-all`，共 165 项），NVPTX 专属的有这些：

```
Assign valid PTX names to globals          给全局变量起合法的 PTX 名字
Ensure that the global variables are in the global address space
NVPTX Mark Kernel Pointers Global          标记 kernel 指针为 global
Promote alignment of parameters and return values
Lower pointer arguments of CUDA kernels    处理 byval / param 空间
convert address space of alloca'ed memory to local
NVPTX lower atomics of local memory
NVPTX specific alloca hoisting
NVPTX Tag Invariant Loads
NVPTX IR Peephole
```

**为什么这么多？** 因为 PTX 和 LLVM IR 的抽象层次很接近，
所以"后端该做的事"很多可以在 IR 层就干完：

```
地址空间标注     →  IR 层就能做（插 addrspacecast，然后靠推断传播）
PTX 名字规范化   →  IR 层就能做（改全局变量的名字）
alloca 提升      →  IR 层就能做
```

**这是 NVPTX 后端一个非常值得学习的设计选择：**

> **能在 IR 层做的事，就不要等到机器层。** 因为 IR 层有全套成熟的中端优化可用
> （GVN、InstCombine、SROA……），而到了机器层，可用的优化手段就少多了。

## 四、把两边的账算在一起

```
普通后端（比如 x86）：
  IR 层活少 ── 机器层活多（ISel + 分配 + 调度 + 编码，全是硬骨头）

NVPTX：
  IR 层活多 ── 机器层活少（ISel 之后基本不分配、不调度、不编码）
```

**这不是"简单"和"复杂"的区别，是"活放在哪一层"的区别。**

而"活放在哪一层"的根本原因，就是那条：

> **PTX 是虚拟 ISA，后面还有一个 ptxas。**

## 五、换个角度：谁在什么时候接手

把整条链路按"谁负责"重新画一遍：

```
  .cu  ──clang──►  LLVM IR  ──中端──►  优化后 IR
   │                                        │
   │  前端：语言语义                        │  中端：通用优化
   │                                        ▼
   │                          ┌──────────────────────────────┐
   │                          │   NVPTX 后端（LLVM 里的这一半）│
   │                          │  · IR 层：地址空间/参数/名字    │
   │                          │  · ISel：DAG → 机器指令        │
   │                          │  · 机器层：PHI 消除/合并/调度前置│
   │                          │  · 发射：PTX 文本              │
   │                          └──────────────┬───────────────┘
   │                                         ▼
   │                                      PTX 文本
   │                                         │
   │                          ┌──────────────▼───────────────┐
   │                          │   ptxas（NVIDIA 的汇编器）      │
   │                          │  · 寄存器分配                  │
   │                          │  · 指令调度（含软件流水）        │
   │                          │  · 编码成 SASS                 │
   │                          └──────────────┬───────────────┘
   │                                         ▼
   └─────────────────────────────────────  SASS / cubin
```

**这张图是整门课最重要的一张"分工图"。** 第 7、8 部分就是在这张图的右半边往下走。

## 六、一个具体的例子：同一条 load，两边的分工

用我们案例里的那条 A fragment 加载，看它两边各做了什么：

| 步骤 | 谁做 | 产物 |
| --- | --- | --- |
| 分析 `readonly` 属性，决定用 `ld.global.nc` | LLVM（NVPTX 后端） | `LD_GLOBAL_NC_i32`（MIR） |
| 把它印成 PTX | LLVM（AsmPrinter） | `ld.global.nc.b32 %r22, [%rd38+-16];` |
| 分配物理寄存器（`%r22` → `R13`） | **ptxas** | SASS 里的 `R13` |
| 排进流水、避开 bank 冲突 | **ptxas** | SASS 里的指令顺序 |
| 编码成 128 位机器码 | **ptxas** | `/*03c0*/` 那串十六进制 |

**"LLVM 决定语义，ptxas 决定实现"**——这就是这条链的分界线。

## 七、小结

1. NVPTX 后端的**零件清单和第 5 部分那张表完全一致**——这就是 LLVM 后端框架的力量。
2. 它"薄"在两处：**不做寄存器分配**、**MC 层没有二进制编码器**；
   原因都是同一个：PTX 是虚拟 ISA，后面还有 ptxas。
3. 它"厚"在 IR 层：地址空间标注、参数 lowering、名字规范化都在 IR 层完成。
   **能放到 IR 层做的活就放 IR 层做**，因为那里有成熟的中端优化可用。

## 八、动手题

1. 对照第 05-1 讲那张零件清单，把 NVPTX 目录里的文件一个一个找出来
   （有些零件可能没有独立文件，比如 `TargetLowering` 可能在 `NVPTXISelLowering.cpp` 里）。
2. 想想：如果 NVIDIA 让 LLVM 直接生成 SASS（绕过 PTX），
   NVPTX 后端需要补上哪些东西？（提示：第 05-10、05-11、05-12 讲里那些"空着的"零件。）

下一讲我们看 NVPTX 在 IR 层做的最重要的一件事：**参数和地址空间的 lowering**。
