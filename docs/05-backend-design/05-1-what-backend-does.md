# 05-1 · 后端到底负责什么

这一讲不碰具体代码，先把"后端"这个词讲清楚。
**如果你只记得这门课第 5 部分的一句话，我希望是这一句：**

> **后端的工作不是"翻译"，是"落地"——把一份不关心机器的 IR，
> 变成一份必须尊重机器所有约束的指令序列。**

这中间的差别，就是这一讲要讲的。

## 一、先划清边界：前端 / 中端 / 后端

LLVM 的三段划分非常清楚：

| 段 | 输入 | 输出 | 关心什么 |
| --- | --- | --- | --- |
| 前端 | 源码（C/C++/CUDA/…） | LLVM IR | 语言语义 |
| 中端 | LLVM IR | LLVM IR | 通用优化（不关心机器） |
| **后端** | LLVM IR | 机器指令（汇编或目标文件） | **机器约束** |

关键差别在最后一列。中端做优化时的假设是"这台机器有无限寄存器、所有指令一样快、
内存访问都能随便重排"；而**后端必须把所有这些假设一条一条打掉**。

## 二、后端要处理的三类问题

我把后端的活分成三类。你读任何一个后端，都能按这三类去找代码。

### 第一类：抽象层次下降（Lowering）

IR 里有大量"高层概念"，硬件上没有直接对应。典型的：

```
IR 里的 aggregate（结构体/数组）           → 拆成多个标量或内存操作
IR 里的 i128 / i256                        → 拆成多条 64 位指令（或调用库函数）
IR 里的 <4 x i32> 向量                     → 拆成几个标量，或用 SIMD 指令（如果硬件有）
IR 里的 alloca                             → 栈帧分配，或者干脆提升成寄存器
IR 里的函数调用                             → 具体的调用约定（参数放哪儿、返回值放哪儿）
IR 里的 switch                             → 跳转表或比较链
```

对 NVPTX 来说，还有一类特别的 lowering：

```
地址空间：generic 指针 → global / shared（第 06-2 讲）
kernel 参数：从 param 空间读出来
```

### 第二类：机器约束（Constraints）

硬件不是"想怎么用就怎么用"的。后端要处理：

```
寄存器数量有限          → 寄存器分配（不够就 spill 到栈上）
指令有延迟              → 指令调度（把有依赖的指令排开）
某些指令必须成对出现     → 伪指令扩展（比如 64 位运算在 32 位机器上）
某些寄存器有别名         → 寄存器类与子/超寄存器关系
访存必须对齐            → 对齐检查与拆分
分支距离有限制           → 跳转范围扩展（branch relaxation）
调用约定               → 参数/返回值的物理位置
```

**这一类是"后端之所以难"的根本原因。** 中端可以随口说"把这两个值换一下"，
后端得先问：换过去会不会超出寄存器数量？会不会破坏指令的依赖？

### 第三类：表示载体（Representation）

后端内部用什么数据结构来表示"正在生成的代码"？LLVM 的答案是三层：

```
SelectionDAG        DAG 形式，适合做模式匹配与合法化
   ↓
MachineIR (MIR)     类汇编的线性形式，带基本块、虚拟寄存器
   ↓
MC 层               更底层的表示，能直接发射成汇编文本或目标文件
```

**注意这三层不是"三选一"，而是同一条流水线上的三个阶段。**
我们的 `dumps/` 里正好能看到前两层的快照：

```
dumps/09-mir-after-isel.mir      <- MIR：指令选择之后
dumps/10-mir-after-regalloc.mir  <- MIR：编码之前
dumps/03-clang-O2.ptx            <- 最终发射出来的汇编（对 NVPTX 来说就是 PTX）
```

## 三、一条指令的旅程：从 IR 到 SASS

抽象地讲不如跟着一条真实指令走一遍。我们挑最简单的：

**源码里的一次 A fragment 加载** → 在 IR 里是 `%68 = load i32, ptr %67, align 4`。

| 阶段 | 形态 | 谁负责 |
| --- | --- | --- |
| IR | `%68 = load i32, ptr %67, align 4, !tbaa !13` | clang + 中端 |
| 地址空间推断后的 IR | 指针变成 `addrspace(1)`（因为参数被标记成 global） | NVPTX 的 IR 层 pass |
| SelectionDAG | 一个 `ISD::LOAD` 节点 + 一个 `GlobalAddress`/`CopyFromReg` 节点 | `SelectionDAGBuilder` |
| DAG 合法化后 | 仍然是 load（因为这是合法操作），但类型/寻址被规范化 | `LegalizeTypes` / `LegalizeDAG` |
| 指令选择后（MIR） | `%82:b32 = LD_GLOBAL_NC_i32 3, 32, -1, %28, -16, 0, $noreg` | TableGen 生成的匹配器 |
| 寄存器分配 | （NVPTX 不分配，虚拟寄存器原样保留） | 见 05-10 |
| PTX | `ld.global.nc.b32 %r22, [%rd38+-16];` | `NVPTXAsmPrinter` |
| SASS | `LDG.E.CONSTANT R13, [R32.64+-0x10] ;` | **ptxas**（已经出了 LLVM 的世界） |

请特别看最后两行：

> **LLVM 对 NVPTX 的"最终产物"是 PTX 文本。把它变成 SASS 的是 ptxas，不是 LLVM。**

这就是为什么 NVPTX 是个"特殊"的后端（第 06-1 讲会展开）：
它把一个虚拟 ISA 当作目标，于是**很多别的后端必须做的事（寄存器分配、指令调度的一部分）
它可以推给下游**。

## 四、一个后端要提供哪些"零件"

如果把后端比作一台机器，那么每个目标（target）都要提供下面这些零件。
这张表你要记住，因为它就是第 5 部分后半段的目录：

| 零件 | 作用 | NVPTX 里的实体 |
| --- | --- | --- |
| `TargetMachine` | 后端的入口对象 | `NVPTXTargetMachine` |
| `TargetSubtargetInfo` | 某个具体 CPU/特性组合的信息 | `NVPTXSubtarget` |
| `TargetLowering` | IR/DAG 怎么降级到这台机器 | `NVPTXTargetLowering` |
| `TargetInstrInfo` | 指令相关查询（复制、比较、展开…） | `NVPTXInstrInfo` |
| `TargetRegisterInfo` | 寄存器类、别名、预留寄存器 | `NVPTXRegisterInfo` |
| `TargetFrameLowering` | 栈帧、前导/后记代码 | `NVPTXFrameLowering` |
| 指令描述（`.td`） | 指令、寄存器、pattern | `NVPTXInstrInfo.td` 等 |
| 指令选择器 | IR/DAG → 机器指令 | TableGen 生成的 `NVPTXGenDAGISel.inc` |
| `AsmPrinter` | 机器指令 → 汇编文本 | `NVPTXAsmPrinter` |
| MC 层（可选） | 汇编/目标文件编码 | NVPTX 直接发文本，比较薄 |
| `TargetPassConfig` | 拼装后端的 pass 流水线 | `NVPTXPassConfig` |

**任何一个 LLVM 后端，你都能在它的目录下找到这些文件。**
比如你去 RISC-V 的目录看，文件名会是 `RISCVTargetMachine.cpp`、`RISCVSubtarget.h`、
`RISCVISelLowering.cpp`……一一对应。

## 五、指令选择的三条技术路线

LLVM 历史上有三套指令选择器，它们至今共存：

| 路线 | 代码 | 特点 | 谁在用 |
| --- | --- | --- | --- |
| **SelectionDAG** | `lib/CodeGen/SelectionDAG/` | 最成熟，DAG 形式，TableGen 驱动 | 几乎所有后端（包括 NVPTX） |
| **GlobalISel** | `lib/CodeGen/GlobalISel/` | 在 MIR 上做，支持"分阶段降低"，适合多阶段目标 | AArch64、AMDGPU 等（逐步铺开） |
| **FastISel** | `lib/CodeGen/SelectionDAG/FastISel.cpp` | 只在 `-O0` 用，直接翻 IR，快但结果差 | 各后端可选 |

我们这门课讲的 NVPTX 走的是**第一条路线（SelectionDAG）**，所以第 05-6 ~ 05-8
三讲都在讲它。但你要知道另外两条存在，否则读代码时会困惑：
为什么同一个后端目录里既有 `*ISelLowering.cpp` 又有 `*InstructionSelector.cpp`？
——因为后者是 GlobalISel 的。

## 六、后端的输入输出契约

最后用一个"契约"的视角总结。后端拿到的是：

```
一个 llvm::Module（IR）
  + 一个 TargetMachine（描述"为谁编译"）
  + 优化级别（CodeGenOptLevel）
  + 目标选项（TargetOptions，比如是否开 unwind table）
```

它要产出的是：

```
对一个函数：一段机器指令序列（以 MIR 形式存在，最终发射成文本或二进制）
对整个模块：全局变量的布局、常量池、调试信息……
```

而**"怎么从输入走到输出"，就是由那个 TargetMachine 提供的零件拼出来的流水线决定的**。
这就是第 05-2 讲要讲的内容。

## 七、小结

1. 后端做的不是"翻译"，是"**落地**"——把不关心机器的 IR 变成必须尊重机器约束的指令。
2. 后端的活分三类：**抽象层次下降**、**机器约束**、**表示载体**；
   表示载体又分三层：SelectionDAG → MachineIR → MC。
3. 每个后端都要提供同一套零件（TargetMachine / Subtarget / Lowering /
   InstrInfo / RegisterInfo / FrameLowering / AsmPrinter / PassConfig）。
   后面十一讲就是把这套零件一个一个拆开。

## 八、动手题

1. 用下面的命令看看 NVPTX 后端都提供了哪些文件，把它们和自己心里那份"零件清单"对一下：

   ```bash
   ls /root/llvm-project/llvm/lib/Target/NVPTX/
   ```

   提示：先找 `NVPTXTargetMachine.*`、`NVPTXSubtarget.*`、`NVPTXISelLowering.*`、
   `NVPTXInstrInfo.*`、`NVPTXRegisterInfo.*`、`NVPTXFrameLowering.*`、`NVPTXAsmPrinter.*`。
2. 想一想：如果我们用的是 AArch64，"PTX 那一层"对应的是什么？
   为什么 NVPTX 的后端可以比 AArch64 的后端"薄"？

下一讲我们看后端的总装配图：TargetMachine、Subtarget 和 TargetOptions。
