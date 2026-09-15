# 第 5 部分 · LLVM 后端设计：一套后端是怎么搭起来的

这是整套课里篇幅最大的一部分，也是"通用知识"最多的一部分。

前面四部分我们一直在讲**这一个案例**：一个 Tensor Core kernel 的编译链路。
但从这一部分开始，我们要回答一个更大的问题：

> **LLVM 的后端到底是怎么搭起来的？**
> 一个新目标（比如某家新硬件厂的自研加速器）要从零支持 LLVM，需要提供哪些东西？

为什么值得花 12 讲？因为：

1. 你读 NVPTX、AMDGPU、RISC-V 任何一个后端，看到的都是同一套骨架；
2. 你调优时遇到的很多"为什么编译器这么干"，答案在骨架里而不在具体后端里；
3. 如果你想给新硬件加支持、或者给 MLIR 写 lowering，你必须懂这套骨架。

我们仍然拿 NVPTX 当活标本——**每讲一个抽象概念，就立刻在 NVPTX 里找到它的实体**。

## 本部分的十二讲

| 讲 | 标题 | 核心问题 |
| --- | --- | --- |
| [05-1](05-1-what-backend-does.md) | 后端到底负责什么 | 从 IR 到机器码，中间要做哪几件事 |
| [05-2](05-2-targetmachine-subtarget.md) | TargetMachine / Subtarget / TargetOptions | 后端的"总装配图" |
| [05-3](05-3-tablegen-registers-instrs.md) | TableGen（上）：寄存器与指令 | `.td` 是怎么变成 C++ 的 |
| [05-4](05-4-tablegen-patterns-features.md) | TableGen（下）：pattern 与 Feature | 指令选择表、谓词、SubtargetFeature |
| [05-5](05-5-pass-pipeline.md) | 后端 Pass Pipeline | CodeGenPassBuilder、扩展点、`-stop-after` |
| [05-6](05-6-selectiondag-build.md) | SelectionDAG 的构建 | IR 怎么变成 DAG |
| [05-7](05-7-legalize-combine.md) | Legalize 与 DAG Combine | 类型合法化、操作合法化 |
| [05-8](05-8-instruction-selection.md) | 指令选择 | pattern 匹配与 ISel 结果 |
| [05-9](05-9-ssa-to-machineir.md) | 从 SSA 到 MachineIR | PHI 消除、Two-Address、虚拟寄存器 |
| [05-10](05-10-register-allocation.md) | 寄存器分配 | Greedy 的原理，与 NVPTX 的"不分配" |
| [05-11](05-11-scheduling-frame-callingconv.md) | 调度、帧降低与调用约定 | MachineScheduler / FrameLowering / CallingConv |
| [05-12](05-12-mc-and-asmprinter.md) | MC 层与 AsmPrinter | 从 MachineInstr 到汇编/目标文件 |

## 读这一部分需要什么

需要你能读懂 C++ 类之间的关系（继承、虚函数）、能看懂 TableGen 的语法（我们会讲）、
以及**已经读过第 3、4 部分**（IR 和中端）。

不需要你事先懂编译原理，也不需要你写过编译器。**每一讲的概念都是当场讲清楚的。**
