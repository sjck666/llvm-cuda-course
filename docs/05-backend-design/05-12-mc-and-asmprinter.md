# 05-12 · MC 层与 AsmPrinter：从 MachineInstr 到汇编文本

第 5 部分最后一讲。我们要把最后一段路走完：

```
MachineInstr（带虚拟寄存器的机器指令）
        ↓  AsmPrinter
汇编文本（对 NVPTX 来说就是 PTX）
```

顺便讲清楚 LLVM 里那个名字有点拗口的 **MC 层**。

## 一、先看问题：为什么要分成"MC 层"和"AsmPrinter"

从 "机器指令" 到 "最终发射物"，其实有好几条路：

```
路线 A：机器指令 → 汇编文本（给外部汇编器用）
路线 B：机器指令 → 目标文件（直接产出 .o）
路线 C：机器指令 → 内存里的机器码（JIT 用）
```

如果这三种都各写一遍，代码会三倍冗余。所以 LLVM 抽了一层：

```
        机器指令（MachineInstr）
              ↓  AsmPrinter：把 MachineInstr 变成"更底层的指令表示"
            MCInst
              ↓  MCCodeEmitter（编码器）
          机器码字节 + 重定位信息
              ↓  MCObjectWriter / MCStreamer
       汇编文本 或 目标文件
```

**`MCInst` 是"比 MachineInstr 更低一层"的指令表示**：
它不再有虚拟寄存器、不再有 SSA 概念，只有"opcode + 操作数列表"。

## 二、MC 层的几个关键组件

| 组件 | 职责 | NVPTX 里对应 |
| --- | --- | --- |
| `MCStreamer` | 发射接口：`emitInstruction`、`emitBytes`、`emitLabel`… | 通用 |
| `MCAsmInfo` | 汇编语法信息：注释符、指令前缀、对齐写法… | `NVPTXMCAsmInfo` |
| `MCInstPrinter` | 把 `MCInst` 打成汇编文本 | `NVPTXInstPrinter` |
| `MCCodeEmitter` | 把 `MCInst` 编码成字节 | **NVPTX 没有**（见下） |
| `MCTargetStreamer` | 目标的特殊指示（`.reg`、`.shared` 之类） | `NVPTXTargetStreamer` |

对应到文件（`llvm/lib/Target/NVPTX/MCTargetDesc/`）：

```
NVPTXMCAsmInfo.cpp       汇编语法信息
NVPTXInstPrinter.cpp     指令打印
NVPTXTargetStreamer.cpp  目标特殊指示
NVPTXMCTargetDesc.cpp    注册这些组件
```

## 三、NVPTX 的特殊之处：它没有二进制编码器

注意上面那张表里这一行：

> `MCCodeEmitter` | 把 `MCInst` 编码成字节 | **NVPTX 没有**

为什么？因为 **PTX 是虚拟 ISA，它没有"机器码"**。
PTX 的"最终产物"就是一段文本，交给 ptxas 去汇编。

所以 NVPTX 的 MC 层是**"只写文本"的一半**：

```
x86 的 MC 层：  MCInst → （文本 或 字节）  两套都有
NVPTX 的 MC 层：MCInst → 文本             只有这一套
```

**这又一次印证了那句总结：NVPTX 后端只负责"到 PTX 为止"。**

## 四、AsmPrinter：真正干活的那个类

`AsmPrinter` 是每目标一个（NVPTX 里是 `NVPTXAsmPrinter`），它负责：

```
遍历 MachineFunction 里的每条指令 → 交给 MC 层发射
处理函数级的"包装"：函数头、参数声明、返回、指令序列的收尾
发射目标特有的指示：段、对齐、符号……
```

它的核心方法一般是 `emitFunctionBody`，大致长这样（示意）：

```cpp
void NVPTXAsmPrinter::emitFunctionBody(Module &M, const Function &F,
                                       MachineFunction *MF) {
  setAndEmitFunctionVirtualRegisters(*MF);   // ← 发射 .reg 声明（第 7 部分细讲）
  ...
  AsmPrinter::emitFunctionBody(...);          // 逐条发射指令
  ...
}
```

**`setAndEmitFunctionVirtualRegisters`** 这个名字你在第 7 部分会再遇到：
它就是"给虚拟寄存器按类重新编号，然后发射 `.reg .b32 %r<37>;` 这种声明"的地方。

### 指令本身是怎么被打印的

逐条打印指令的部分，是 TableGen 生成的（`NVPTXGenAsmWriter.inc`）：

```
O << "ld.global.nc";
printOperand(MI, 0, STI, O);
```

**你在 `.td` 的 `AsmString` 里写的模板，就是被编译成这些 `O << ...` 语句的**
（第 05-3 讲看过这个文件）。

## 五、走一遍完整的路：从 llc 到 PTX

把前面几讲串起来，`llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 in.ll -o out.ptx`
内部发生的事：

```
1. llc 解析参数，通过 TargetRegistry 找到 NVPTX 目标
2. 创建 NVPTXTargetMachine（其中含 NVPTXSubtarget）
3. 调用 TM.addPassesToEmitFile(PM, Out, ..., CodeGenFileType::Assembly)
      → 这条路径去问 CodeGenPassBuilder "流水线怎么排"
4. 流水线跑起来：IR pass → ISel → 机器层 pass → AsmPrinter
5. AsmPrinter 逐函数发射：
      - 函数头（.visible .entry ...）
      - .reg 声明
      - 每条指令（经 MC 层的 InstPrinter）
      - 函数尾
6. 输出到 out.ptx
```

**第 5 步里"发射函数头"这件事是 NVPTX 自己写的**——因为 PTX 的函数语法
（`.entry`/`.func`/`.param`）是它独有的。你在 `NVPTXAsmPrinter.cpp` 里能看到
大量这样的手工拼装代码。

## 六、既然有 MC 层，为什么 NVPTX 的 AsmPrinter 还这么"重"

因为 **PTX 不是普通汇编**。它有：

```
虚拟寄存器声明（.reg .b32 %r<37>;）    ← 别的 ISA 没有
共享内存声明（.shared .align 16 .b8 ...）
参数空间（.param .u64 .ptr .align 1 ...）
特殊指示（.address_size 64、.version 8.7、.target sm_89）
```

这些东西**都不是"一条指令"**，而是"一段声明"，所以没法靠指令打印器生成，
只能由 `NVPTXAsmPrinter` 手工发射。

**这是个很好的对照**：如果你给一个"普通"ISA（比如 RISC-V）写后端，
AsmPrinter 会薄得多；而给 NVPTX 这种"带大量元信息的虚拟 ISA"写后端，
AsmPrinter 就得承担更多。

## 七、第 5 部分总结：一套后端的完整生命周期

把十二讲串成一条线：

```
[05-1] 后端要做三类事：抽象层次下降、机器约束、表示载体

[05-2] Target / TargetMachine / Subtarget   ← 谁来提供信息
[05-3] TableGen：寄存器与指令描述           ← 信息怎么写下来
[05-4] TableGen：pattern 与 Feature         ← 指令什么时候能用
[05-5] Pass Pipeline：钩子机制              ← 流水线怎么拼

[05-6] IR → DAG                             ← 换一种表示
[05-7] Legalize / Combine                   ← 修成机器能吃的形状
[05-8] Instruction Selection                ← DAG → 机器指令
[05-9] SSA → MachineIR                      ← PHI 消除、COPY、合并
[05-10] 寄存器分配（NVPTX：不做）            ← 无限 → 有限
[05-11] 调度 / 帧降低 / 调用约定             ← 硬件约束落地
[05-12] MC 层与 AsmPrinter                  ← 发射成文本或字节
```

而 NVPTX 在这条线上的"特殊性"，可以总结成三句话：

```
1. 目标是一个虚拟 ISA（PTX），所以很多活可以推给下游（ptxas）
2. 因此它不做寄存器分配，调度也只做一半
3. 但它的"元信息"特别多（地址空间、param 空间、虚拟寄存器声明），
   所以 IR 层 pass 和 AsmPrinter 反而更重
```

## 八、小结

1. 从机器指令到最终产物中间有 **MC 层**（`MCInst` + `MCStreamer` + `MCAsmInfo` +
   `MCInstPrinter`），它让"发文本 / 发目标文件 / JIT"三条路复用同一套代码。
2. **NVPTX 没有二进制编码器**——因为 PTX 是虚拟 ISA，产物就是文本。
   这就是"NVPTX 的 MC 层只做一半"的原因。
3. `AsmPrinter`（NVPTX 里是 `NVPTXAsmPrinter`）负责函数级的包装：
   函数头、`.reg` 声明、指令序列、目标特有指示。
   **PTX 的元信息多，所以它的 AsmPrinter 比普通 ISA 重。**

## 九、动手题

1. 看看 NVPTX 的 MC 层有哪些文件，数一数比 x86 少多少：

   ```bash
   ls /root/llvm-project/llvm/lib/Target/NVPTX/MCTargetDesc/
   ls /root/llvm-project/llvm/lib/Target/X86/MCTargetDesc/ | head -20
   ```

2. 找一找 PTX 里那些 `.xx` 指示是在哪儿发射的。先看直接输出的：

   ```bash
   grep -n 'O << "\."' /root/llvm-project/llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp | head
   ```

   只有两处。**那 `.reg .b32 %r<37>;` 和 `.shared ...` 是谁发射的？**
   去找 `TargetStreamer`：

   ```bash
   grep -n "TS->emit\|emitRegDirective\|emitSharedDirective" \
     /root/llvm-project/llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp | head
   ```

   （提示：这类"结构化指示"走的是 `NVPTXTargetStreamer`，
   而不是简单的字符串拼接——因为它们是"声明"而不是"指令"。）

到这里，第 5 部分结束。你现在应该有一套完整的"后端地图"了。

下一部分我们回到 NVPTX：**用这张地图去看它的具体实现**，
把第 5 部分学到的抽象概念一一对上真实代码和真实产物。
