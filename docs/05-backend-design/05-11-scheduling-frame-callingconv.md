# 05-11 · 调度、帧降低与调用约定

这一讲处理三个"看起来零散、但都直接决定最终代码质量"的话题：

```
1. 指令调度：怎么把指令排得让硬件跑得快
2. 帧降低：栈帧、前导代码、后记代码
3. 调用约定：参数和返回值放在哪儿
```

它们在流水线里挨得很近（都在机器层后半段），而且经常互相牵制。

## 一、指令调度：为什么"顺序"能影响性能

先看一个直观的例子。假设硬件上"加载要 20 个周期才能拿到数据"，
而我们这样写：

```
LDG R1, [addr]      ; 20 个周期后才能用 R1
ADD  R2, R1, R3     ; 立刻就要用 R1 → 只能等
```

后面那 20 个周期里，流水线是空的。如果改成：

```
LDG R1, [addr]
ADD  R4, R5, R6     ; 把无关的活插进来
ADD  R7, R8, R9
...
ADD  R2, R1, R3     ; 到这儿 R1 已经好了
```

就填上了等待时间。**这就是指令调度要干的事：在保持语义的前提下重排指令，
把硬件的等待时间填满。**

### LLVM 里的调度器

| pass | 位置 | 干什么 |
| --- | --- | --- |
| **MachineScheduler**（Pre-RA） | 寄存器分配**之前** | 按"调度模型"重排机器指令 |
| PostRA Scheduler | 寄存器分配之后 | 有的目标才有（NVPTX 里被禁用了） |

第 06-1 讲的 pass 清单里有一项 `Machine Instruction Scheduler`，就是它。
它需要一份"调度模型"（每条指令几个周期、占哪些资源），这份模型也是从
`.td` 里生成的（`SchedMachineModel`、`WriteRes` 之类）。

### 为什么 NVPTX 的调度"看起来不起作用"

因为我们的最终产物是 PTX，而 **PTX 的执行时间是未知的**——
它要被 ptxas 再编译一次。所以：

```
LLVM 侧的调度：影响 PTX 里指令的顺序
ptxas 侧的调度：影响 SASS 里指令的顺序（这才是真正跑在硬件上的顺序）
```

你在第 8 部分会看到 ptxas 干了多少调度工作——它甚至做了**软件流水**
（把 `cp.async` 的发起提前一整轮）。**那是 LLVM 侧完全做不到的**，
因为 LLVM 不知道 GPU 的延迟有多长。

**这是一个很好的一般性结论**：

> **调度发生在"离硬件最近的那一层"。** 谁掌握延迟模型，谁就负责调度。

## 二、帧降低：栈帧、前导代码、后记代码

一个函数如果在执行期间需要栈空间，就得：

```
进入函数时：调整栈指针、保存被调用者保存寄存器（"前导代码" / prologue）
离开函数时：恢复它们、调整回栈指针（"后记代码" / epilogue）
```

负责这件事的类是 **`TargetFrameLowering`**（框架接口在
`llvm/include/llvm/CodeGen/TargetFrameLowering.h`）：

```cpp
  virtual void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const = 0;
  virtual void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const = 0;
  virtual bool hasFPImpl(const MachineFunction &MF) const = 0;
```

`hasFPImpl` 问的是"这个函数要不要帧指针"。

### NVPTX 的特殊情况：它没有栈帧

还记得第 05-2 讲那条注释吗：

```cpp
  // NVPTX does not have any call stack frame, but need a NVPTX specific
  // FrameLowering class because TargetFrameLowering is abstract.
  NVPTXFrameLowering FrameLowering;
```

**NVPTX 没有调用栈帧**——PTX 是虚拟 ISA，寄存器"无限"，参数走 param 空间，
所以不需要压栈保存。但框架要求必须提供一个 FrameLowering 实现，
于是有了一个"几乎空"的 `NVPTXFrameLowering`。

不过它也不是完全没活干：**寄存器溢出到 local memory 的时候，仍然需要一个"栈"的概念**
（NVPTX 用 `.local` 段模拟）。所以 FrameLowering 还是有存在意义的。

### 前导/后记代码由谁插入

通用框架里的 pass 叫 `PrologEpilogInserter`（PEI）。**但 NVPTX 把它禁用了**，
改用自己实现的 `NVPTXPrologEpilogPass`：

```
pass 清单第 86 项：NVPTX Prolog Epilog Pass
```

而这个文件开头的注释把原因写得非常清楚
（`llvm/lib/Target/NVPTX/NVPTXPrologEpilogPass.cpp`）：

```
// This file is a copy of the generic LLVM PrologEpilogInserter pass, modified
// to remove unneeded functionality and to handle virtual registers. Most code
// here is a copy of PrologEpilogInserter.cpp.
```

**"为了处理虚拟寄存器"** ——一句话点中了要害：通用 PEI 假设寄存器已经被分配成物理寄存器了，
而 NVPTX 根本没有物理寄存器（第 05-10 讲）。所以只能抄一份、改一份。

这就是第 01-5 讲那个坑的根源：

```bash
llc ... -stop-after=prologepilog         # 报错：pass is not registered
llc ... -stop-after=nvptx-prolog-epilog  # 正确
```

我们实测过：`dumps/28-prologepilog.mir` 比 `dumps/27-regalloc.mir` 多出 13 条 `COPY`
——那就是这个 pass 插进去的帧相关初始化。

## 三、调用约定：参数放哪儿、返回值放哪儿

调用约定（calling convention）定义了：

```
1. 参数用什么方式传递（寄存器？栈？还是别的空间？）
2. 返回值放在哪儿
3. 谁负责保存寄存器（caller-saved / callee-saved）
4. 栈怎么对齐、怎么清理
```

LLVM 里负责生成"参数搬运代码"的组件叫 **`CallingConvLower`**：
它把"抽象的参数列表"翻译成"具体的寄存器/栈位置"。

### NVPTX 的调用约定非常特别

对 **kernel**（`ptx_kernel` 那些）来说：

```
参数不在寄存器里，而在 param 空间（addrspace 101）
访问方式：LD_i32 / LD_i64 从参数符号里读（我们在 MIR 里见过）
返回值：写进 func_retval0 这个 .param 变量
```

证据就在我们的 MIR 里：

```
%36:b32 = LD_i32 0, 0, 101, 3, 32, -1, <mcsymbol mma_tc_ldmatrix_param_5>, 0, 0, $noreg
```

`101` 就是 param 空间的编号，`<mcsymbol ..._param_5>` 就是第 6 个参数的符号。

而普通 `__device__` 函数（`.func`）走的是另一套约定，返回值放在
`.param .b32 func_retval0` 里：

```
.visible .func  (.param .b32 func_retval0) arch_id()
{
	st.param.b32 	[func_retval0], 890;
	ret;
}
```

**这就是为什么我们在第 02-2 讲反复强调 `ptx_kernel` 这个 calling convention**：
它不只是个标记，它决定了参数和返回值走哪套机制。

## 四、三件事的相互牵制

这三个话题经常互相影响，举两个例子：

**例子一：调度和寄存器压力。**

调度想把指令排得"填满等待"，但排得越开，同时活着的值就越多——
寄存器压力就越大。所以调度器要在"性能"和"寄存器压力"之间找平衡。

NVPTX 的代码里甚至有一条注释专门提到这点（`NVPTXCodeGenPassBuilder.cpp`）：

```cpp
  if (getOptLevel() == CodeGenOptLevel::Aggressive)
    // Disable scalar PRE due to Register Pressure increase
    addFunctionPass(GVNPass(GVNOptions().setScalarPRE(false)), PMW);
```

**"因为寄存器压力上升，所以关掉标量 PRE"**——这是编译器设计里非常真实的一类取舍。

**例子二：调用约定和帧降低。**

参数放寄存器还是放栈，直接决定前导/后记代码要做多少事；
被调用者保存寄存器越多，前导代码就越长。

## 五、小结

1. **指令调度**要"填满硬件等待"，需要调度模型；**而调度总是发生在离硬件最近的那一层**——
   NVPTX 把这件事连同寄存器分配一起交给了 ptxas。
2. **帧降低**由 `TargetFrameLowering` 负责（前导/后记代码）。
   NVPTX 没有栈帧却必须提供这个类；而且它**禁用了通用的 PrologEpilogInserter**，
   改用自己的 `NVPTXPrologEpilogPass`——这就是 `-stop-after` 要换名字的原因。
3. **调用约定**决定参数和返回值的物理位置；NVPTX 的 kernel 参数走 param 空间
   （`LD_*` + `<mcsymbol ..._param_N>`），返回值写 `func_retval0`。

## 六、动手题

1. 找出 NVPTX 的前导/后记 pass 在源码里的位置和注释，看看它为什么自己实现：

   ```bash
   sed -n '1,40p' /root/llvm-project/llvm/lib/Target/NVPTX/NVPTXPrologEpilogPass.cpp
   ```
2. 在 `dumps/28-prologepilog.mir` 里找出那 13 条 COPY，看看它们涉及哪些寄存器
   （提示：和 `%SP`、帧相关）。
3. 读一读 `NVPTXPrologEpilogPass.cpp` 的开头注释，再说说：
   如果 NVPTX 直接沿用通用的 `PrologEpilogInserter`，会出什么问题？

下一讲是第 5 部分的最后一讲：**MC 层与 AsmPrinter——从 MachineInstr 到汇编文本。**
