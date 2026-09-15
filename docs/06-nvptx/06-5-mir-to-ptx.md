# 06-5 · 从 MIR 到 PTX：AsmPrinter 做了什么

第 6 部分最后一讲。我们要把 MIR 变成 PTX 文本，看看 `NVPTXAsmPrinter` 到底做了哪几件事。

表面上看这一步"就是打印"，但 PTX 有几样别的 ISA 没有的东西，
所以这一步比你想的要有内容。

## 一、四件事

`NVPTXAsmPrinter` 在发射一个函数时，主要做四件事：

```
1. 给虚拟寄存器按类重新编号，发射 .reg 声明
2. 发射函数头（.visible .entry / .func）和参数声明
3. 发射共享内存等全局对象的声明（.shared ...）
4. 逐条发射指令（大部分由 TableGen 生成的 AsmWriter 完成）
```

我们一件一件看。

## 二、第一件：虚拟寄存器编号（per-class renumbering）

PTX 里的寄存器名字长这样：`%r1`、`%rd2`、`%rs3`、`%f4`。
它们**不是** MIR 里那些虚拟寄存器编号（`%125`、`%126`…）。

中间的换算在 `NVPTXAsmPrinter::setAndEmitFunctionVirtualRegisters()`
（`NVPTXAsmPrinter.cpp:2057`）里：

```cpp
  // Go through all virtual registers to establish the mapping between the
  // global virtual register number and the per class virtual register number.
  // We use the per class virtual register number in the ptx output.
  for (unsigned I : llvm::seq(MRI->getNumVirtRegs())) {
    Register VR = Register::index2VirtReg(I);
    if (MRI->use_empty(VR) && MRI->def_empty(VR))
      continue;
    auto &RCRegMap = VRegMapping[MRI->getRegClass(VR)];
    RCRegMap[VR] = RCRegMap.size() + 1;      // ← 每个类各自从 1 开始编号
  }

  // Emit declaration of the virtual registers or 'physical' registers for
  // each register class
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (const TargetRegisterClass &RC : TRI->regclasses()) {
    const auto It = VRegMapping.find(&RC);
    if (It == VRegMapping.end() || It->second.empty())
      continue;

    TS->emitRegDirective(
        TRI->getRegSizeInBits(RC).getFixedValue(),
        NVPTX::getVirtualRegisterPrefix(getVirtualRegisterKind(&RC)),
        It->second.size() + 1);
  }
```

关键在注释那句：**"把全局虚拟寄存器编号映射成 per-class 的编号，从 1 开始"**。

于是：

```
MIR 里的 %125（B32 类） → PTX 里的某个 %rNN（该类里第 N 个）
声明写成 .reg .b32 %r<37>;   ← 表示这个类用到了 %r1..%r36
                                        （<N+1> 是因为编号从 1 开始、0 号留空）
```

**这就是为什么 PTX 里的寄存器编号看起来总是比 MIR 里的小**——
它们换了一套编号空间。

### 一个值得注意的细节：没有 `.reg .f32`

我们这份 PTX 的寄存器声明是：

```
	.reg .pred 	%p<3>;
	.reg .b16 	%rs<5>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<40>;
```

**没有 `.reg .f32`！** 但我们的累加器明明是 `float c[4]`。

原因还是那条：**PTX 的虚拟寄存器按宽度分类**（`B32` 类同时装 `i32` 和 `f32`），
所以浮点累加器和整型打包数据住在同一批 `.b32` 寄存器里。

## 三、第二件：函数头与参数

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

三个可以讲的点：

1. **`.visible .entry` vs `.visible .func`**：前者是 kernel（能被 host 启动），
   后者是普通设备函数。这个区别来自 IR 里的 `ptx_kernel` calling convention（第 02-2 讲）。
2. **`.param .u64 .ptr .align 1`**：新版 PTX 的参数声明会带上 `.ptr` 和 `.align`
   修饰，语义更准确（老版本只有 `.param .u64`）。
3. **`.version 8.7` / `.target sm_89`**：来自 IR 里的函数属性
   `"target-cpu"="sm_89" "target-features"="+ptx87"`。

## 四、第三件：共享内存声明

```
	.shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2As[512];
	.shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2Bs[256];
```

这两行是**从 IR 里那两个 `addrspace(3)` 全局变量翻译过来的**（第 02-3、03-2 讲）：

```
IR:  @_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
PTX: .shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2As[512];
```

字节数是算出来的：`16 × 16 × 2 = 512`、`16 × 8 × 2 = 256`
（`%struct.__half` 是 `{ i16 }`，2 字节）。

**注意它发射成 `.b8` 数组**——因为 PTX 的共享内存声明是按字节数组给的，
而不是按"16×16 的 half 数组"。这类"类型信息丢失"是汇编层很常见的现象。

## 五、第四件：指令（大部分是自动的）

普通指令的发射由 TableGen 生成的 `NVPTXGenAsmWriter.inc` 完成
（第 05-3、05-12 讲看过它长什么样）：

```cpp
    O << "ld.global.nc";
    printOperand(MI, 0, STI, O);
```

而**内联汇编是"原文照抄"**：

```
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r33,%r34,%r35,%r36}, {%r22,%r23,%r24,%r25}, {%r20,%r21}, {%r33,%r34,%r35,%r36};
```

**和我们源码里写的一模一样**，只是 `%0..%13` 被换成了具体寄存器名。

这就是内联汇编的本质：**LLVM 只负责"把操作数放进寄存器、把占位符替换成寄存器名"，
指令本身的语义它完全不理解**——这也正是它只能保守地标 `convergent` 的原因（第 02-6 讲）。

## 六、三层的操作数对照（一个漂亮的验证）

把源码约束、MIR 类别、PTX 寄存器摆在一起：

| 层 | D/C（4 个 f32） | A（4 个 b32） | B（2 个 b32） |
| --- | --- | --- | --- |
| 源码约束 | `"=f"` × 4 | `"r"` × 4 | `"r"` × 2 |
| MIR 类别 | `regdef:B32` × 4 | `reguse:B32` × 4 | `reguse:B32` × 2 |
| PTX 寄存器 | `%r33,%r34,%r35,%r36` | `%r22,%r23,%r24,%r25` | `%r20,%r21` |

**三层严丝合缝。** 这也回答了第 02-6 讲那个问题："约束字符串里的 `f` 和 `r`
到了后端变成什么？"——在 NVPTX 上，它们都落到同一个 32 位寄存器类。

## 七、第 6 部分总结

把五讲串起来：

```
[06-1] NVPTX 的整体结构：零件清单齐全，但"薄一半、厚一半"
[06-2] 参数与地址空间：三棒接力，generic → global 的完整证据链
[06-3] ISel 结果：三大家族 opcode 与命名规律
[06-4] MIR 逐段精读：每行都能对回源码
[06-5] MIR → PTX：寄存器重新编号、函数头、共享内存声明、内联汇编照抄
```

而 NVPTX 后端最核心的两个"设计选择"，现在你应该能说清楚了：

```
选择一：能在 IR 层做的事就放 IR 层做
        （地址空间标注、参数 lowering、名字规范化）
        ——因为 IR 层有成熟的中端优化可用

选择二：能推给 ptxas 的事就推给 ptxas
        （寄存器分配、大部分调度、最终编码）
        ——因为 PTX 是虚拟 ISA，下游更懂硬件
```

## 八、动手题

1. 验证"per-class 重新编号"这件事：在 MIR 里找一个最大的虚拟寄存器编号，
   在 PTX 里找 `.reg` 声明的上界，比较两者。

   ```bash
   grep -oE "%[0-9]+:b32" dumps/28-prologepilog.mir | tr -d '%:b32' | sort -n | tail -1
   grep -E "\.reg" dumps/03-clang-O2.ptx | head -4
   ```

2. 在 PTX 里找出共享内存声明，算一遍字节数，确认它和 IR 里的类型对得上。

到这里第 6 部分结束。下一部分我们把镜头拉到 PTX 本身：
**`mma.sync`、`ldmatrix`、`cp.async` 逐条讲清楚它们的语义和操作数。**
