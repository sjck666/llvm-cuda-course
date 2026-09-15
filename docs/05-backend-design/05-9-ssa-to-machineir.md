# 05-9 · 从 SSA 到 MachineIR：PHI 消除、Two-Address、虚拟寄存器

指令选择之后，我们手里有了一份 MachineIR（`dumps/09-mir-after-isel.mir`）。
这一讲看它**在变成最终机器码之前，还要经历哪些"形式转换"**。

这些转换听起来琐碎，但它们是"SSA 世界"和"寄存器世界"之间的桥。

## 一、先认识 MachineIR 的四个概念

| 概念 | 对应 IR 里的什么 | 说明 |
| --- | --- | --- |
| `MachineFunction` | `Function` | 一个函数 |
| `MachineBasicBlock` | `BasicBlock` | 一个基本块 |
| `MachineInstr` | `Instruction` | 一条**机器指令**（opcode 是机器 opcode） |
| 虚拟寄存器 | SSA 值 | 数量和 SSA 值差不多，但**不再是"只赋值一次"** |

**关键差别在最后一行**：

```
SSA 值：每个值只被赋值一次
虚拟寄存器：可以被赋值多次（它更接近"变量"）
```

为什么？因为底层硬件就是这样：一个物理寄存器会被反复写入。
所以从 SSA 到寄存器世界，第一步就是**放弃"单赋值"这个约束**。

## 二、MIR 里的操作数：和 IR 完全不同的风格

看我们的一条 MIR（`dumps/26-isel.mir:482`）：

```
%45:b32 = LD_i32 0, 0, 101, 3, 32, -1, <mcsymbol mma_tc_manual_param_5>, 0, 0, $noreg
          :: (dereferenceable invariant load (s32), addrspace 101)
```

和 IR 相比，几个明显的差别：

1. **`%45:b32`**：虚拟寄存器带**寄存器类**（`b32`），IR 里只有类型（`i32`）。
2. **操作数是一串数字和符号**：`0, 0, 101, 3, 32, -1, <mcsymbol ...>, 0, 0, $noreg` ——
   这是 `.td` 里 `(ins ...)` 那张表按顺序展开的结果（第 05-8 讲解释过怎么读）。
3. **`:: (...)` 后面的注释**：机器指令的**内存操作数信息**（这里是"从 addrspace 101
   读一个 s32"）。
4. **`$noreg`**：表示"没有寄存器"，是个占位符。

所以读 MIR 的方法和读 IR 完全不同：

> **IR 的语义写在指令名和类型上；MIR 的语义一半写在 opcode 上、
> 一半写在"按位置排列的操作数"上。** 要知道每个位置是什么，
> 必须去查 `.td` 里那条指令的 `(ins ...)`。

## 三、PHI 消除：SSA 概念在寄存器世界里必须消失

PHI 是 SSA 的产物（第 03-1 讲），但**硬件没有"PHI"这种东西**。
所以进入寄存器分配之前，PHI 必须被消除。

做法很直观：**在每个前驱块的末尾插入一条 COPY**。

```
原来（SSA）：
  bb1: ... ; br bb3
  bb2: ... ; br bb3
  bb3: %x = phi [ %a, %bb1 ], [ %b, %bb2 ]     ; 用 %x

消除之后（MIR）：
  bb1: ... ; %x = COPY %a ; br bb3
  bb2: ... ; %x = COPY %b ; br bb3
  bb3: ; 直接用 %x
```

代价是**多了一堆 COPY 指令**——但别担心，下一步（寄存器合并）会把它们优化掉。

我们的实测数据：

```bash
for f in 26-isel 27-regalloc 28-prologepilog; do
  printf "%-16s PHI=%-4s COPY=%s\n" $f \
    "$(grep -c PHI dumps/$f.mir)" "$(grep -c COPY dumps/$f.mir)"
done
```

```
26-isel            PHI=27   COPY=29      ← ISel 刚结束：PHI 还在
27-regalloc        PHI=0    COPY=0       ← PHI 消除 + 寄存器合并之后
28-prologepilog    PHI=0    COPY=13      ← 前导/后记代码插入的 COPY
```

**`PHI=27 → 0`**：27 个 PHI 全部被消除了。

**`COPY=29 → 0`**：合并（coalescing）把它们都吃掉了。

**`COPY=0 → 13`**：最后那 13 条 COPY 是 `NVPTXPrologEpilogPass` 插进来的
（帧指针相关的初始化），**不是 SSA 遗留的**。

## 四、Two-Address 转换：硬件指令的"破坏性"

很多机器的算术指令是"两地址"形式：

```
add r0, r1        # r0 = r0 + r1        （结果写回第一个源）
```

而 IR 里的加法是三地址的：

```llvm
%3 = add i32 %1, %2       ; 结果 %3 和 %1、%2 都不同
```

如果 `%3` 和 `%1` 被分配到不同的寄存器，硬件就没法一条指令完成。
所以有个 pass（Two-Address Instruction Pass）会**插入 COPY 把两者对齐**：

```
%3 = COPY %1
%3 = ADD %3, %2
```

这就是 "two-address" 这个名字的由来。**它和 PHI 消除一样，都是"为了迁就硬件形式"
而引入 COPY**。而寄存器合并（Register Coalescer）的工作就是**把这些 COPY 再消掉**——
消不掉的时候（寄存器冲突），才会真的留下一条 `mov`。

## 五、寄存器合并（Coalescing）：把 COPY 的两个人合成一个

Coalescer 的目标很简单：

```
%a = COPY %b ; ... 只有 %a 和 %b 都在这里被用 ...
            ↓
让 %a 和 %b 用同一个寄存器，然后删掉这条 COPY
```

但"能不能合并"取决于**活跃区间是否冲突**（两个变量同时活着就不能共用一个寄存器）。
所以它需要活跃变量分析（LiveVariables）。

**在 NVPTX 上，合并这一步特别重要**——原因第 05-10 讲会讲：
因为 NVPTX 不做寄存器分配，所以**减少寄存器数量的唯一机会就在合并这一步**。

## 六、机器层优化：SSA 没了，但优化还在继续

PHI 消除之后，机器层还有一批优化 pass 在跑（第 06-1 讲的 165 项清单里能看到）：

```
Machine code sinking         不能把指令下沉到别的基本块（convergent 会挡住，第 04-6 讲）
Machine CSE                  机器指令级的公共子表达式消除
Early Machine LICM           机器层的循环不变量外提
Peephole Optimizations       窥孔优化
NVPTX Address Folder         NVPTX 专属：地址计算折叠
NVPTX IR Peephole / Proxy Reg Erasure / Forward Params ...
```

它们和 IR 层的 pass 是同一类东西（都是"改写指令序列"），只是工作在更低的层次上，
而且**多了硬件约束**（寄存器类、指令副作用、对齐……）。

**这就是"后端优化"和"中端优化"的分界**：

```
中端优化：只看数据流，假设机器是理想的
后端优化：必须在机器约束下工作
```

## 七、小结

1. MachineIR 的四个概念（MachineFunction / MachineBasicBlock / MachineInstr /
   虚拟寄存器）和 IR 一一对应，但**虚拟寄存器不再是"单赋值"的**。
2. **PHI 消除**把 SSA 的 PHI 变成前驱块里的 COPY（我们的实测：27 → 0）；
   **Two-Address 转换**为了迁就两地址指令又插一批 COPY；
   而 **Coalescer 负责把 COPY 再消掉**（实测：29 → 0）。
3. 机器层还有一批优化 pass（MachineSink/CSE/LICM/Peephole 等），
   它们和中端 pass 是同类东西，但**多了硬件约束**。

## 八、动手题

1. 亲自验证那组数字：

   ```bash
   for f in 26-isel 27-regalloc 28-prologepilog; do
     printf "%-16s PHI=%-4s COPY=%s\n" $f \
       "$(grep -c PHI dumps/$f.mir)" "$(grep -c COPY dumps/$f.mir)"
   done
   ```

   然后想想：28 里那 13 条 COPY 是谁插的？（提示：看它们的寄存器名，
   和栈指针/帧指针有关。）
2. 在 `dumps/26-isel.mir` 里找出所有 `PHI`，挑一个说说它对应源码里哪个变量。

下一讲是后端设计的压轴问题之一：**寄存器分配**——以及为什么 NVPTX 干脆不做。
