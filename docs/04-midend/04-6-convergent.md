# 04-6 · convergent 的约束：它一路传到了机器层

第 02-6 讲我们弄清了 `convergent` 是怎么被标上去的。这一讲看它的后果——
**它怎么一路传到机器层，让一整族优化收手。**

## 一、先回忆：谁被标了 convergent

```llvm
attributes #7 = { convergent nounwind }        ; mma 那条内联汇编
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3   ; barrier
```

**不只是 barrier，连一条 `mov.b32` 打包汇编都被标了**（第 02-7 讲）。
因为 clang 的规则是"CUDA 设备端所有内联汇编一律标 convergent"。

## 二、标记怎么从 IR 传到机器指令

这是第一段链路。`llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp:10210`：

```cpp
    if (Call.isConvergent())
      Flags |= InlineAsm::Extra_IsConvergent;
```

于是 MIR 里那条 `INLINEASM` 长这样（`dumps/26-isel.mir`）：

```
INLINEASM &"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};\0A",
          sideeffect isconvergent attdialect,
          regdef:B32, def %78, ..., !18
```

两个标志：`sideeffect`（来自 `asm volatile`）+ **`isconvergent`**。

（老版本这里打印的是数字 `33`，新版直接打名字，读起来省事多了。
这是 MIR 打印格式的一个改进，你在老资料里看到 `33 /* sideeffect isconvergent */`
不用惊讶。）

## 三、后端里"看见 convergent 就收手"的 pass

`MachineInstr::isConvergent()`
（`llvm/include/llvm/CodeGen/MachineInstr.h:1082`）就是读上面那个标志位：

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

同一份文件里还有另外两处同样的检查（第 752、1857 行），分别属于不同的下沉路径。

**"不能被 MachineSink"这句话的完整含义是：**
不能让这条指令变成"只在某个条件成立时才执行"。因为 `mma.sync` 是 warp 级操作——
32 个线程必须一起到达这条指令；如果它被下沉到一个只有部分线程会走的路径里，
硬件行为就未定义了（轻则结果错，重则死锁）。

## 四、受影响的 pass 远不止 MachineSink

在 LLVM 源码里搜一下谁检查 `isConvergent()`，你会看到这一族人：

```
lib/CodeGen/MachineSink.cpp            <- 不能下沉到别的分支
lib/CodeGen/MachineLICM.cpp            <- 不能提升出循环
lib/CodeGen/MachineCSE.cpp             <- 不能做公共子表达式合并
lib/CodeGen/IfConversion.cpp           <- 不能做 if-conversion（谓词化）
lib/CodeGen/TailDuplicator.cpp         <- 不能复制到尾部
lib/CodeGen/MachineConvergenceVerifier.cpp  <- 专门验证收敛性
lib/CodeGen/GlobalISel/InlineAsmLowering.cpp
```

**一条 `mma` 汇编进来，等于给一整族机器级优化按了暂停键。**

这解释了一个你大概遇到过的现象：

> **把某个操作改写成内联汇编之后，周围代码的优化质量会突然下降。**

不是编译器变笨了，是它必须保守。

## 五、能不能把这个标记摘掉（回顾）

第 02-6 讲我们实测了四条路，这里再汇总一次（脚本 `tools/convergent-experiment.sh` G 段）：

| 做法 | 结果 |
| --- | --- |
| `-fno-convergent-functions` | **有效**（整个设备端 IR 里不再有 convergent） |
| `__attribute__((noconvergent))` | 语法错误：`an attribute list cannot appear here` |
| `[[clang::noconvergent]]` 直接放 asm 前 | 被忽略（`-Wignored-attributes`），无效 |
| `[[clang::noconvergent]] { asm ...; }` | **有效**（属性挂到 `AttributedStmt` 上） |

实测输出：

```
  with_convergent          属性组 #3  含 convergent: 是
  with_noconvergent        属性组 #3  含 convergent: 是     <- 直接挂，无效
  with_noconvergent_stmt   属性组 #4  含 convergent: 否     <- 挂语句块，有效
```

### 那什么时候该摘

我的建议：**默认不要摘，只有确定是纯数据操作时才逐条放行。**

判断标准很简单：**这条汇编会不会让整个 warp 一起做某件事？**

```
bar.sync / shfl / mma / ldmatrix / cp.async   → 会（warp 级或半收敛）→ 保持 convergent
mov.b32 打包、纯算术                        → 不会            → 可以考虑摘
```

全局的 `-fno-convergent-functions` 更要小心：它会把 `bar.sync` 的保护一起关掉。

## 六、还有一条"暴力"路径：直接改 IR

我们做实验时经常直接改 IR 把 `convergent` 抹掉，再喂给 `llc`
（第 04-5 讲那个 A/B 对照就是这么做的，产物是 `dumps/21-kconst-noconvergent-ir.ll`）。

**但请不要在生产代码里这么干**——除非你 100% 确定那段汇编里没有 warp 级同步语义。

## 七、小结

1. `convergent` 从 IR 的属性组 → MIR 的 `isconvergent` 标志位，
   一路传到机器层；`MachineInstr::isConvergent()` 就是读它。
2. 它让 **MachineSink / MachineLICM / MachineCSE / IfConversion / TailDuplicator
   等一整族 pass 收手**。这是"mma 挪不动"的完整理由。
3. 能摘掉它的只有两条路（全局 `-fno-convergent-functions`、
   或给某条 asm 包一层 `[[clang::noconvergent]] { ... }`）；
   **默认建议别摘**，除非那条汇编确实是纯数据操作。

到这里第 4 部分结束。我们现在知道了：

```
哪些 pass 在跑（04-1）
数组是怎么消失的（04-2）
表达式是怎么被改写的（04-3）
CUDA 特有的事谁在做（04-4）
循环为什么不展开（04-5）
convergent 挡住了什么（04-6）
```

接下来，我们要进入这套课里篇幅最大的部分：**第 5 部分 · LLVM 后端设计**。
前面所有关于"后端"的说法（"这个 pass 是目标机插进来的"、"指令选择怎么回事"、
"为什么 NVPTX 不做寄存器分配"）都会在那里被系统性地讲清楚。

## 八、动手题

1. 在 `dumps/26-isel.mir` 里数一数带 `isconvergent` 的 `INLINEASM` 有几条，
   再说说它们分别对应源码里的哪几条汇编。
2. 把 `code/noconvergent_demo.cu` 里那个用语句块放行的 kernel 编译出来，
   对照它的 IR 属性组，确认 `convergent` 只从**那一条** asm 上消失了，
   而 kernel 函数本身还是 convergent。
