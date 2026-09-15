# 05-8 · 指令选择：把 DAG 匹配成机器指令

这是 DAG 旅途的终点：**把每个 DAG 节点换成一条（或几条）机器指令**。

做完这一步，我们手里就有了 `dumps/09-mir-after-isel.mir` 那份东西。

## 一、输入输出

```
输入：一个合法化之后的 DAG（每个基本块一张）
输出：MachineIR —— 基本块 + 机器指令 + 虚拟寄存器
```

从"图"到"线性序列"的这个转换，就在这一步发生。

## 二、框架：`SelectionDAGISel` + 生成的匹配器

指令选择的工作由两部分组成：

| 部分 | 是什么 | 代码 |
| --- | --- | --- |
| **框架** | 通用流程：遍历 DAG、驱动匹配、处理失败 | `llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp` |
| **匹配表** | TableGen 从 pattern 生成的字节码 | `NVPTXGenDAGISel.inc`（3.2 MB） |

目标自己做两件事：

1. **在 `.td` 里写 pattern**（第 05-4 讲）；
2. **写一个 `XXXDAGToDAGISel` 类**，处理那些 pattern 表达不了的情况。

NVPTX 的这个类在 `NVPTXISelDAGToDAG.cpp:88`：

```cpp
class NVPTXDAGToDAGISel : public SelectionDAGISel {
  ...
  void Select(SDNode *N) override;      // ← 入口：给一个节点，选出指令
  ...
  void SelectTexSurfHandle(SDNode *N);
  void SelectV2I64toI128(SDNode *N);
  void SelectI128toV2I64(SDNode *N);
  void SelectCpAsyncBulkTensorReduceCommon(SDNode *N, unsigned RedOp, ...);
  void SelectTcgen05Ld(SDNode *N, bool hasOffset = false);
  void SelectTcgen05St(SDNode *N, bool hasOffset = false);
```

**那些 `SelectXxx` 就是"pattern 表达不了、必须手写 C++"的部分。**
命名很直白：`SelectV2I64toI128` 就是把两个 i64 合成一个 i128 的选择逻辑。

`Select()` 的典型结构是：

```cpp
void NVPTXDAGToDAGISel::Select(SDNode *N) {
  unsigned Opcode = N->getOpcode();
  if (N->isMachineOpcode()) { ... return; }      // 已经是机器指令了，跳过

  switch (Opcode) {
  case ISD::INTRINSIC_W_CHAIN: { ... 手写处理 ... return; }
  case ISD::SELECT: ...
  default: break;
  }

  SelectCode(N);      // ← 自己处理不了，交给生成的匹配表
}
```

**关键就是最后那句 `SelectCode(N)`**：它进入 TableGen 生成的状态机。

## 三、生成的匹配表长什么样

我们来看看那个 3.2 MB 的文件里都有什么"指令"。它的核心是一个字节码解释器，
指令种类（opcode）有这些：

```bash
grep -oE "OPC_[A-Za-z_0-9]+" /root/llvm-build/lib/Target/NVPTX/NVPTXGenDAGISel.inc \
  | sort | uniq -c | sort -rn | head -8
```

```
   7441 OPC_MoveSibling
   6996 OPC_EmitMergeInputChains1_0
   6682 OPC_CheckInteger
   6412 OPC_MoveParent
   6387 OPC_RecordNode
   4181 OPC_MorphNodeTo0Chain
   3006 OPC_CheckPatternPredicate
   2520 OPC_Scope
```

逐条解释（这就是"pattern 匹配"的真身）：

| 字节码 | 干什么 | 出现的次数说明了什么 |
| --- | --- | --- |
| `OPC_MoveSibling` / `OPC_MoveParent` | 在 DAG 里走到父节点/兄弟节点 | 匹配就是"沿着图走" |
| `OPC_CheckInteger` | 检查某个操作数是不是指定常量 | 大量 pattern 带常量条件（掩码、移位量…） |
| `OPC_RecordNode` | 记住这个节点（待会儿要"变形"它） | 匹配成功后会把它换成机器指令 |
| `OPC_MorphNodeTo0Chain` | **把 DAG 节点变成机器指令** | 这就是"选中"的时刻 |
| `OPC_CheckPatternPredicate` | **检查谓词**（`Requires<...>`） | 3006 处！ |
| `OPC_Scope` | 进入一个候选作用域（匹配失败可回退） | 复杂 pattern 用 |

**`OPC_CheckPatternPredicate` 出现 3006 次**——这正是第 05-4 讲那些
`Requires<[PTX80, SM90]>` 落地的地方。

所以整件事可以这样总结：

> **TableGen 把几千条 pattern 编译成"前缀树 + 字节码"，
> 运行时逐节点匹配；匹配成功就 `MorphNodeTo` 成机器指令。**

## 四、一个真实的选择结果

我们在 MIR 里看到的这一行：

```
%82:b32 = LD_GLOBAL_NC_i32 3, 32, -1, %28, -16, 0, $noreg
```

现在可以完整解释了：

| 部分 | 含义 |
| --- | --- |
| `%82:b32` | 结果：一个 32 位虚拟寄存器（`b32` 类） |
| `LD_GLOBAL_NC_i32` | **opcode 名字来自 `.td` 的 `def` 名字**（第 05-3 讲） |
| `3` | 第一个操作数（`Sign`/符号相关的枚举值） |
| `32` | 位宽（`fromWidth` = 32） |
| `-1` | 有效字节掩码（`usedBytes`，-1 表示全用） |
| `%28` | 地址的基寄存器 |
| `-16` | 立即数偏移 |
| `0` | 驱逐/预取提示 |
| `$noreg` | 缓存策略（没有额外策略） |

**这七个数字就是 `.td` 里 `(ins AtomicCode:$Sign, i32imm:$fromWidth,
UsedBytesMask:$usedBytes, ADDR:$src, EvictionAndPrefetchHint:..., CachePolicy:...)`
那一串操作数的值。**

第 06-4 讲我们还会逐个核对一遍——那时你会看到"`.td` 的操作数表"和
"MIR 里逗号分隔的那串数字"是一一对应的。

## 五、匹配失败会怎样

如果某个 DAG 节点找不到任何 pattern，你会看到这样的报错：

```
LLVM ERROR: Cannot select: t5: i32 = add t2, t3
```

这是**写后端最常见的一类错误**。它的意思是：

> "合法化告诉我这个操作应该由你自己处理（`Custom`），或者我以为它是合法的，
> 但没有一条 pattern 能匹配上它。"

排查思路（这个顺序很实用）：

```
1. 这个操作在 TargetLowering 里被标成了什么？（Legal / Expand / Custom / LibCall）
2. 如果是 Custom，LowerOperation 里处理了吗？
3. 如果是 Legal，.td 里有对应的 pattern 吗？
4. 如果有 pattern，是不是被 Requires<> 的谓词挡住了？（比如 SM 版本不够）
```

第 4 条是最容易被忽略的：**pattern 存在，但谓词不满足**，
表现和"没有 pattern"一模一样。

## 六、怎么观察这一步

理想情况下，你当然想用 `-debug-only=isel` 看匹配过程。但第 00-2 讲说过，
那需要断言构建。我们这台机器上的替代方案是：

```bash
# 1) 看 ISel 的结果（MIR）
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 \
    -stop-after=finalize-isel -o /tmp/after-isel.mir dumps/02-device-O2.ll

# 2) 看 ISel 之前的 IR（喂给后端的形状）
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 \
    -print-before=finalize-isel -o /dev/null dumps/02-device-O2.ll 2>&1 | head -40

# 3) 看每个 pass 之后的变化
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 -print-after-all \
    -o /dev/null dumps/02-device-O2.ll 2>&1 | grep -A5 "ISEL"
```

**`-stop-after=finalize-isel` 是最实用的一条**：它给你"刚选完指令"的快照，
还没有经过机器层优化和寄存器分配，最适合看清楚"哪个 IR 变成了哪条指令"。

## 七、小结

1. 指令选择 = **框架（`SelectionDAGISel`） + 生成的匹配表（`*GenDAGISel.inc`）**；
   目标只需要写 pattern，以及处理 pattern 表达不了的少数情况（`Select()` 里那些
   `SelectXxx` 函数）。
2. 匹配表是一套**字节码状态机**：走图（`MoveSibling`/`MoveParent`）、
   检查（`CheckInteger`/`CheckPatternPredicate`）、命中后变形（`MorphNodeTo`）。
   其中 `CheckPatternPredicate` 有 3006 处——就是 `Requires<>` 的落地。
3. MIR 里那串"逗号分隔的数字"就是 `.td` 里操作数表的取值；
   **opcode 的名字就是 `.td` 里 `def` 的名字**。

## 八、动手题

1. 在生成文件里数一数各种 opcode 的数量，看看哪些检查用得最多：

   ```bash
   grep -oE "OPC_[A-Za-z_0-9]+" /root/llvm-build/lib/Target/NVPTX/NVPTXGenDAGISel.inc \
     | sort | uniq -c | sort -rn | head -20
   ```
2. 构造一个"选不出来"的例子：随便改一条 IR 的命令，让某个操作变得不合法
   （比如把 `add i32` 改成对 `<8 x i32>` 的加法），然后看 `llc` 报什么错。

下一讲我们从 DAG 回到线性世界：**SSA 怎么变成可分配的 MachineIR**。
