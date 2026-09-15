# 05-6 · SelectionDAG 的构建：IR 怎么变成 DAG

从这一讲开始，我们进入后端的核心地带。前面讲的都是"配置"和"描述"，
从这里开始是**真正把 IR 变成机器指令**的过程。

这一讲只做一件事：**说清楚 SelectionDAG 是什么、它是怎么从 IR 构建出来的。**

## 一、为什么需要 DAG，而不是直接翻 IR

先问一个问题：为什么后端不直接从 IR 一条条翻译成机器指令？

因为**翻译需要"看全局"**。举个最简单的例子：

```llvm
%1 = mul i32 %a, 2        ; 乘 2
%2 = add i32 %1, %b       ; 再加
```

在某些机器上这能合并成一条"乘加"指令；在另一些机器上不行。
**"能不能合并"取决于机器的指令集和类型约束**，而这些约束必须一起考虑。

更麻烦的是内存操作：`load A; store B; load A` 这三条之间没有数据依赖，
但顺序不能乱（如果 A 和 B 是同一块内存）。

所以 LLVM 的做法是：**先把 IR 变成一张图（DAG），在图这种表示上做
"合法化、合并、匹配"三件事，最后再线性化回基本块。**

```
IR（线性） → DAG（图） → 合法化 / 合并 → 匹配成机器指令 → MIR（线性）
              ↑                                          ↑
       本讲 + 下两讲                               第 05-9 讲
```

## 二、DAG 的四个基本元素

理解 SelectionDAG，只需要理解四个概念：

| 概念 | 是什么 | 例子 |
| --- | --- | --- |
| **SDNode** | 图上的一个节点（一次操作） | `ISD::ADD`、`ISD::LOAD`、`ISD::CALLSEQ_START` |
| **SDValue** | "某个节点的第几个结果" | 一个节点可以有多个结果（除法的商和余数） |
| **Chain（链）** | 表示副作用顺序的隐含边 | 内存操作、调用、原子操作靠它排序 |
| **Glue（胶水）** | 表示"这几条必须挨在一起" | 一条机器指令在 DAG 里拆成多个节点时用 |

## 三、Chain：为什么需要一条专门的"链"

这是 DAG 里最容易被忽略、但最关键的一点。

在 DAG 里，节点之间的依赖本来靠**数据流**表达：谁用了谁的结果，谁就在下游。
但**内存操作之间没有数据依赖，却有顺序要求**：

```c
x = *p;      // 读
*q = 1;      // 写
y = *p;      // 再读
```

如果 `p` 和 `q` 可能指向同一块内存，这三条的顺序就不能乱——但数据流上它们毫无关系。
于是 SelectionDAG 引入了一条**显式表达副作用的边（chain）**：

```
[load #1] --chain--> [store] --chain--> [load #2]
```

**所有有副作用的操作（访存、调用、原子操作）都串在这条链上**，
保证它们不会被打乱顺序；而纯计算节点可以在链旁边自由调度。

这个设计直接影响你在 MIR 里看到的东西：第 05-4 讲那张节点属性表里的
`SDNPHasChain` / `SDNPMemOperand`，说的就是这件事。

## 四、Glue：为什么需要"胶水"

有些机器指令在 DAG 层面必须表达成多个节点（比如"结果"和"标志位"分开，
或者一条伪指令要展开成几条）。为了保证它们在后续阶段不被拆开，
LLVM 用 glue 边把它们粘在一起。

读后端代码时看到 `ISD::CopyToReg`、`Glue`、`CALLSEQ_START/END` 这些字样，
知道它们是在处理"必须成组"的语义就够了。

## 五、`SelectionDAGBuilder`：IR → DAG 的翻译官

负责这件事的文件是：

```
llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.cpp
```

它做的事很直接：**遍历 IR 的每条指令，为它创建对应的 DAG 节点**。比如：

```
IR:  %2 = add i32 %1, 1
DAG: ADD(结果类型 i32, 左操作数 = %1 的节点, 右操作数 = 常量 1)

IR:  %3 = load i32, ptr %p, align 4
DAG: LOAD(结果类型 i32, 地址 = %p 的节点, 对齐 4, chain = 当前链)

IR:  br i1 %4, label %a, label %b
DAG: BR(条件 = %4 的节点, true_bb = a, false_bb = b)
```

**DAG 是"每个基本块一张"**（严格说，构建时按基本块推进）。
这一点很重要：跨基本块的优化已经在中端做完了，后端只负责把每个块变成指令序列。

### 我们那条 load 在 DAG 里是什么

第 03-4 讲那条 IR：

```llvm
%68 = load i32, ptr %67, align 4, !tbaa !13
```

在 DAG 里变成一个 `ISD::LOAD` 节点，操作数包括：

```
chain           当前内存操作链
address         算好的地址（由若干个 ADD/GEP 节点组成）
pointer info    对齐、别名信息（来自 align 和 !tbaa）
结果类型        i32
```

而它最终会被匹配成 `LD_GLOBAL_NC_i32`（第 05-8 讲）。

## 六、目标自己的 DAG 节点：`NVPTXISD::*`

通用节点（`ISD::ADD`、`ISD::LOAD`…）不够用时，**目标可以定义自己的节点**。
NVPTX 定义了一批，都列在生成文件里：

```bash
head -40 /root/llvm-build/lib/Target/NVPTX/NVPTXGenSDNodeInfo.inc
```

```
namespace llvm::NVPTXISD {

enum GenNodeType : unsigned {
  BFI = ISD::BUILTIN_OP_END,      // ← 从通用节点的末尾开始编号
  BUILD_VECTOR,
  CALL,
  CLUSTERLAUNCHCONTROL_QUERY_CANCEL_GET_FIRST_CTAID_X,
  ...
  MUL_WIDE_SIGNED,
  MUL_WIDE_UNSIGNED,
  MoveParam,
  ...
```

**`= ISD::BUILTIN_OP_END` 是个关键细节**：目标节点从通用节点之后开始编号，
两套编号互不冲突。

它们在 `.td` 里是这么定义的（`NVPTXInstrInfo.td:1146`）：

```tablegen
def smul_wide : SDNode<"NVPTXISD::MUL_WIDE_SIGNED",   SDTMulWide, [SDNPCommutative]>;
def umul_wide : SDNode<"NVPTXISD::MUL_WIDE_UNSIGNED", SDTMulWide, [SDNPCommutative]>;
```

读法：

```
SDNode<"NVPTXISD::MUL_WIDE_SIGNED",   ← C++ 侧看到的枚举名
       SDTMulWide,                     ← 类型描述（哪些类型组合合法）
       [SDNPCommutative]>              ← 节点属性（可交换）
```

`SDNPCommutative` 这类属性会直接影响 DAG Combine：**可交换的节点，
合并时可以交换操作数来凑出更好的形状。**

### 这些节点对应到哪些机器指令

`MUL_WIDE_*` 在 MIR 里就是我们在第 06-3 讲会遇到的那一族：

```
%105:b64 = MUL_WIDEs32_ri %22, 2
```

**"目标节点 → 机器指令"的桥梁就是 pattern**（第 05-4 讲），
而这张桥被编译成了 `NVPTXGenDAGISel.inc` 那个 3.2 MB 的匹配器。

## 七、节点属性表：给通用算法看的"说明书"

生成文件里还有一张节点属性表：

```bash
grep -n "SDNPHasChain" /root/llvm-build/lib/Target/NVPTX/NVPTXGenSDNodeInfo.inc | head -3
```

```
    {129, 5, 0|1<<SDNPHasChain|1<<SDNPMemOperand, 0, 0, 1227, 973, 134}, // ...
```

| 属性 | 含义 | 谁在用 |
| --- | --- | --- |
| `SDNPHasChain` | 有副作用，必须挂在链上 | DAG 构建、Combine、调度 |
| `SDNPMemOperand` | 某个操作数是内存操作数 | 别名分析、窥孔优化 |
| `SDNPCommutative` | 可交换 | DAG Combine |
| `SDNPAssociative` | 可结合 | 重组（reassociation） |

**这就是"目标告诉通用代码"的机制**：后端不用改任何通用 pass，
只要在 `.td` 里把属性写清楚，通用算法就会正确地对待它的节点。

## 八、小结

1. 后端不直接翻译 IR，而是**先变成 DAG**——因为"合法化、合并、匹配"需要看全局形状。
2. DAG 的四个基本元素：**SDNode**（操作）、**SDValue**（结果引用）、
   **Chain**（副作用顺序）、**Glue**（必须相邻）。
   **Chain 是理解 DAG 的关键**：它让"无数据依赖但有顺序要求"的访存保持顺序。
3. 目标可以定义自己的 DAG 节点（`NVPTXISD::*`，从 `ISD::BUILTIN_OP_END` 起编号），
   并用**节点属性**告诉通用算法该怎么对待它们。

## 九、动手题

1. 看看 NVPTX 定义了多少个自己的 DAG 节点：

   ```bash
   sed -n '/enum GenNodeType/,/};/p' \
     /root/llvm-build/lib/Target/NVPTX/NVPTXGenSDNodeInfo.inc | wc -l
   ```
2. 找出 `smul_wide` 被哪个 pattern 用了（这就是"节点 → 指令"的那座桥）：

   ```bash
   grep -rn "smul_wide" /root/llvm-project/llvm/lib/Target/NVPTX/*.td | head
   ```

下一讲我们看拿到 DAG 之后做的两件事：**Legalize（合法化）和 DAG Combine（合并）**。
