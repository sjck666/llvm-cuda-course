# 05-7 · Legalize 与 DAG Combine：把 DAG 修成"这台机器能吃的形状"

上一讲我们把 IR 变成了 DAG。但那个 DAG 是**"理想化"的**——
里面可能有这台机器根本不支持的类型和操作。

这一讲讲两件事：

```
Legalize（合法化）：把不支持的变成支持的
DAG Combine（合并） ：把啰嗦的变成简洁的
```

顺序上它们是**交替进行**的：合并 → 合法化 → 再合并 → 再合法化……
直到 DAG 稳定下来。

## 一、什么叫"不合法"

举几个例子你就明白了（这些都是真实会出现的）：

```
IR 里有一个 i128 的加法        →  大多数机器没有 128 位加法，得拆成多条 64 位
IR 里有 <3 x i32> 的向量运算    →  大多数机器没有"3 元素向量"，得拆或补
IR 里对 i1 做算术运算           →  有的机器没有 1 位整数的算术，得先提升到 i8/i32
IR 里有一个 i64 的除法          →  很多 GPU 没有 64 位除法，得展开成一串指令
```

**"合法"的定义是：这台机器有直接对应的指令（或指令序列）。**

所以合法化分两类：

| 类型 | 处理什么 | 例子 |
| --- | --- | --- |
| **类型合法化** | 把不支持的类型变成支持的 | `i128` → 两个 `i64`；`i1` 加法 → 提升到 `i32` |
| **操作合法化** | 把不支持的操作变成支持的 | `sdiv i64` → 展开成一串指令或调用库函数 |

## 二、谁来定义"什么合法"：`TargetLowering` 的两张表

目标机通过两个接口告诉框架"我支持什么"：

**第一张表：哪些类型有寄存器**

`NVPTXISelLowering.cpp:599` 开始：

```cpp
  addRegisterClass(MVT::i1,   &NVPTX::B1RegClass);
  addRegisterClass(MVT::i16,  &NVPTX::B16RegClass);
  addRegisterClass(MVT::v2i16, &NVPTX::B32RegClass);
  addRegisterClass(MVT::v4i8, &NVPTX::B32RegClass);
  addRegisterClass(MVT::i32,  &NVPTX::B32RegClass);
  addRegisterClass(MVT::i64,  &NVPTX::B64RegClass);
  addRegisterClass(MVT::f32,  &NVPTX::B32RegClass);
  addRegisterClass(MVT::f64,  &NVPTX::B64RegClass);
  addRegisterClass(MVT::f16,  &NVPTX::B16RegClass);
  addRegisterClass(MVT::v2f16, &NVPTX::B32RegClass);
```

这段话的信息量很大：

1. **`MVT::i32` 对应 `NVPTX::B32RegClass`**——`B32` 就是我们在第 05-3 讲
   在 `NVPTXRegisterInfo.td` 里看到的那个寄存器类。**TableGen 里的 `def B32`
   生成了 C++ 里的 `B32RegClass`。**
2. **`f32` 和 `i32` 共用 `B32RegClass`**——又一次印证了第 02-6、05-3 讲的
   "PTX 虚拟寄存器按宽度分类"。
3. **`v4i8` 和 `v2i16` 也归 `B32`**——因为一个 32 位寄存器正好装下它们。

**第二张表：每个操作在每种类型上是什么状态**

```cpp
  setOperationAction(ISD::BUILD_VECTOR,       MVT::v2f16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT,  MVT::v2f16, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE,     MVT::v2f16, Expand);
```

`setOperationAction(操作, 类型, 动作)` 里的"动作"有四种：

| 动作 | 含义 | 谁来处理 |
| --- | --- | --- |
| `Legal` | 直接支持，不用管 | — |
| `Expand` | 展开成一串更基础的操作 | 通用代码（自动） |
| `Custom` | 目标的特殊 lowering | **你自己写 `LowerOperation`** |
| `LibCall` | 变成一个库函数调用 | 通用代码 |

**这就是"目标把自己的知识告诉框架"的方式**：

- `Expand` 让通用代码去干活（比如把 `v2f16` 的插入元素展开成位运算）；
- `Custom` 说"这个我会自己处理"，然后框架会调用目标的 `LowerOperation`。

## 三、`LowerOperation`：目标的"我特殊照顾这些"

在 NVPTX 里，`Custom` 的操作会走到这个函数（`NVPTXISelLowering.cpp:3443`）：

```cpp
SDValue
NVPTXTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  ...
```

它的典型结构是一个巨大的 `switch (Op.getOpcode())`，每个 case 处理一种操作。
你要研究某个目标的"特殊 lowering"，就是从这个函数开始读。

## 四、DAG Combine：合并与规范化

合法化解决"能不能生成"，Combine 解决"生成得好不好"。

Combine 做的事可以分成三类：

### 1）规范化（canonicalization）

把等价的形状统一成一种，方便后面匹配。比如：

```
add x, 0        → x
add x, C1, C2   → add x, (C1+C2)     常量折叠
x * 2           → x << 1               （在合适的机器上）
(x + y) + z     → x + (y + z)          规范化结合顺序
```

**为什么要规范化？** 因为 pattern 匹配是"照着形状找"的。
如果同一件事有十种写法，就得写十条 pattern；规范化之后只需要一条。

### 2）窥孔优化（peephole）

把"两条指令"合成"一条"：

```
shl x, 2 ; add y, that     →  （某些机器）一条"带移位的加法"
mul_wide a, 2 ; add b      →  （NVPTX）一步算出地址
```

### 3）目标自定义的合并

通用 Combine 处理不了的目标特殊性，靠这个钩子（`NVPTXISelLowering.cpp:7344`）：

```cpp
SDValue NVPTXTargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
```

**读一个后端时，`PerformDAGCombine` 是第二个必看函数**（第一个是 `LowerOperation`）。
它告诉你"这个目标自己做了哪些额外的图形优化"。

## 五、为什么它俩要交替进行

因为**合并可能产生新的不合法形状，合法化又可能产生新的合并机会**。

举个循环的例子：

```
初始 DAG：i128 的加法
   ↓ 类型合法化
两个 i64 的加法 + 进位处理（可能引入了新的小类型操作）
   ↓ Combine
把进位处理里的常量折叠掉
   ↓ 再合法化
某个中间类型又不合法，继续拆
   ↓ ...
```

所以 SelectionDAG 的流程其实是一个**固定点迭代**：

```
DAGCombiner → LegalizeTypes → DAGCombiner → LegalizeDAG → ... → 稳定
```

`-print-after-all` 会把这些阶段都打印出来（你会看到 `DAG combine`、
`Legalize types`、`Legalize DAG` 之类的标题）。

## 六、我们案例里的 DAG 经历了什么

我们的 kernel 其实**几乎不需要合法化**——这是第 06-3 讲会强调的一个点。为什么？

因为我们在源码里就"替编译器做完了合法化"：

```cpp
uint32_t a[4];       // 用 32 位整数装 2 个 f16，而不是用向量类型
float c[4];          // 用 4 个标量，而不是用 <4 x float>
```

如果 IR 里出现的是 `<4 x half>` 或 `<2 x float>` 这类向量类型，
后端就得处理"向量怎么拆"的问题。

**我们在第 01-2 讲说过一句话："用 32 位视角看 half 是 Tensor Core 编程的标配。"**
现在你看到了它的另一层价值：**它把类型合法化的活儿提前到源码层做完了**，
后端看到的就是最简单的标量 IR。

这是"手写 fragment 代码"的一个隐性好处——**你写得越"贴近机器"，
编译器的 lowering 就越简单，可控性也就越高**。

## 七、小结

1. 合法化分**类型合法化**（`i128` → 2×`i64`）和**操作合法化**（`sdiv i64` → 展开/库调用）；
   目标通过 `addRegisterClass` 和 `setOperationAction` 两张表告诉框架自己的底牌。
2. 动作有四种：`Legal` / `Expand`（通用展开）/ `Custom`（目标自己 lowering）/
   `LibCall`；`Custom` 的实现在 `LowerOperation`。
3. DAG Combine 负责**规范化、窥孔、目标自定义合并**（`PerformDAGCombine`），
   它和合法化**交替迭代**直到稳定。

## 八、动手题

1. 看看 NVPTX 把哪些操作标成了 `Custom`（这些是它的"特殊性"所在）：

   ```bash
   grep -n "Custom" /root/llvm-project/llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp | head -15
   ```
2. 想想：如果我们的 kernel 里 `float c[4]` 改成 `<4 x float> c`，
   后端的 DAG 会多出哪些节点？（提示：`BUILD_VECTOR`、`EXTRACT_VECTOR_ELT`……）

下一讲我们走到 DAG 旅途的终点：**指令选择——把 DAG 节点匹配成机器指令。**
