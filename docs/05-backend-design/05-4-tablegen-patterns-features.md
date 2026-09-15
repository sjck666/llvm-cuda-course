# 05-4 · TableGen（下）：pattern、谓词与 Feature

上一讲我们看了 TableGen 怎么描述"一条指令长什么样"。这一讲看另外两件事：

```
1. 这条指令什么时候能被选中？（pattern）
2. 它在什么条件下才存在？（predicate / feature）
```

这两件事加起来，就是"**指令选择**"的全部依据。

## 一、pattern：把 IR 形状映射到指令

回头看上一讲那条指令模板（`NVPTXInstrInfo.td:367`）：

```tablegen
multiclass I3Inst<string op_str, SDPatternOperator op_node, RegTyInfo t,
                  bit commutative, list<Predicate> requires = []> {
  def rr :
    BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.RC:$b),
              op_str,
              [(set t.Ty:$dst, (op_node t.Ty:$a, t.Ty:$b))]>,   // ← pattern
              Requires<requires>;
  def ri :
    BasicNVPTXInst<(outs t.RC:$dst), (ins t.RC:$a, t.Imm:$b),
              op_str,
              [(set t.Ty:$dst, (op_node t.Ty:$a, (t.Ty imm:$b)))]>, // ← 另一个 pattern
              Requires<requires>;
```

那行 `[(set t.Ty:$dst, (op_node t.Ty:$a, t.Ty:$b))]` 就是 **pattern**。
它的意思是：

> **如果 DAG 里出现了 `op_node(类型为 t.Ty 的 a, 类型为 t.Ty 的 b)`，
> 并且结果要被塞进 `t.Ty` 类型的 dst，那就用我这条指令。**

`op_node` 是个参数。往下一层看是谁在实例化它：

```tablegen
multiclass I3<string op_str, SDPatternOperator op_node, bit commutative> {
  foreach t = [I16RT, I32RT, I64RT] in
    defm t.Size# : I3Inst<op_str # t.Size, op_node, t, commutative>;
}

defm ADD : I3<"add.s", add, commutative = true>;
defm SUB : I3<"sub.s", sub, commutative = false>;
defm MULT : I3<"mul.lo.s", mul, commutative = true>;
```

于是 `defm ADD : I3<"add.s", add, ...>` 会展开成：

```
ADD16rr / ADD16ri  模式：(set i16:$dst, (add i16:$a, i16:$b)) / (add i16:$a, imm)
ADD32rr / ADD32ri  模式：(set i32:$dst, (add i32:$a, i32:$b)) / ...
ADD64rr / ADD64ri  ...
```

**这就是我们在 MIR 里看到的 `ADD32rr`、`ADD64ri` 这些名字的来源。**
（第 06-4 讲读 MIR 时，你会看到它们真实出现。）

## 二、验证一下：`.td` 里的一行 ↔ 生成物里的一行

```bash
grep -n "ADD32rr\s*=" /root/llvm-build/lib/Target/NVPTX/NVPTXGenInstrInfo.inc
```

```
    ADD32rr                                     = 351, // NVPTXInstrInfo.td:369
```

注意那个行号 `369`——正是上一讲里 `def rr :` 所在的行。

**一条 `defm`，展开了 6 条指令；每一条在生成文件里都标着自己的物种起源。**
这就是表驱动开发的力量。

## 三、谓词（Predicate）：让指令"有条件地存在"

有些指令不是所有硬件都有。比如 `add.s16x2`（两个 16 位打包相加）
要求 PTX 8.0 且 SM 9.0 以上：

```tablegen
class I16x2<string OpcStr, SDNode OpNode> :
  BasicNVPTXInst<(outs B32:$dst), (ins B32:$a, B32:$b),
              OpcStr # "16x2",
              [(set v2i16:$dst, (OpNode v2i16:$a, v2i16:$b))]>,
              Requires<[PTX80, SM90]>;      // ← 谓词
```

`Requires<[PTX80, SM90]>` 的含义是：**只有当目标同时满足 `PTX80` 和 `SM90` 时，
这条指令才在候选集合里。**

而我们这台机器是 `sm_89` + `ptx87`，所以这条指令**根本不会被考虑**。

### 谓词从哪来

`PTX80` / `SM90` 这些名字，是 `SubtargetFeature` 的定义名字。看 `NVPTX.td:21`：

```tablegen
class FeaturePTX<int version, list<SubtargetFeature> implies>:
   SubtargetFeature<"ptx" # version, "PTXVersion",
                    "" # version,
                    "Use PTX version " # version, implies>,
   Predicate<"Subtarget->hasFeature(NVPTX::" # NAME # ")">;

defvar PTXVersions = [32, 40, 41, ..., 87, 88, 90, 91, 92, 93, 94];

foreach version = PTXVersions in
  def PTX # version : FeaturePTX<version, ...>;
```

注意这个类**同时继承了两个东西**：

```
SubtargetFeature<...>    →  它是个"特性"（能被 -mattr 打开、能在 Subtarget 里查询）
Predicate<"Subtarget->hasFeature(NVPTX::PTX87)">  →  它同时是个"谓词"（能写在 Requires<> 里）
```

**一个 `def PTX87` 同时扮演两个角色**——这是 TableGen 里很典型的一箭双雕写法。

于是整条链是：

```
命令行 -mattr=+ptx87
   → SubtargetFeature 表（TableGen 生成）设置对应位
   → Subtarget 里 PTXVersion = 87
   → 指令选择时检查 Requires<[PTX87]> → 通过
```

### 架构特性：sm_89 是怎么来的

架构那一套更复杂一点（`NVPTX.td:70` 起）：

```tablegen
class FeatureSM<list<SubtargetFeature> implies>:
   SubtargetFeature<"sm_"# !substr(NAME, 2), "HasArchitecture", "true",
                    "Target SM " # ..., implies>;
```

`"HasArchitecture"` 这个字段对应 C++ 里的 `bool HasArchitecture`（第 05-2 讲
在 `NVPTXSubtarget.h` 里见过它）。源码注释还解释了一句：

> *"Set by every architecture feature. Their bits are the point, so this is only
> here because a subtarget feature must name a field."*

翻译：**字段本身没意义，重要的是特性位；但因为框架要求"特性必须指向一个字段"，
所以随便找个 bool 挂上。**

又是一个"框架要求 vs 实际需要"的例子——和第 05-2 讲那个空壳 FrameLowering 是同一类现象。

而 sm 之间的"包含关系"（sm_89 隐含 sm_88、sm_87……）也是在 `.td` 里显式列出来的：

```tablegen
// Architectures are named sm_XYz ...
// Together they form a lattice, spelled out below as `Implies` edges:
// sm_A implies sm_B exactly when PTX written against sm_B is accepted for sm_A.
```

**这就是"我要支持 43 个 SM 版本"这件事的落地方式**：一张格（lattice）
用 `Implies` 边描述，编译成位集。

## 四、pattern 是怎么变成匹配器的

上一讲我们说过，`NVPTXGenDAGISel.inc` 有 3.2 MB。现在我们能解释它是什么了：

```bash
head -25 /root/llvm-build/lib/Target/NVPTX/NVPTXGenDAGISel.inc
```

```
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|* DAG Instruction Selector for the NVPTX target                              *|
|* Automatically generated file, do not edit!                                 *|
\*===----------------------------------------------------------------------===*/

// *** NOTE: This file is #included into the middle of the target
// *** instruction selector class.  These functions are really methods.
//
// If GET_DAGISEL_DECL is #defined with any value, only function
// declarations will be included when this file is included.
// If GET_DAGISEL_BODY is #defined, its value should be the name of the
// instruction selector class. Function bodies will be emitted
// and each function's name will be qualified with the name of the
// class.
```

它的内部结构是一个**巨大的匹配状态机**：`MatcherTable` 数组 + 一个解释它的循环。
生成的匹配器用"字节码"的形式编码所有 pattern，运行时逐条匹配 DAG 节点。

**为什么要用状态机？** 因为 pattern 之间有大量共同前缀
（比如所有 `i32` 算术都先检查类型是 `i32`），生成器会把它们**合并成前缀树**，
匹配时只走一遍。这就是"几千条 pattern 也能快速匹配"的原因。

你不需要读懂那 3.2 MB，但你需要知道三件事：

1. **它是生成的**，改 `.td` 就会变；
2. **它的输入是 `Pattern` 字段**；
3. **它的输出是一张"把 DAG 节点映射到 opcode"的表**。

## 五、自己动手：改一个 pattern 会怎样

这一段是留给你的实验。找一个你感兴趣的指令，把它的 `Requires` 改掉，
重新编译 LLVM，看行为变化。

不过"重新编译 LLVM"太慢了。有个快得多的办法——**只看 TableGen 的输出**：

```bash
/root/llvm-build/bin/llvm-tblgen \
    -I /root/llvm-project/llvm/lib/Target/NVPTX \
    -I /root/llvm-project/llvm/include \
    --gen-dag-isel /root/llvm-project/llvm/lib/Target/NVPTX/NVPTX.td \
  | grep -c "OPC_" 
```

它能在一秒内把整个匹配器生成出来（因为 TableGen 不依赖编译器的其它部分）。
**这是研究"我的 pattern 有没有被正确生成"的最快路径。**

## 六、小结

1. **pattern** 写在指令定义里（`[(set 类型:$dst, (op_node ...))]`），
   声明"什么样的 DAG 形状可以选我"；`multiclass` + `defm` 负责批量生成
   （`defm ADD : I3<...>` → `ADD16rr/ri`、`ADD32rr/ri`、`ADD64rr/ri`）。
2. **谓词**写在 `Requires<[...]>` 里，声明"我在什么条件下才存在"；
   `SubtargetFeature` 和 `Predicate` 常常是同一个 `def`（`FeaturePTX` 就是双身份）。
3. 所有 pattern 最终被编译成**一个 3.2 MB 的匹配状态机**（`NVPTXGenDAGISel.inc`），
   它就是"指令选择"的核心数据。下一部分我们会看它的输出（ISel 的结果）。

## 七、动手题

1. 在 `.td` 里找出 `defm ADD`，再用 `grep` 在生成文件里找出 `ADD16rr`、`ADD32rr`、
   `ADD64rr`，确认它们都来自同一个 `defm`。
2. 思考题：为什么 `Requires<[PTX80, SM90]>` 里要同时写 PTX 版本和 SM 版本？
   （提示：一条指令能不能用，既取决于 ISA 版本，也取决于硬件代次——
   这两个维度是独立的。）

下一讲我们看后端的 pass 流水线：这些零件是**怎么被组装成一条流水线**的。
