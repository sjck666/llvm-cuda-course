# 05-2 · TargetMachine / Subtarget / TargetOptions：后端的总装配图

上一讲我们列出了后端要提供的"零件"。这一讲看**这些零件是怎么被组织起来的**——
也就是 LLVM 三大目标类（target classes）的设计。

理解这三层，你读任何一个后端都不会迷路。

## 一、三层结构：Target / TargetMachine / Subtarget

LLVM 把"目标"这件事分成三层，粒度从粗到细：

```
Target            "有这个目标吗" —— 一个静态注册的对象，负责创建 TargetMachine
   │
TargetMachine     "为谁编译"    —— 一个具体的编译会话（三元组 + CPU + 特性 + 选项）
   │
TargetSubtarget   "这一版 CPU 有什么" —— 某个具体特性组合的信息
```

### 为什么要有 Subtarget 这一层

因为它支持**同一个模块里混用不同的 CPU 特性**。

举个现实例子：x86 可以写"这个函数用 AVX-512，那个函数用 SSE2"，
通过函数属性 `"target-features"="+avx512f"` 指定。
于是**每个函数有自己的 Subtarget**。

LLVM 的设计是：`TargetMachine::getSubtargetImpl(const Function &F)` ——
**传一个函数进去，返回它该用的 Subtarget**。

而 NVPTX 的实现在 `llvm/lib/Target/NVPTX/NVPTXTargetMachine.h` 里长这样：

```cpp
class NVPTXTargetMachine : public CodeGenTargetMachineImpl {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  NVPTXSubtarget Subtarget;          // ← 注意：只有一个

public:
  ...
  const NVPTXSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;               // ← 无视参数，永远返回同一个
  }
  const NVPTXSubtarget *getSubtargetImpl() const { return &Subtarget; }
```

**"无视参数，永远返回同一个"**——因为一个 NVPTX 模块整体只为一个 SM 版本编译
（一个 cubin 里不会混多个 SM）。这是个很典型的设计取舍：
**接口是通用的，实现是按需简化的**。

## 二、TargetMachine 提供什么

`TargetMachine` 是后端的"入口对象"。它要回答的问题包括：

```
这个目标的数据布局是什么？        getDataLayout()
指针大小、对齐、端序……

它的对象文件格式是什么？          getTargetTriple() / getMCAsmInfo()
（ELF / COFF / 纯文本？）

怎么生成代码？                    addPassesToEmitFile(...)
（这是 llc 真正调用的入口）

它有哪些子目标信息？              getSubtargetImpl(F)

优化级别、重定位模型、代码模型     getOptLevel() / getRelocationModel() / ...
```

在 LLVM 的新架构里，`TargetMachine` 的实现基类是
`CodeGenTargetMachineImpl`（NVPTX 继承的就是它）。这个细节你在读源码时会看到。

### 一个关键接口：`addPassesToEmitFile`

这是"后端组装流水线"的入口。`llc` 最终就是调用它：

```cpp
// 伪代码，表达意思
TM.addPassesToEmitFile(PM, OutStream, DwoOut, CodeGenFileType::Assembly, ...);
```

它内部会去问 `TargetPassConfig`（或者新架构里的 `CodeGenPassBuilder`）
"我们的流水线长什么样"，然后把那些 pass 加进 PassManager。

**第 05-5 讲会把这条路径完整走一遍。**

## 三、Subtarget 里装了什么

Subtarget 是"某个具体 CPU/特性组合"的信息。看 NVPTX 的定义
（`llvm/lib/Target/NVPTX/NVPTXSubtarget.h`）：

```cpp
class NVPTXSubtarget : public NVPTXGenSubtargetInfo {
  virtual void anchor();

  // PTX version x.y is represented as 10*x+y, e.g. 3.1 == 31
  unsigned PTXVersion;

  NVPTX::GPUKind Arch = NVPTX::GK_NONE;

  // Set by every architecture feature.
  bool HasArchitecture = false;

  NVPTXInstrInfo InstrInfo;                 // ← 指令信息
  NVPTXTargetLowering TLInfo;               // ← lowering 规则
  std::unique_ptr<const SelectionDAGTargetInfo> TSInfo;

  // NVPTX does not have any call stack frame, but we need an
  // NVPTX-specific FrameLowering class because TargetFrameLowering is abstract.
  NVPTXFrameLowering FrameLowering;         // ← 栈帧
```

注意三个细节，每一个都很有信息量：

**第一，Subtarget 是那些"信息类"的持有者。**
`InstrInfo`、`TLInfo`、`FrameLowering` 都是它的成员。
所以后端里到处可见这种写法：

```cpp
const NVPTXSubtarget &ST = MF.getSubtarget<NVPTXSubtarget>();
const NVPTXInstrInfo *TII = ST.getInstrInfo();
```

**第二，`PTXVersion` 用一个整数表示 `x.y`。**
注释写得很明白：`3.1 == 31`、`8.7 == 87`。这是"用整数编码版本号"的常见做法，
好处是比较大小很方便（`PTXVersion >= 70`）。

**第三，那条关于 FrameLowering 的注释特别有意思：**

> *"NVPTX does not have any call stack frame, but we need an NVPTX-specific
> FrameLowering class because TargetFrameLowering is abstract."*

翻译：**NVPTX 压根没有调用栈帧**（PTX 是虚拟 ISA，寄存器无限，不需要压栈），
但框架要求必须提供一个 FrameLowering 类，所以只好写一个几乎空的。

**这是"通用框架"和"具体目标"之间张力的一个好例子**：
框架要求你实现某个接口，你的目标可能根本用不到它的功能——
那就实现一个空壳。

## 四、TargetOptions：影响代码生成的开关

`TargetOptions` 是"一组影响代码生成的选项"，它不属于某个具体后端，
而是所有后端共享的一组通用开关。常见的有：

```
EnableFastISel        用不用 FastISel
NoInfsFPMath          浮点假设：没有 inf
NoNaNsFPMath          浮点假设：没有 NaN
UnsafeFPMath          允许不安全的浮点优化
StackAlignmentOverride 栈对齐覆盖
AllowFPOpFusion       能不能把 mul+add 融合成 fma
FunctionSections      每个函数单独放一个段
...
```

对 NVPTX 来说，最重要的几项和浮点有关（因为 GPU 上 FTZ、fma 融合是常规操作）。
你在 IR 里看到 `"no-trapping-math"="true"` 之类的函数属性，
一部分就是从这里传下去的。

## 五、从命令行到 Subtarget：一条完整路径

现在把上面的东西串起来。当你敲下：

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 input.ll -o out.ptx
```

背后发生的事：

```
1. llc 解析 -mtriple，问 TargetRegistry："谁认识 nvptx64？"
      → 找到注册好的 NVPTX target

2. 调用 NVPTX target 的 TargetMachine 构造函数，把
   Triple / CPU("sm_89") / FS("+ptx87") / TargetOptions 传进去

3. TargetMachine 构造 NVPTXSubtarget（用 CPU 和 FS 解析出：
   Arch = GK_SM89，PTXVersion = 87，以及一堆 feature 标志）

4. 后面所有 pass 需要"这台机器的信息"时，都去问 Subtarget
```

第 3 步里"用 CPU 和 FS 解析出特性"这件事，是 **TableGen 生成的表**在干活。
那张表就在我们编译出来的文件里：

```bash
head -30 /root/llvm-build/lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc
```

```
enum {
  PTX32 = 0,
  PTX40 = 1,
  PTX41 = 2,
  ...
  FeaturePTX87,
  ...
```

而另一个文件记录着"哪个 feature 对应哪一位"：

```bash
grep -n "PTX87" /root/llvm-build/lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc | head
```

**下一讲我们就把 TableGen 这层"魔法"拆开**：这些 `.inc` 文件是怎么从 `.td`
生成的，以及它们到底在被谁用。

## 六、为什么这个设计是这样的

最后回答一个设计层面的问题：**为什么要搞这么多层？**

因为要同时满足三个互相冲突的需求：

| 需求 | 设计上的应对 |
| --- | --- |
| 一个后端要支持"同一 ISA 的多个版本"（sm_80/sm_89/sm_90） | Subtarget 层 |
| 同一个模块里可能要混用不同特性（x86 的 AVX、ARM 的 SVE） | `getSubtargetImpl(Function)` |
| 不同目标的公共逻辑要复用（比如寄存器分配的算法） | 用接口类（TargetLowering 等）隔离 |

**这就是"框架设计"的典型手法：把变化的部分塞进接口，把不变的部分写成通用代码。**

而对你读代码来说，最实用的推论是：

> 遇到"这个行为为什么是这样"的问题时，先分清它是**通用代码**（在 `lib/CodeGen/`）
> 还是**目标实现**（在 `lib/Target/<目标>/`）。前者对所有后端成立，
> 后者只对这一个目标成立。

## 七、小结

1. 三层结构：**Target**（有没有这个目标）→ **TargetMachine**（为谁编译）
   → **TargetSubtarget**（这一版 CPU 有什么）。
2. Subtarget 是那些信息类的持有者（InstrInfo / TLInfo / FrameLowering）；
   NVPTX 只有一个 Subtarget（因为一个模块只为一个 SM 编译），
   而且它的 FrameLowering 是个"因为框架要求"而存在的空壳。
3. `llc -mtriple/-mcpu/-mattr` 一路变成 Subtarget 里的 Arch + PTXVersion，
   中间靠的是 TableGen 生成的特性表——下一讲就拆它。

## 八、动手题

1. 从生成文件里把所有能用的 `-mcpu` 名字列出来：

   ```bash
   grep -oE "sm_[0-9]+[a-z]*" \
     /root/llvm-build/lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc | sort -u
   ```

   你会看到 43 个名字（从 `sm_20` 一直排到 `sm_121f`）——**这就是 NVPTX 后端
   能识别的全部架构**，也是"一套后端支持所有 SM 版本"的具体体现。
2. 试试改 `-mcpu`，看同一个 IR 会生成什么不同的 PTX：

   ```bash
   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 -o - dumps/02-device-O2.ll | head -8
   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -o - dumps/02-device-O2.ll | head -8
   ```

   （提示：`.target` 那一行会变。至于指令有没有变化，取决于哪些指令有 SM 版本谓词。）

下一讲开始拆 TableGen：寄存器、指令、操作数是怎么从 `.td` 变成 C++ 的。
