# 05-5 · 后端 Pass Pipeline：这些零件是怎么拼成流水线的

前四讲我们看的是"零件"：TargetMachine、Subtarget、指令描述、pattern。
这一讲看"**装配线**"——LLVM 怎么把这些零件拼成一条 pass 流水线，
以及为什么它要设计成"**基类定顺序、子类填内容**"。

## 一、先看问题：流水线里的顺序是不是随便定的

后端要做的事很多，顺序不是随便的。举个最直观的例子：

```
寄存器分配必须在指令调度之前吗？不一定——取决于目标机，有的先调度后分配
指令选择必须在合法化之后吗？必须是——先保证操作是合法的，才能选指令
PHI 消除必须在寄存器分配之前吗？必须是——PHI 是 SSA 概念，寄存器里没有 PHI
```

所以"流水线"本身是一门学问。LLVM 的答案是：

> **框架定义一个标准的骨架顺序，每个目标通过"钩子"在特定位置插入自己的 pass。**

这是经典的**模板方法模式**（template method pattern）。

## 二、两代机制：TargetPassConfig 和 CodeGenPassBuilder

LLVM 里现在**同时存在两套**拼装流水线的机制：

| 机制 | 文件 | 状态 |
| --- | --- | --- |
| **TargetPassConfig**（旧） | `llvm/include/llvm/CodeGen/TargetPassConfig.h` | 基于旧的 legacy pass manager，仍在用 |
| **CodeGenPassBuilder**（新） | `llvm/include/llvm/Passes/CodeGenPassBuilder.h` | 基于新 PM，逐步铺开 |

NVPTX **两套都有**：

```
llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp       ← 里面是旧机制（NVPTXPassConfig）
llvm/lib/Target/NVPTX/NVPTXCodeGenPassBuilder.cpp  ← 新机制（NVPTXCodeGenPassBuilder）
```

这就是为什么你在第 04-1 讲看到"同一个 pass 注册在两个地方"（`registerPassBuilderCallbacks`
和 `NVPTXPassConfig::addIRPasses`）——**同一件事，新老两套入口各写一遍**。

读源码时遇到这种"重复"，先看看是不是新老两套机制并存，别以为是冗余代码。

## 三、骨架：基类定义顺序，子类填空

基类在 `llvm/include/llvm/Passes/CodeGenPassBuilder.h`。它定义了**一大串钩子**：

```cpp
  // IR 层
  virtual void addIRPasses(PassManagerWrapper &PMW);
  virtual void addCodeGenPrepare(PassManagerWrapper &PMW);
  virtual void addISelPrepare(PassManagerWrapper &PMW);

  // 指令选择之前
  virtual void addPreISel(PassManagerWrapper &PMW) {}

  // 机器层
  virtual void addMachineSSAOptimization(PassManagerWrapper &PMW);
  virtual void addPreRegAlloc(PassManagerWrapper &PMW) {}      // 寄存器分配前
  virtual void addPostRegAlloc(PassManagerWrapper &PMW) {}      // 寄存器分配后
  virtual void addPreSched2(PassManagerWrapper &PMW) {}
  virtual void addPreEmitPass(PassManagerWrapper &PMW) {}
  virtual void addPreEmitPass2(PassManagerWrapper &PMW) {}

  // 发射
  virtual void addAsmPrinterBegin(PassManagerWrapper &PMW);
  virtual void addAsmPrinter(PassManagerWrapper &PMW);
  virtual void addAsmPrinterEnd(PassManagerWrapper &PMW);
```

注意基类里很多钩子**默认是空的**（`{}`）——也就是说：**如果目标不重写它，
就什么都不加**。

而入口只有一个：

```cpp
  Error buildPipeline(ModulePassManager &MPM, ModuleAnalysisManager &MAM,
                      raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
                      CodeGenFileType FileType, MCContext &Ctx);
```

**`buildPipeline` 就是装配线的总调度**：它按固定顺序调用那些钩子，
同时插入框架自己的通用 pass。子类只管在钩子里加自己的东西。

## 四、NVPTX 填了什么

看 NVPTX 的实现（`NVPTXCodeGenPassBuilder.cpp`）。它重写了这些钩子：

```bash
grep -n "void NVPTXCodeGenPassBuilder::add" \
  /root/llvm-project/llvm/lib/Target/NVPTX/NVPTXCodeGenPassBuilder.cpp
```

```
135:void NVPTXCodeGenPassBuilder::addEarlyCSEOrGVNPass(...)
143:void NVPTXCodeGenPassBuilder::addAddressSpaceInferencePasses(...)
157:void NVPTXCodeGenPassBuilder::addStraightLineScalarOptimizationPasses(...)
175:void NVPTXCodeGenPassBuilder::addIRPasses(...)
257:void NVPTXCodeGenPassBuilder::addPreRegAlloc(...)
265:void NVPTXCodeGenPassBuilder::addPostRegAlloc(...)
307:void NVPTXCodeGenPassBuilder::addAsmPrinterBegin(...)
311:void NVPTXCodeGenPassBuilder::addAsmPrinter(...)
315:void NVPTXCodeGenPassBuilder::addAsmPrinterEnd(...)
```

**这九个函数，就是 NVPTX 对"我这条流水线长什么样"的全部发言。**
前面我们看到的那 165 项 pass 里，属于 NVPTX 特有的那一小撮，
全部来自这几个函数。

### 举一个具体的：`addIRPasses`

```cpp
void NVPTXCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  const NVPTXSubtarget &ST = *getTM().getSubtargetImpl();

  // NVVMReflectPass is added in the pipeline-start extension point, so
  // hopefully running it here does nothing. But since we need it for
  // correctness when lowering to NVPTX, run it here too, in case whoever built
  // our pass pipeline didn't add it.
  flushFPMsToMPM(PMW);
  addModulePass(NVVMReflectPass(ST.getSmVersion()), PMW);

  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(NVPTXImageOptimizerPass(), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(NVPTXAssignValidGlobalNamesPass(), PMW);   // 分配合法的 PTX 全局名
  addModulePass(GenericToNVVMPass(), PMW);

  // Lower variadic calls before address space inference.
  addModulePass(ExpandVariadicsPass(ExpandVariadicsMode::Lowering), PMW);

  // NVPTXLowerArgs is required for correctness and should be run right
  // before the address space inference passes.
  if (getTM().getDrvInterface() == NVPTX::CUDA) {
    addFunctionPass(NVPTXMarkKernelPtrsGlobalPass(), PMW);
    flushFPMsToMPM(PMW);
  }
  addModulePass(NVPTXPromoteParamAlignPass(), PMW);
  addModulePass(NVPTXLowerArgsPass(TM), PMW);
  if (getOptLevel() != CodeGenOptLevel::None) {
    addAddressSpaceInferencePasses(PMW);
    addStraightLineScalarOptimizationPasses(PMW);
  } else {
    // Required for correct stack lowering
    addFunctionPass(NVPTXLowerAllocaPass(), PMW);
  }
  ...
```

这段话里有两个信息值得记：

1. **注释写明了顺序要求**："Lower variadic calls **before** address space inference"、
   "NVPTXLowerArgs ... should be run **right before** the address space inference passes"。
   还有开头那句"NVVMReflectPass 是为了**正确性**再加一次"。
   **后端的 pass 顺序经常是"正确性"要求，不是性能优化**——顺序错了会生成错误代码。
2. **优化级别会影响流水线内容**：`-O0` 时不跑地址空间推断，改为跑
   `NVPTXLowerAllocaPass`（"Required for correct stack lowering"——
   又是"为了正确性"）。

顺便注意代码里反复出现的两个词：

```
addModulePass(...)        加一个模块级 pass
addFunctionPass(...)      加一个函数级 pass
flushFPMsToMPM(PMW)       把当前攒着的函数流水线"冲刷"进模块流水线
```

**这就是新 pass manager 的显式性**：你不能再像老版那样随意混插模块级和函数级 pass，
必须明确"先把函数级的冲刷掉，再加模块级的"。
写后端流水线时忘记 `flushFPMsToMPM` 是常见的错误来源。

## 五、`-stop-after` 是怎么实现的

第 01-5 讲我们一直在用：

```bash
llc ... -stop-after=finalize-isel -o out.mir input.ll
```

它的实现就在基类里：

```cpp
private:
  bool runBeforeAdding(StringRef Name) {
    bool ShouldAdd = true;
    for (auto &C : BeforeCallbacks)
      ShouldAdd &= C(Name);
    return ShouldAdd;
  }

  void setStartStopPasses(const TargetPassConfig::StartStopInfo &Info);
  SmallVector<llvm::unique_function<bool(StringRef)>, 4> BeforeCallbacks;
```

再看那几个 `addXXXPass` 的实现：

```cpp
  template <typename PassT>
  void addMachineFunctionPass(PassT &&Pass, PassManagerWrapper &PMW,
                              bool Force = false, StringRef Name = PassT::name()) {
    if (!Force && !runBeforeAdding(Name))
      return;                      // ← 这里直接不加了
    PMW.MFPM.addPass(std::forward<PassT>(Pass));
  }
```

**每次往流水线里加 pass 之前，都会先问一遍回调"还该加吗"。**
`-stop-after=xxx` 就是让回调在遇到 `xxx` 之后返回 false，
于是后面的 pass 全都不加了——流水线自然就"停在"那里。

这个设计的优雅之处：**任何 pass 都能被停住，不需要每个 pass 自己支持这个功能。**

### 为什么 NVPTX 上 `-stop-after=prologepilog` 会报错

原因在第 06-1 讲过：NVPTX 禁用了通用的 `PrologEpilogInserter`，
改用自己实现的 `NVPTXPrologEpilogPass`，它的名字是 `nvptx-prolog-epilog`。

所以：

```bash
llc ... -stop-after=prologepilog          # LLVM ERROR: "prologepilog" pass is not registered.
llc ... -stop-after=nvptx-prolog-epilog   # 正确
```

**这个报错本身就是证据**：说明那一步在 NVPTX 上确实换人了。

## 六、`-print-after-all` 又是怎么实现的

和上面类似，但用的是 LLVM 的 **pass instrumentation**（埋点）机制：

```
PassInstrumentation        在 pass 运行前后插入回调的通用机制
PrintIRInstrumentation     把它实现成"打印 IR"
```

于是 `-print-after-all` 会打印**每一个** pass 之后的 IR/MIR。
我们在第 06-1 讲拿它生成了 165 项的后端 pass 清单：

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 -print-after-all \
    -o /dev/null dumps/02-device-O2.ll 2>&1 \
  | awk '/IR Dump After/ { sub(/^.*IR Dump After /, ""); sub(/ \(.*$/, ""); print }' \
  | uniq > dumps/25-codegen-passes.txt
```

**这个清单是"读了才敢信"的东西**：你不必相信任何文档说"后端会做 X"，
跑一遍就知道你的这版 LLVM 到底做了什么、按什么顺序做。

## 七、把 165 项分回它的来源

现在我们可以回答第 04-1 讲留下的问题了：那 165 项分别从哪来？

| 来源 | 例子 |
| --- | --- |
| 框架通用（基类固定加） | `Expand IR instructions`、`Module Verifier`、`CodeGen Prepare`、`Branch Probability Basic Block Placement` |
| NVPTX 钩子（`addIRPasses` 等） | `Assign valid PTX names to globals`、`NVPTX Mark Kernel Pointers Global`、`Lower pointer arguments of CUDA kernels`、`NVPTX lower atomics of local memory` |
| TableGen/目标信息驱动的机器层 | `NVPTX DAG->DAG Pattern Instruction Selection` |
| 目标包（`addPreRegAlloc`/`addPostRegAlloc`） | `NVPTX Forward Params`、`NVPTX Address Folder`、`NVPTX Proxy Register Instruction Erasure` |
| 发射 | `NVPTX Assembly Printer` |

**读一个陌生后端时，最快的入手方式就是**：

```bash
1. 找到它的 CodeGenPassBuilder / TargetPassConfig
2. 看看它重写了哪几个钩子
3. 打印一遍 pass 清单，对一下
```

这三步能让你在十分钟内对"这个后端在多做了什么"有个整体感觉。

## 八、小结

1. 后端流水线用**模板方法模式**拼装：基类 `CodeGenPassBuilder` 定顺序、
   提供空钩子；目标子类重写钩子往关键位置插自己的 pass。
2. NVPTX 通过九个 `addXXX` 函数发言；这些函数里的**注释经常写着"顺序是正确性要求"**。
3. `-stop-after` 的本质是"加 pass 前先问回调"（`runBeforeAdding`），
   `-print-after-all` 靠 pass instrumentation；两者都是**通用机制**，
   任何目标、任何 pass 都能用。

## 九、动手题

1. 打开 `NVPTXCodeGenPassBuilder.cpp` 的 `addIRPasses`，数一数它加了多少个 pass，
   其中哪些是 NVPTX 专属的。
2. 用 `-stop-after` 停在几个不同的位置，看看 MIR 的变化：

   ```bash
   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 \
       -stop-after=finalize-isel -o dumps/09-mir-after-isel.mir dumps/02-device-O2.ll
   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 \
       -stop-after=nvptx-prolog-epilog -o dumps/10-mir-after-regalloc.mir dumps/02-device-O2.ll
   diff <(grep -c "" dumps/09-mir-after-isel.mir) <(grep -c "" dumps/10-mir-after-regalloc.mir)
   ```

   两个文件的差别，就是"ISel 之后到编码之前"这一整段机器层优化的效果。

下一讲我们进入后端的核心：**SelectionDAG 是怎么从 IR 构建出来的**。
