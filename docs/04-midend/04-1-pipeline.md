# 04-1 · 流水线全景：121 项怎么读

这一讲我们做一件看起来枯燥、但极其实用的事：**把 `-O2` 的 pass 列表打出来读一遍。**

读懂它的好处是：以后看到任何奇怪的现象，你都能先问一句
"**这是哪个 pass 干的？**"，而不是瞎猜。

## 一、把流水线摊开

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 \
        --cuda-device-only --cuda-gpu-arch=sm_89 -O2 -S -emit-llvm \
        -o /dev/null -mllvm -print-pipeline-passes code/tc_mma.cu \
  | tr ',' '\n' > dumps/14-device-pipeline.txt
wc -l dumps/14-device-pipeline.txt
```

输出是一个**逗号分隔的长字符串**，用 `tr` 拆成一行一项：

```
121 dumps/14-device-pipeline.txt
```

**121 项。** 别被这个数字吓到——绝大多数是"大 pass 里的小步骤"，
真正需要你记住的只有十来项。

## 二、开头这几十项，值得完整看一眼

```
memprof-remove-attributes
annotation2metadata
forceattrs
nvvm-reflect                       <-- 第 4 项：NVPTX 目标插进来的
function(nvvm-intr-range)          <-- 第 5 项：也是 NVPTX 专属
declare-to-assign
inferattrs
coro-early
function<eager-inv>(ee-instrument<>
  lower-expect
  simplifycfg<...>
  sroa<modify-cfg>
  early-cse<>)
openmp-opt
ipsccp
called-value-propagation
globalopt
function<eager-inv>(mem2reg
  instcombine<max-iterations=1;no-verify-fixpoint>
  simplifycfg<...>)
always-inline
require<globals-aa>
...
cgscc(devirt<4>(inline
  function-attrs<...>
  ...
  function<eager-inv;no-rerun>(sroa<modify-cfg>
    early-cse<memssa>
    ...
    loop-mssa(loop-instsimplify
      loop-simplifycfg
      licm<no-allowspeculation>
      ...
      loop-unroll-full)
    ...
```

后面还有一大段（`loop-vectorize`、`slp-vectorizer`、`loop-unroll`、`gvn`、
`memcpyopt`、`dse`、`adce`、`globaldce`、`constmerge` 等等）。

**不用逐项背。** 我们要做的是把它**归纳成阶段**。

## 三、把 121 项归纳成阶段表

| 阶段 | 代表 pass | 在本案例里干了什么 |
| --- | --- | --- |
| 模块级先手 | `memprof-remove-attributes`、`annotation2metadata`、`forceattrs`、**`nvvm-reflect`**、**`nvvm-intr-range`**、`inferattrs` | 把 `__nvvm_reflect` 折成常数；给 NVVM intrinsic 补范围信息；补全属性 |
| 早期简化 | `sroa`、`early-cse`、`mem2reg`、`instcombine`、`simplifycfg` | **消灭 alloca**、做局部折叠 |
| 过程间 | `inline`、`function-attrs`、`ipsccp`、`globalopt` | 本例的 kernel 不被内联（`__global__` 不能内联），但 helper 全被内联了 |
| 循环层 | `licm`、`loop-unroll-full`、`loop-unroll<O2>`、`loop-vectorize`、`loop-sink` | **K 是编译期常量时会展开**；运行期参数时不会（04-5 讲） |
| 向量化 | `loop-vectorize`、`slp-vectorizer` | 对 `<4 x i8>` 之外的标量代码有机会；本例没触发 |
| 收尾 | `gvn`、`dse`、`adce`、`globaldce`、`constmerge` | 清死代码 |

**这张表比那 121 项有用得多。** 遇到问题先定位阶段，再找具体 pass。

## 四、第一眼就该注意到的事：第 4 项是 NVPTX 专属的

看第 4、5 项：

```
nvvm-reflect
function(nvvm-intr-range)
```

**它们不是通用 O2 流程的一部分**，而是 **NVPTX 目标自己插进来的**。
证据在源码里（`llvm/lib/Target/NVPTX/NVPTXCodeGenPassBuilder.cpp:321`）：

```cpp
void NVPTXTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  ...
  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &PM, OptimizationLevel Level) {
        // We do not want to fold out calls to nvvm.reflect early if the user
        // has not provided a target architecture just yet.
        if (Subtarget.hasTargetName())
          PM.addPass(NVVMReflectPass(Subtarget.getSmVersion()));

        FunctionPassManager FPM;
        FPM.addPass(NVVMIntrRangePass());
        ...
      });
}
```

`registerPipelineStartEPCallback` 是 PassBuilder 的**扩展点**（extension point）：
目标机注册一个回调，在建 pipeline 时把自己的 pass 挂在**最前面**。

**这是这一讲最值得带走的知识点：**

> **你看到的"clang 的 `-O2` 流水线"，其实是"通用 O2 流程 + 目标机插进来的东西"。**

第 5 部分会专门讲这套扩展点机制——它是"后端设计"里非常核心的一块。

顺带说一句：传统 codegen 路径还有第二处注册
（`llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp:244` 的 `NVPTXPassConfig::addIRPasses`，
里面第 270 行）：

```cpp
addPass(createNVVMReflectPass(ST.getSmVersion()));
```

注释写得很清楚：*"NVVMReflectPass is added in addEarlyAsPossiblePasses,
so hopefully running it here does nothing. But since we need it for correctness
when lowering to NVPTX, run it here too."* ——**"为了正确性，再跑一次也无妨"**。

## 五、另一个反直觉的点：同一个 pass 会跑好几次

在列表里搜 `sroa`、`instcombine`、`early-cse`、`simplifycfg`，
你会发现它们**反复出现**。原因有两个：

1. 有些 pass 在**不同阶段**都要跑（早期简形、后期清理）；
2. 有些 pass 是**嵌套**的，比如
   `function<eager-inv>(sroa, instcombine, simplifycfg)` 表示这一组会反复迭代到收敛。

所以"一个 pass 跑几次"没有意义。**有意义的是"在哪一步跑、为什么在那儿跑"。**

## 六、小结

1. `-O2` 的设备端流水线在这台机器上是 **121 项**，
   用 `-mllvm -print-pipeline-passes` 一条命令就能拿到。
2. 读法不是逐项背，而是**归纳成阶段**：
   模块级先手 → 早期简化 → 过程间 → 循环 → 向量化 → 收尾。
3. 第 4、5 项 `nvvm-reflect` / `nvvm-intr-range` 是 **NVPTX 目标插进来的**；
   这提醒我们：clang 的流水线 = 通用流程 + 目标机扩展。

## 七、动手题

1. 把流水线 dump 出来，数一数关键 pass 出现了几次：

   ```bash
   grep -c instcombine dumps/14-device-pipeline.txt
   grep -c sroa        dumps/14-device-pipeline.txt
   ```
2. 把 `-O2` 改成 `-O1` 或 `-O3`，重新 dump 一遍，看列表怎么变。
   变化的那部分，就是"优化级别"真正控制的东西。

下一讲我们看第一个真正的动手实验：SROA 和 mem2reg 是怎么把数组消灭掉的。
