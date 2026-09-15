# 04-5 · 循环展开：一个"分情况"的反直觉结论

这一讲是第 4 部分的重头戏。我们要回答一个看起来很简单的问题：

> **我们的 k 循环，为什么不展开？**

你可能会想："因为里面有 convergent 啊，教科书都这么说。"——**这句话只对一半。**

我们把四种情形都跑了一遍，结论是：

```
trip count 已知（K 是常量）      → 会被完全展开，convergent 拦不住
trip count 未知（K 是函数参数）  → convergent 真的会拦住 runtime unroll
```

下面一步一步来。

## 一、先看我们的循环长什么样

`dumps/02-device-O2.ll` 的循环尾部：

```llvm
  %96 = add nuw nsw i32 %59, 16
  %97 = icmp slt i32 %96, %5           ; 和运行期参数 K 比较
  br i1 %97, label %58, label %38, !llvm.loop !18
```

`%5` 是函数参数（也就是 `K`），所以**编译器不知道循环要转几圈**。

## 二、实验一：K 是运行期参数 → 不可能展开

这一条没什么争议：SCEV（标量演化分析）拿不到精确的 trip count，
所以"完全展开"根本没有依据。

那"运行时展开"（runtime unroll，展开成 4 份 + 1 份余数）行不行？
**理论上可以**——但这里就是 convergent 起作用的地方（实验三会验证）。

## 三、实验二：把 K 变成编译期常量，循环就被展开了

`code/unroll_demo.cu` 是同一份代码，但 `K` 是 `constexpr int K = 64`。

```cpp
constexpr int K = 64, N = 128;
for (int k0 = 0; k0 < K; k0 += 16) { ... }   // 圈数已知：4 圈
```

编译出来：

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 --cuda-gpu-arch=sm_89 \
        --cuda-device-only -O2 -S -emit-llvm -o dumps/17-kconst-O2.ll code/unroll_demo.cu
grep -c 'mma\.sync' dumps/17-kconst-O2.ll    # 4
grep -c 'br i1'     dumps/17-kconst-O2.ll    # 0
```

```
mma asm 调用 = 4     循环 br i1 = 0
```

**循环没了，4 条 `mma` 平铺在基本块里。**

等一下——那 convergent 呢？它不是应该拦着吗？

**这就是这一讲最重要的结论：完全展开（full unroll）不需要造余数循环，
没有引入新的控制流依赖，所以 convergent 不拦它。**

| 展开方式 | 需要余数循环吗 | 含 convergent 时能不能做 |
| --- | --- | --- |
| **完全展开**（trip count 已知，一次铺开） | 不需要 | **能** |
| **运行时展开**（trip count 未知，4 份 + 1 份余数） | 需要 | **不能** |

（顺便说：老版本的 LLVM 在这里**不会**展开。这一处是版本差异的典型例子——
如果你按老教程读 IR，会以为"K 是常量也不展开"。）

## 四、实验三：运行期 K 的循环，convergent 真的挡住了

现在我们做那个关键的对照实验：拿**运行期 K** 的 IR（我们的主案例），
手工打开 runtime unroll，看 convergent 在不在会不会有差别。

```bash
# A) 原样（asm 带 convergent）
opt -passes='loop-unroll' -unroll-runtime -unroll-count=4 \
    -S dumps/02-device-O2.ll -o /tmp/a.ll

# B) 先把 convergent 从 IR 里抹掉，再跑同一条命令
grep -v '^declare.*llvm.nvvm' dumps/02-device-O2.ll \
  | sed -e 's/^\(attributes #[0-9]* = {\) convergent /\1 /' \
        -e 's/, convergent /, /' -e 's/ convergent / /' > /tmp/nocg.ll
opt -passes='loop-unroll' -unroll-runtime -unroll-count=4 \
    -S /tmp/nocg.ll -o /tmp/b.ll
```

结果（`tools/convergent-experiment.sh` 里也能跑到）：

```
A) mma_tc_manual 里的 mma 条数 = 1    <- 一点没展开
B) mma_tc_manual 里的 mma 条数 = 4    <- 展开成 4 份
```

**同一份 IR、同一条命令，唯一的差别就是 `convergent`。**

### 为什么余数循环和 convergent 冲突

源码里写得很清楚。`llvm/lib/Transforms/Scalar/LoopUnrollPass.cpp:1372`：

```cpp
  // If the loop contains a convergent operation, the prelude we'd add
  // to do the first few instructions before we hit the unrolled loop
  // is unsafe -- it adds a control-flow dependency to the convergent
  // operation. Therefore restrict remainder loop (try unrolling without).
  UP.AllowRemainder &= UCE.ConvergenceAllowsRuntime;
```

同文件第 1401 行还有一条：

```cpp
  UP.Runtime &= UCE.ConvergenceAllowsRuntime;
```

以及第 720 行：

```cpp
    ReportCannotUnroll("contains convergent operations");
```

翻译成人话：**运行时展开要造一个"余数循环"来处理不足一整轮的尾巴，
而余数循环会给 convergent 操作新增一层控制流依赖——LLVM 直接放弃。**

## 五、实验四：`#pragma unroll 4`

`code/unroll_pragma_demo.cu` 只是多了一行 `#pragma unroll 4`：

```
unroll_demo.cu        (无 pragma): mma 条数 = 4
unroll_pragma_demo.cu (有 pragma): mma 条数 = 4
unroll_pragma_demo.cu             循环 br = 0
```

两个写法的结果一样——因为 trip count 已知时本来就会展开。

那 `#pragma unroll N` 什么时候有用？两种场合：

1. **trip count 已知，但代价模型认为不划算**：它会在循环元数据里挂上
   `llvm.loop.unroll.count`，明确要求 unroller 照做；
2. **trip count 有上界但编译器证明不了精确值**：帮它选定展开因子。

**它是"我要你展开"的显式指令，代价模型管不了它。**

## 六、最后：SASS 那层还有一次展开

别忘了我们读的是 IR。运行期 K 的那个循环在 IR 层没展开，
**但 ptxas 在生成 SASS 时替我们展开了**：

```
mma_tc_manual 的 7 个 HMMA 站点：
  主循环（4 个）：0x4b0  0x5f0  0x730  0x820
  中间循环（2 个）：0xa50  0xbb0
  收尾（1 个）：0xd60
```

这是标准的"**unroll by 4 / by 2 / by 1**"分解：任意 trip count 都能写成 `4a + 2b + c`。
`tools/analyze-sass.sh` 的输出里能直接看到这个结构。

**所以"IR 没展开"不等于"机器码没展开"。** 这条经验在你调优时非常值钱：
想控制展开，你要知道自己在控制**哪一层**的展开。

## 七、小结

1. **"convergent 挡住循环展开"这个说法只对一半**：
   - trip count 已知 → 完全展开不需要余数循环 → **convergent 不拦**（LLVM 24 会展开成 4 条 mma）；
   - trip count 未知 → 运行时展开需要余数循环 → **convergent 直接否决**（A/B 对照实验：1 条 vs 4 条）。
2. 想让编译器展开，`#pragma unroll N` 是最可靠的显式手段；
   代价模型只在"没有明确指令"时才说了算。
3. IR 层不展开，不代表 SASS 层不展开——ptxas 会用 4x/2x/1x 的方式替你展开。

## 八、动手题

1. 把 `K` 改成 `constexpr` 但把循环体写得更大（比如加一些无用的计算），
   看看完全展开还会不会发生。**代价模型在什么时候开始说"不"？**
2. 用 `-pass-remarks-missed=loop-unroll` 让编译器告诉你它为什么不展开：

   ```bash
   opt -passes='loop-unroll' -unroll-runtime -unroll-count=4 \
       -pass-remarks-missed=loop-unroll -S dumps/02-device-O2.ll -o /dev/null
   ```

   （提示：这条命令在这台机器上**什么都不会打印**——因为对带 convergent 的循环，
   它连"考虑"这一步都没走到。你能解释这个"什么都不打印"意味着什么吗？）

下一讲我们把 convergent 这条线收到尾：它到底让哪些优化收手。
