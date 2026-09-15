# 第 4 部分 · 中端优化：-O2 到底动了哪些手脚

第 3 部分我们读的是"结果"，这一部分读"过程"。

中端（middle end）指的是"前端之后、后端之前"那一段：**输入 IR、输出 IR，
不涉及任何机器细节**。它就是一堆 pass 排队跑，每个 pass 干一件小事。

这一部分要回答四个具体问题：

```
1. -O2 到底调了哪些 pass、按什么顺序？（121 项，而且头几项很反直觉）
2. float c[4] / uint32_t a[4] 这些数组是谁消灭的？
3. 循环为什么没有展开？—— 这个问题有一个"分情况"的答案
4. convergent 到底挡住了什么？为什么一条 mma 会让一整族优化收手？
```

所有实验都在 `tools/midend-demos.sh` 和 `tools/convergent-experiment.sh` 里，可以自己重跑。

| 讲 | 标题 | 你会拿到什么 |
| --- | --- | --- |
| [04-1](04-1-pipeline.md) | 流水线全景：121 项怎么读 | 怎么把 pass 列表读成"阶段表" |
| [04-2](04-2-sroa-mem2reg.md) | SROA + mem2reg：数组是怎么消失的 | 亲手跑一遍 796 → 353 |
| [04-3](04-3-instcombine.md) | InstCombine：优化器的小账本 | `(lane&3)*2` → `(lane*2)&6` |
| [04-4](04-4-nvvm-reflect.md) | NVVMReflect：CUDA 特有的 pass | 为什么它挤在流水线最前面 |
| [04-5](04-5-loop-unroll.md) | 循环展开：四种情形 | 一个"分情况"的反直觉结论 |
| [04-6](04-6-convergent.md) | convergent 的约束 | 它一路传到机器层，让多少 pass 收手 |
