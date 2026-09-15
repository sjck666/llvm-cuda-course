# 02-5 · `__syncthreads()` 与 barrier

上一讲看的是"读硬件状态"。这一讲看"做同步"——`__syncthreads()`。

这是全课最短的一条映射（就一行 IR），但它身上有两个必须讲清楚的点：
**一个是它的名字变了**，**一个是它带着 `convergent` 标记**。
后者会在第 4 部分和第 6 部分引起一连串后果，所以从这里就开始铺垫。

## 一、一行 IR

在 `dumps/01-device-O0.ll` 里搜 `barrier`：

```llvm
call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
```

以及文件末尾的声明：

```llvm
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3
```

对应源码里的：

```cpp
__syncthreads();
```

**就这么多。** 一个参数：barrier 的编号，`0`。

## 二、这个名字是什么意思

名字很长，但拆开看逻辑很清楚：

```
llvm.nvvm.barrier.cta.sync.aligned.all
              │     │    │      │     └── all：整个 CTA 的所有线程都要到
              │     │    │      └──────── aligned：所有线程必须一起执行
              │     │    └─────────────── sync：这是一条同步操作
              │     └──────────────────── 作用范围：cta（= block）；也有 sys
              └────────────────────────── barrier 家族
```

对应的 PTX 指令是 `bar.sync 0;`——barrier 家族里"CTA 级、全线程、编号 0"的那一格。

## 三、重点：它的名字已经换过了

如果你去翻我们的旧笔记（`notes/`），或者翻网上大量 CUDA + LLVM 的文章，
你会看到这个名字：

```llvm
call void @llvm.nvvm.barrier0()
```

**这条已经不存在了。** 新版本把 NVVM 里所有 barrier 变体（`.cta`/`.sys`、
`.aligned`、`.all`/`.count`，以及各种内存序）统一收进了一套命名规则，
`__syncthreads()` 落到"CTA 级、全线程、aligned"这一格。

这类改名在这门课里不是第一次出现（第 02-2 讲是 `ptx_kernel`），
它们有一个共同的教训：

> **写 IR 之前，先确认你手上这套 LLVM 的实际名字。**
> 判断方法只有一个：**去看 PTX 里有没有真的出现那条指令**，
> 而不是看 IR "编没编过"。

你想亲自验证这条教训吗？第 9 部分我们会用 `tools/ch10-experiments.sh`
做三组对照实验（名字对 / 名字错 / 签名错），你会看到 LLVM 对这三种情况的反应完全不同。

## 四、第二个重点：它必须带 `convergent`

`bar.sync` 是**典型的收敛操作**：它要求一个 block 里所有线程都到达同一个点。
如果 warp 内部有部分线程走不到这个 barrier，硬件就会一直等下去——**死锁**。

所以这个 intrinsic 必须带 `convergent` 属性，这样 LLVM 才不会做下面这些事：

- **把它搬进条件分支**（那样只有部分线程会执行到它）；
- **把它复制成两份**（同一个 barrier 执行两次，语义就变了）；
- 一些 pass 里，它还会阻止"给循环加余数循环"这类会引入新控制流依赖的变换。

具体落到编译器里，`convergent` 会一路传下去：IR 的属性 → 机器指令的标志位
（`INLINEASM ... isconvergent`）→ 让机器层的一整族 pass 收手。
**第 4 部分和第 6 部分都在讲这条链。**

## 五、一路向下

| 层 | 形态 |
| --- | --- |
| CUDA C++ | `__syncthreads()` |
| LLVM IR | `call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)` |
| Machine IR | `BARRIER_CTA_SYNC_ALIGNED_ALL_i 0` |
| PTX | `bar.sync 0;` |
| SASS | `BAR.SYNC.DEFER_BLOCKING 0x0 ;` |

三个可以留意的点：

1. **Machine IR 里的 opcode 就是名字的"大写+下划线"版**，末尾那个 `_i` 表示
   "带一个立即数操作数"。这类命名在第 6 部分会反复出现。
2. **PTX 里就一条 `bar.sync`**，和 IR 几乎一一对应——因为 PTX 的抽象层次和 IR 很接近，
   这也是 NVPTX 后端相对"薄"的原因之一。
3. **SASS 里多了个 `.DEFER_BLOCKING`**：这是 ptxas 选的调度变体，
   意思是"让出调度槽而不是完全阻塞"。**这是 ptxas 自己加的，LLVM 没参与。**

## 六、为什么"等异步拷贝"还要再同步一次

我们的 v2 里有这么一段（第 7 部分会细讲）：

```cpp
cp_async16(&As[r][c], A + ...);       // 发起异步拷贝
cp_async8 (&Bs[r][c], B + ...);
cp_async_commit();                    // 提交成一组
cp_async_wait_all();                  // 等我这线程的拷贝完成
__syncthreads();                      // ← 为什么还要这个？
ldmatrix_x4(a, &As[arow][acol]);      // 然后才用共享内存
```

初学者最容易问的就是那句"为什么还要 `__syncthreads()`"。答案在于
**`cp.async` 的等待是"线程级"的**：

- `cp.async.wait_group 0` 只保证**当前线程**发起的那几条拷贝完成了；
- 它**不保证别的线程**（别的 warp）也把自己那份数据写好了；
- 而 `ldmatrix` 读的是**整个 block 共享的**那块 tile。

所以必须再加一次 **block 级**的同步。这两道同步的分工可以记成一句话：

> **`cp.async.wait` 管"我自己的拷贝好了没"，`__syncthreads()` 管"全 block 的活都干完了没"。**

## 七、小结

1. `__syncthreads()` → `llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)` →
   `bar.sync 0;` → `BAR.SYNC.DEFER_BLOCKING`。这条链短得几乎没有"翻译损耗"。
2. **名字已经换了**（老写法 `llvm.nvvm.barrier0` 不再存在），
   判断对错的唯一方法是在 PTX 里确认指令真的出来了。
3. 它必须带 `convergent`：barrier 要求全部线程到达，
   编译器不能把它搬进分支、也不能复制它。这个标记会一路传到机器层。

## 八、动手题

1. 在 `dumps/01-device-O0.ll`、`dumps/02-device-O2.ll`、`dumps/03-clang-O2.ptx`、
   `dumps/13-sass-clean.txt` 里各搜一次 barrier，把四种形态抄在一张纸上对照。
2. 想一想：如果我把 `__syncthreads()` 放进一个 `if (threadIdx.x < 16)` 的分支里会怎样？
   （提示：这是 CUDA 编程里的经典死锁错误。那你能说说编译器**为什么不阻止**你写这种代码吗？
   它只知道这条指令是 convergent，但它**不知道**你的分支是不是均匀的。）

下一讲我们看内联汇编——这门课的核心指令 `mma` 是怎么进来的，
以及它身上那个 `convergent` 标记到底从哪儿来。
