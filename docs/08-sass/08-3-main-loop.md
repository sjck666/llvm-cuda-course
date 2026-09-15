# 08-3 · `ldmatrix` 主循环逐条注释

这一讲把 v2 主循环的一轮完整摘出来，逐条讲清楚。

## 一、一轮循环的原文

（`dumps/13-sass-clean.txt`，地址 0x360–0x0410）

```
/*0360*/  DEPBAR.LE SB0, 0x0 ;                       ; 等异步拷贝的 scoreboard 满足
/*0370*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;              ; __syncthreads() 之一
/*0380*/  LDSM.16.MT88.2 R22, [R2] ;                 ; ldmatrix.x2.trans -> B（R22,R23）
/*0390*/  LDSM.16.M88.4  R16, [R4] ;                 ; ldmatrix.x4       -> A（R16..R19）
/*03a0*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;              ; __syncthreads() 之二
/*03b0*/  @!PT LDS RZ, [RZ] ;                        ; 占位/scoreboard 填充
/*03c0*/  @!PT LDS RZ, [RZ] ;
/*03d0*/  @!PT LDS RZ, [RZ] ;
/*03e0*/  LDGSTS.E.BYPASS.128 [R7], [R32.64+0x20] ;  ; cp.async.cg 16 字节（A tile）
/*03f0*/  LDGSTS.E.64 [R6], [R28.64] ;               ; cp.async.ca  8 字节（B tile）
/*0400*/  LDGDEPBAR ;                                ; cp.async.commit_group
/*0410*/  HMMA.16816.F32 R16, R16, R22, R12 ;        ; 这一轮的 mma
```

## 二、逐条讲

### `DEPBAR.LE SB0, 0x0` —— 等异步操作

`DEPBAR` = **dependency barrier**。它读的是 **scoreboard**（硬件里的一组计数器），
用来登记"还有多少异步操作没完成"。`SB0` 是 scoreboard 0，`.LE 0x0` 表示
"等到它小于等于 0"。

它对应 PTX 里的 `cp.async.wait_group 0`（第 07-4 讲）。

**为什么需要它？** 因为 `cp.async` 是异步的——发起之后立刻返回，
数据什么时候到不知道。要等的时候就得靠这种机制。

### `BAR.SYNC.DEFER_BLOCKING 0x0` —— block 同步

对应 PTX 的 `bar.sync 0`，源码里的 `__syncthreads()`。

`.DEFER_BLOCKING` 是 ptxas 选的调度变体：让出调度槽而不是完全阻塞。
**这是 ptxas 自己加的，LLVM 没参与。**

注意这一轮里有**两条** `BAR.SYNC`：

```
0x370：等数据搬进来（配合 DEPBAR）
0x3a0：本轮用完共享内存，准备下一轮覆盖它
```

### `LDSM.16.MT88.2 R22, [R2]` —— `ldmatrix.x2.trans`

| 片段 | 含义 |
| --- | --- |
| `LDSM` | Load Matrix |
| `.16` | 元素 16 位（b16） |
| `.MT88` | **M**atrix 8×8，**T** = transposed（对应 PTX 的 `.trans`） |
| `.2` | 装载 2 个矩阵（对应 `.x2`）→ 目标占 **2 个寄存器**（R22, R23） |
| `R22` | 目标起始寄存器 |
| `[R2]` | 共享内存地址（32 位偏移） |

### `LDSM.16.M88.4 R16, [R4]` —— `ldmatrix.x4`

`.M88` 不带 T（非转置），`.4` 表示装载 4 个矩阵 → 目标占 **4 个寄存器**（R16–R19）。

**顺序值得注意**：这里是先 `MT88.2`（B）后 `M88.4`（A），
而源码里是先装 A 再装 B。**ptxas 把两条指令交换了**——因为两者没有依赖，
交换之后更利于硬件流水。

### 三条 `@!PT LDS RZ, [RZ]` —— 占位指令

`@!PT` 表示"谓词 PT 为假时执行"，而 `PT` 恒为真——**所以这三条永远不会执行**。

它们是 ptxas 用来"填充分支/流水线间隙"的占位指令（有些地方也叫 padding 或
scoreboard 填充）。**读 SASS 时可以完全忽略它们。**

（这类"永不执行的指令"在真实 SASS 里很常见，看到不要慌。）

### `LDGSTS.E.BYPASS.128 [R7], [R32.64+0x20]` —— `cp.async.cg` 的真身

| 片段 | 含义 |
| --- | --- |
| `LDGSTS` | **LD G**lobal + **ST**ore **S**hared，一条指令完成 global → shared |
| `.E` | 扩展寻址（64 位地址） |
| `.BYPASS` | 绕过 L1 —— 对应 PTX 的 `.cg` |
| `.128` | 搬 128 位 = 16 字节 |
| `[R7]` | 目的：共享内存地址 |
| `[R32.64+0x20]` | 源：global 地址 + 立即数偏移 |

**注意那个 `+0x20`**：32 = 4 步 × 8 字节？不，是我们的 A tile 每行 32 字节。
这个偏移说明它搬的是**下一行/下一段**的数据——**也就是"下一轮"的数据**。

### `LDGSTS.E.64 [R6], [R28.64]` —— `cp.async.ca`

**没有 `.BYPASS`**，对应 PTX 的 `.ca`（走 L1），64 位 = 8 字节。

**源码里 `.cg` / `.ca` 的选择，精确地映射到了 SASS 里 BYPASS 的有无。**

### `LDGDEPBAR` —— 提交

登记依赖屏障，对应 `cp.async.commit_group`。

### `HMMA.16816.F32 R16, R16, R22, R12` —— 主角

下一讲专门拆它。

## 三、这一轮整体在干什么（重点）

把这 11 条指令按语义重排，你会看到 ptxas 的调度思路：

```
等上一次 cp.async 完成（DEPBAR）
  → 同步（BAR）
  → 用上一轮搬进来的数据装 fragment（LDSM × 2）
  → 再同步（BAR）
  → 发起下一轮的异步拷贝（LDGSTS × 2）   ← 注意：搬的是"下一轮"的数据
  → 提交（LDGDEPBAR）
  → 用刚装好的 fragment 做 mma（HMMA）
```

**它把"发起下一轮拷贝"塞在"本轮计算"之前——这就是软件流水（software pipelining）。**

拷贝是异步的，所以不会挡住后面的 HMMA；等到下一轮开头 `DEPBAR` 时才真正等它。

### 对比源码

源码里写的是一个直白的循环：

```cpp
for (int k0 = 0; k0 < K; k0 += 16) {
  cp_async16(...); cp_async8(...);        // 搬本轮数据
  cp_async_commit(); cp_async_wait_all();
  __syncthreads();
  ldmatrix_x4(...); ldmatrix_x2_trans(...);
  mma_m16n8k16(...);
  __syncthreads();
}
```

**SASS 把"搬"提前到了上一轮的尾部。**

这就是"编译器比你想得聪明"的地方——`cp.async` 的异步语义给了 ptxas 重排的自由度，
它把这个自由度用满了。

（顺带说：**这个优化 LLVM 做不到**，因为 LLVM 不知道 GPU 的延迟有多长，
也就没法判断"提前一轮"是不是值得。见第 05-11 讲"调度发生在离硬件最近的那一层"。）

## 四、小结

1. 一轮循环的骨架是：**`DEPBAR` → `BAR` → `LDSM×2` → `BAR` → `LDGSTS×2` →
   `LDGDEPBAR` → `HMMA`**。
2. 源码里 `.cg` / `.ca` 的选择变成了 **`LDGSTS` 有没有 `.BYPASS`**；
   两条 `LDSM` 的顺序被 ptxas 交换过（因为没依赖）。
3. **`cp.async` 被提前了一整轮**——这是 ptxas 做的软件流水，源码里看不出来。
   那些 `@!PT` 的占位指令可以直接忽略。

## 五、动手题

1. 在 `dumps/13-sass-clean.txt` 里找出所有 `@!PT` 指令，数一数有多少条。
   想想它们为什么存在。（提示：和 scoreboard、分支对齐有关。）
2. 找出这个 kernel 里所有的 `BAR.SYNC`，说说每一条对应源码里的哪个 `__syncthreads()`。

下一讲我们拆 `HMMA.16816.F32` 的四个操作数。
