# 08-5 · `LDSM` 的两个变体：`M88` 与 `MT88`

这一讲把 `ldmatrix` 在 SASS 里的样子讲清楚——毕竟它是 v2 性能的关键。

## 一、两条指令对照

```
/*0380*/  LDSM.16.MT88.2 R22, [R2] ;   ; ldmatrix.x2.trans
/*0390*/  LDSM.16.M88.4  R16, [R4] ;   ; ldmatrix.x4
```

| 字段 | `M88` | `MT88` |
| --- | --- | --- |
| `M` | Matrix | Matrix |
| `88` | 8×8 | 8×8 |
| `T` | 无 | **Transposed**（对应 PTX 的 `.trans`） |
| 对应 PTX | `.m8n8.x?.shared.b16` | `.m8n8.x?.trans.shared.b16` |
| 我们用它装 | A fragment | B fragment |

后缀数字 = **装载矩阵个数 = 目标寄存器个数**：

```
.2  →  R22, R23        （2 个寄存器）
.4  →  R16, R17, R18, R19（4 个寄存器）
```

## 二、和 PTX 的对应关系

第 07-3 讲我们写过 PTX：

```
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r21,%r22,%r23,%r24}, [%r25];
	ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r26,%r27}, [%r28];
```

对照 SASS：

| PTX | SASS |
| --- | --- |
| `ldmatrix...x4.shared.b16 {%r21..%r24}, [%r25]` | `LDSM.16.M88.4 R16, [R4]` |
| `ldmatrix...x2.trans.shared.b16 {%r26,%r27}, [%r28]` | `LDSM.16.MT88.2 R22, [R2]` |

**注意两件事**：

1. PTX 里的四个/两个寄存器，到 SASS 里变成了**一个起始寄存器**（R16 / R22）——
   后面的寄存器是隐含的（R16..R19 / R22..R23）。
2. PTX 里的共享内存地址 `[%r25]` / `[%r28]` 变成了 `[R4]` / `[R2]`。

## 三、`[R2]` / `[R4]` 是从哪儿来的

它们是**共享内存的 32 位偏移**。在 PTX 里，那两步计算长这样：

```
	mov.b64 	%rd9, _ZZ15mma_tc_ldmatrixE2As;   ; 共享内存符号的基址
	add.s64 	%rd10, %rd9, %rd8;
	mul.wide.u32 	%rd11, %r11, 2;
	add.s64 	%rd12, %rd10, %rd11;
	cvt.u32.u64 	%r19, %rd12;                 ; 64 位地址 → 32 位偏移
```

然后 ptxas 把它优化成了"SASS 里一个寄存器里放着的偏移"。

### 关键认识：共享内存的地址在 SASS 里是"基址 + 32 位偏移"

这就是为什么 PTX 里要做那次 `cvt.u32.u64` 截断：
**sm_89 的共享内存寻址本来就是 32 位的**（第 02-3 讲在 datalayout 里也见过这件事）。

## 四、为什么 `LDSM` 能省这么多

对比 v1 和 v2 装载一个 A fragment 的成本：

```
v1（手动）：
  4 条 LDG.E.CONSTANT          （全局读，每次 4 字节）
  + 地址计算（IMAD.WIDE 若干）

v2（ldmatrix）：
  1 条 LDSM.16.M88.4           （从共享内存一次拿 4 个寄存器）
  + 一条 BAR.SYNC 的代价（数据要先进共享内存）
```

**"一条 LDSM 换四条 LDG + 一堆地址计算"**——这就是它省的地方。

但注意**它不是免费的**：

```
1. 数据必须先搬进共享内存（LDGSTS + BAR.SYNC）
2. 共享内存有 bank 冲突问题（LDSM 的地址不整齐时会变慢）
3. 需要额外的同步
```

**所以"用不用 ldmatrix"本质上是一个工程取舍**：
数据复用率高的时候（比如我们每个 tile 被多个线程反复用），
走共享内存 + `LDSM` 更划算；复用率低的时候就直接 `LDG` 更省事。

## 五、一个容易忽略的细节：`LDSM` 也要 32 个 lane 一起执行

因为 `.x4` 需要 32 个 lane 各提供一个行地址（第 07-3 讲），
所以 `LDSM` 也是**收敛操作**。

你在 SASS 里看不到显式的"convergent"标记（那是 LLVM 层的东西），
但硬件层面它和 `HMMA` 一样要求整个 warp 一起到达。

**这就是为什么 LLVM 必须保守地把 `ldmatrix` 那条内联汇编标成 `convergent`**
（第 02-6、04-6 讲）：LLVM 不知道这段汇编里有什么，但它知道"保守一点总没错"。

## 六、小结

1. `LDSM.16.M88.4` = `ldmatrix.x4`（非转置，4 个目标寄存器）；
   `LDSM.16.MT88.2` = `ldmatrix.x2.trans`（转置，2 个目标寄存器）。
   **后缀数字 = 矩阵个数 = 目标寄存器个数。**
2. 地址操作数 `[R2]` / `[R4]` 是**共享内存的 32 位偏移**，
   来自 PTX 里那次 `cvt.u32.u64` 截断。
3. `LDSM` 用"一条指令"换掉了"四条 `LDG` + 地址计算"，但它要求数据先进共享内存，
   **所以用不用它是个工程取舍，不是无脑更优。**

## 七、动手题

1. 数一数 v2 里 `LDSM` 的条数和 v1 里 `LDG` 的条数，算算"一条 LDSM 顶几条 LDG"：

   ```bash
   grep -c "LDSM" dumps/13-sass-clean.txt
   grep -c "LDG"  dumps/13-sass-clean.txt
   ```
2. 想一想：如果 block 里有两个 warp，`ldmatrix` 的地址计算要怎么改？
   （提示：两个 warp 各读自己那一半 tile，地址要按 warp 加偏移。）

下一讲我们看 ptxas 在背后替我们做的那几件大事。
