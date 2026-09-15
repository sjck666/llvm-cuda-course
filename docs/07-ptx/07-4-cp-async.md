# 07-4 · `cp.async` 精讲

共享内存要先把数据搬进去才能用。sm_80 之前这件事是"`ld.global` 到寄存器 +
`st.shared` 写共享内存"两条指令；sm_80 之后有了 `cp.async`：

> **global → shared 直接搬，中间不占寄存器，而且是异步的。**

## 一、名字拆解

```
cp.async.cg.shared.global [dst], [src], 16;
│        │  │      │       │      │    └─ 拷贝 16 字节（可选 4 / 8 / 16）
│        │  │      │       │      └────── 源：global
│        │  │      │       └───────────── 目的：shared
│        │  │      └───────────────────── 修改的是 shared 地址
│        │  └──────────────────────────── 缓存策略：cg = bypass L1（只走 L2）
│        └─────────────────────────────── 异步拷贝
└──────────────────────────────────────── 协处理器指令
```

### `.ca` 和 `.cg` 的区别

| 变体 | 缓存行为 | 我们用它搬什么 |
| --- | --- | --- |
| `.ca` | cache all（L1 + L2） | B tile（每个线程 8 字节） |
| `.cg` | cache global（bypass L1） | A tile（每个线程 16 字节） |

**这个选择会精确地反映到 SASS 上**：

```
LDGSTS.E.BYPASS.128 [R7], [R32.64+0x20] ;   ← .cg（有 .BYPASS）
LDGSTS.E.64         [R6], [R28.64] ;         ← .ca（没有 .BYPASS）
```

**源码里一个 `.cg` / `.ca`，精确地映射到 SASS 里 BYPASS 的有无。**

### 对齐要求

```
16 字节版本：源和目的都必须 16 字节对齐
 8 字节版本：8 字节对齐
 4 字节版本：4 字节对齐
```

我们的 tile 刚好满足：A 每行 16 个 half = 32 字节，B 每行 8 个 half = 16 字节。
**这也是 `__shared__` 上那个 `__align__(16)` 的原因**（第 02-3 讲）。

## 二、我们的四条 `cp.async`

```cpp
cp_async16(&As[r][cc], A + ...);      // 16 字节：A tile
cp_async8 (&Bs[r][cc], B + ...);      //  8 字节：B tile
cp_async_commit();                    // commit_group
cp_async_wait_all();                  // wait_group 0
```

PTX：

```
	cp.async.cg.shared.global [%r19], [%rd43], 16;
	cp.async.ca.shared.global [%r20], [%rd30], 8;
	cp.async.commit_group;
	cp.async.wait_group 0;
	bar.sync 	0;
```

## 三、发起、提交、等待：为什么是三条指令

```
cp.async ...            发起异步拷贝（不阻塞，立刻返回）
cp.async.commit_group   把"到现在为止发起的拷贝"归成一个组
cp.async.wait_group N   等到"最多还剩 N 个组没完成"
```

**分组是软件流水的基础**：你可以发起好几批拷贝、各自编号，
然后只等其中一部分。第 8 部分会看到 ptxas 怎么利用这一点。

### 为什么 `wait` 之后还要 `__syncthreads()`

这是初学者最容易问的问题。关键是**两条同步管的范围不同**：

```
cp.async.wait_group 0   只管"我这个线程发起的拷贝完成了没有"（线程级）
__syncthreads()         管"整个 block 的线程都到了没有"（block 级）
```

而 `ldmatrix` 读的是**整个 block 共享的**那块 tile——包括别人搬的那部分。
所以必须再加一次 block 级同步。

> **`cp.async.wait` 管"我自己的拷贝好了没"，`__syncthreads()` 管"全 block 的活都干完了没"。**

## 四、PTX 里的完整循环

```
$L__BB1_4:                              // =>This Inner Loop Header: Depth=1
	// begin inline asm
	cp.async.cg.shared.global [%r19], [%rd43], 16;
	// end inline asm
	mul.wide.s32 	%rd31, %r36, 2;
	add.s64 	%rd30, %rd4, %rd31;
	// begin inline asm
	cp.async.ca.shared.global [%r20], [%rd30], 8;
	// end inline asm
	// begin inline asm
	cp.async.commit_group;
	// end inline asm
	// begin inline asm
	cp.async.wait_group 0;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r21,%r22,%r23,%r24}, [%r25];
	// end inline asm
	// begin inline asm
	ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r26,%r27}, [%r28];
	// end inline asm
	// begin inline asm
	mma.sync.aligned...;
	// end inline asm
	bar.sync 	0;
```

**这一段几乎是源码的逐行直译**：搬数据 → 等待 → 同步 → 装 fragment → 算 → 再同步。

**但别以为最终执行的就是这个顺序！** ptxas 会把它彻底重排——把"发起下一轮拷贝"
提到"本轮计算"之前，做软件流水。第 8 部分会看到改造后的 SASS。

## 五、为什么 `cp.async` 这么好用

对比"没有它的时代"：

```
老办法（两条指令 + 占寄存器）：
  ld.global.nc.b32 r1, [gmem]      ; 读进寄存器
  st.shared.b32   [smem], r1       ; 写进共享内存
  （r1 被占用；而且这两条是同步的：必须等 load 回来才能 store）

cp.async：
  cp.async.cg.shared.global [smem], [gmem], 16;
  （不占寄存器，异步，一次 16 字节）
```

三个好处：

1. **不占寄存器**：数据从 global 直接流到 shared，不经过寄存器堆；
2. **不阻塞**：发起之后立刻返回，可以做别的事（软件流水的基础）；
3. **批量**：一条指令搬 16 字节。

## 六、小结

1. `cp.async` 是**异步的 global→shared 直搬**；`.ca`/`.cg` 是缓存策略，
   **会精确映射到 SASS 里 `LDGSTS` 的 `.BYPASS` 有没有**。
2. 它用 **`commit_group` / `wait_group`** 分组管理——这是软件流水的基础。
3. **`cp.async.wait` 之后必须再来一次 `__syncthreads()`**：
   前者线程级、后者 block 级，而 `ldmatrix` 读的是整个 block 的数据。

## 七、动手题

1. 数一数 PTX 里 `cp.async` 相关的行：

   ```bash
   grep -n "cp.async" dumps/03-clang-O2.ptx
   ```

   一共几条？其中几条真的搬数据、几条是控制指令？
2. 把源码里 A 和 B 的 `.cg` / `.ca` 对调，重新生成 PTX 和 SASS，
   看 `LDGSTS` 的 `.BYPASS` 有没有跟着变。

下一讲把剩下的 PTX 指令过一遍，并做 clang / nvcc 的对照。
