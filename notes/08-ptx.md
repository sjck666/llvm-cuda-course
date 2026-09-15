# 第 8 章 · PTX 指令精讲

## 8.0 本章目标

前面七章，`mma.sync` 一直以"一条字符串"的身份躲在 IR 和 MIR 里。这一章我们把它拆开：

1. `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 这个名字的每一段是什么意思？
2. 四个操作数 D/A/B/C 分别对应哪几个寄存器？和我们的 fragment 表怎么对上？
3. `ldmatrix` 的地址到底该怎么算？为什么我们第一版写错了？
4. `cp.async` 的 `.ca` / `.cg` / 大小 / `commit_group` / `wait_group` 怎么配合？

## 8.1 拿到 PTX

```bash
CUDA=/usr/local/cuda-12.8
clang++ -x cuda --cuda-path=$CUDA \
        --cuda-device-only --cuda-gpu-arch=sm_89 -O2 -S -o dumps/03-clang-O2.ptx code/tc_mma.cu
nvcc -arch=sm_89 -O3 -ptx -o dumps/04-nvcc-O3.ptx code/tc_mma.cu      # 对照组
```

文件结构：

```
.version 8.7                 <- PTX ISA 版本，来自 IR 的 "+ptx87"
.target sm_89                <- target-cpu
.address_size 64

.visible .entry mma_tc_manual(          <- __global__ kernel
	.param .u64 .ptr .align 1 mma_tc_manual_param_0,   <- 三个指针
	.param .u64 .ptr .align 1 mma_tc_manual_param_1,
	.param .u64 .ptr .align 1 mma_tc_manual_param_2,
	.param .u32 mma_tc_manual_param_3,   <- 三个 int
	.param .u32 mma_tc_manual_param_4,
	.param .u32 mma_tc_manual_param_5
)
{
	.reg .pred 	%p<3>;       <- 谓词寄存器
	.reg .b16 	%rs<5>;      <- 16 位寄存器（B fragment 取 half 用）
	.reg .b32 	%r<37>;      <- 32 位寄存器（整型、打包、累加器共用）
	.reg .b64 	%rd<40>;     <- 64 位地址寄存器
```

**这些 `.reg` 声明就是第 7 章讲的"per-class 编号"**：`%r<37>` 表示"这个类里用到了 `%r1..%r36`"。注意这里**没有 `.reg .f32`**：f32 累加器和整型寄存器共用一个类，所以 `.reg .b32` 里既有 `%r20` 这样的打包数据、也有 `%r33` 这样的浮点累加器——**PTX 的虚拟寄存器按宽度分类，不按类型分类**。

## 8.2 `mma.sync` 逐字段拆解

### 名字的每一段

```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
│   │     │       │        │   │   │   │   │   └─ D 类型：f32
│   │     │       │        │   │   │   │   └───── C 类型：f32
│   │     │       │        │   │   │   └───────── B 类型：f16
│   │     │       │        │   │   └───────────── A 类型：f16
│   │     │       │        │   └───────────────── B 的布局：col（列主序）
│   │     │       │        └───────────────────── A 的布局：row（行主序）
│   │     │       └────────────────────────────── 形状：M=16, N=8, K=16
│   │     └────────────────────────────────────── aligned：warp 全 32 线程必须一起执行
│   └──────────────────────────────────────────── sync：warp 同步指令
└──────────────────────────────────────────────── 矩阵乘累加
```

三个词必须理解：

- **`sync.aligned`**：这是 warp 级指令，32 个线程必须**同时**执行到它。这就是它被标 `convergent` 的根本原因（第 5 章）。
- **`row.col`**：A 按行主序解释，B 按列主序解释。注意这说的是 **fragment 的逻辑布局**，不是内存布局——B 在 global memory 里是 `[k][n]` 行主序，但 fragment 要按"列主序"组织（每线程拿同一列的相邻两个 k）。
- **`m16n8k16`**：一个 warp 一条指令完成 16×8 的输出、吃 16 深度的 K。

### 操作数与寄存器的对应

实测的 PTX（`dumps/03-clang-O2.ptx:106`，可读性折行）：

```
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
	    {%r33,%r34,%r35,%r36},      <- D（输出）：4 个 f32
	    {%r22,%r23,%r24,%r25},      <- A：4 个 b32（8 个 f16）
	    {%r20,%r21},                <- B：2 个 b32（4 个 f16）
	    {%r33,%r34,%r35,%r36};      <- C（累加输入）：4 个 f32
```

把第 2 章那张 fragment 表搬过来对照，**逐操作数说清楚**：

| 操作数 | 寄存器数 | 每线程持有 | 对应源码 |
| --- | --- | --- | --- |
| D（第 1 个） | 4 × f32 | `D[gid][2t]`、`D[gid][2t+1]`、`D[gid+8][2t]`、`D[gid+8][2t+1]` | `float c[4]` |
| A（第 2 个） | 4 × b32 | 第 0 个：`A[gid][2t..2t+1]`；第 1 个：`A[gid+8][2t..2t+1]`；第 2 个：`A[gid][2t+8..2t+9]`；第 3 个：`A[gid+8][2t+8..2t+9]` | `uint32_t a[4]` 的 a[0..3] |
| B（第 3 个） | 2 × b32 | 第 0 个：`B[2t][gid]`,`B[2t+1][gid]`；第 1 个：`B[2t+8][gid]`,`B[2t+9][gid]` | `uint32_t b[2]` |
| C（第 4 个） | 4 × f32 | 同 D | `float c[4]` |

注意 **D 和 C 是同一组寄存器**（`%r33..%r36` 既是输入也是输出）。这就是 `D = A*B + C` 的"原地累加"形式，每次循环把结果喂回自己。

### 为什么 A 是 4 个寄存器、B 是 2 个？

算一下数据量：

- A：16×16 个 f16 = 256 个 half = 512 字节；除以 32 个线程 = 8 个 half = **4 个 b32**；
- B：16×8 个 f16 = 128 个 half = 256 字节；除以 32 = 4 个 half = **2 个 b32**；
- C/D：16×8 个 f32 = 128 个 float；除以 32 = 4 个 float = **4 个 f32**。

**"每线程寄存器数 = 总元素数 / 32"** 这个式子对任何 mma 形状都成立，是判断某个形状要几个寄存器最快的办法。

## 8.3 `ldmatrix` 精讲

### 语义

`ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1,%r2,%r3,%r4}, [addr];`

- 从**共享内存**加载 `x4` 个 `8×8` 的 16 位矩阵（一共 4×64×2 = 512 字节）；
- 每个矩阵的行地址由**连续 8 个 lane** 提供：lane 0–7 给第 1 个矩阵的 8 行，lane 8–15 给第 2 个，lane 16–23 给第 3 个，lane 24–31 给第 4 个；
- 每个 lane 拿到 **2 个连续元素**（一个 b32），分布在 4 个目标寄存器里；
- **加载结果天然就是 mma 需要的 fragment 布局**——这正是 `ldmatrix` 存在的理由：它不是普通的"批量加载"，而是"按 fragment 布局分发"。

后缀的含义：

| 后缀 | 含义 |
| --- | --- |
| `.x1` / `.x2` / `.x4` | 加载 1 / 2 / 4 个 8×8 矩阵，对应 1 / 2 / 4 个目标寄存器 |
| `.trans` | 转置：加载后按转置的方式分发（SASS 里是 `.MT88`） |
| `.b16` | 元素是 16 位 |
| `.shared` | 源地址在共享内存（必须是用 `cvta` 转出来的 32 位偏移） |

### 我们的 A 为什么是 `.x4`，地址怎么算

A 是 16×16，正好是 4 个 8×8 子块：

```
        k=0..7      k=8..15
row0-7  [ 矩阵0 ]   [ 矩阵2 ]
row8-15 [ 矩阵1 ]   [ 矩阵3 ]
```

而 mma 的 A fragment 寄存器顺序是 `a[0]=(row0-7,k0-7)`、`a[1]=(row8-15,k0-7)`、`a[2]=(row0-7,k8-15)`、`a[3]=(row8-15,k8-15)`（第 2 章的表）。所以**地址顺序必须按 0,1,2,3 排**，这就是源码里那段看起来别扭的计算：

```cpp
const int grp  = lane >> 3;                      // 0..3 -> 对应哪个矩阵
const int arow = (lane & 7) + ((grp & 1) ? 8 : 0);   // grp 是奇数 -> 下半（row+8）
const int acol = (grp >= 2) ? 8 : 0;                  // grp >= 2 -> 右半（k+8）
ldmatrix_x4(a, &As[arow][acol]);
```

`grp&1` 决定行偏移（0-7 还是 8-15），`grp>=2` 决定列偏移（k 是 0-7 还是 8-15）。**很多人第一次写 `ldmatrix.x4` 会按"先列后行"的自然顺序排地址，结果 A 的 a[1] 和 a[2] 装反**——这就是第 2 章那句"反直觉的顺序"的由来。

### 我们的 B 为什么是 `.x2.trans`，以及那次失败

B 是 16×8，分两个 8×8 块（k=0-7 和 k=8-15）。每个 lane 需要拿到**同一列 n=gid 上相邻的两个 k**。而共享内存里 `Bs` 是按 `[k][n]` 行主序存的（k 是行、n 是列）——注意这是**自然布局**，和平凡的 global 布局一致。

```cpp
const int krow = lane & 15;      // lane 0-15 分别指向 k=0..15 的行首
ldmatrix_x2_trans(b, &Bs[krow][0]);
```

`.trans` 的语义是：把"按行分发"变成"按列分发"。装载时 8 个 lane 给出的是 `Bs` 的 **k 行**地址，转置之后每个 lane 拿到的是**同一列（n）上相邻两个 k**——正是 B fragment 要的。

**我们第一版写错了**：当时把 `Bs` 在 smem 里存成 `[n][k]`（拷贝时手动转置），再让 `ldmatrix` 去 `[n][k]` 上取，地址指向 n 行。结果是转置转了两遍（拷贝时一次、`.trans` 一次），等价于没转置，`max |D - ref| = 13.9497`。改成 `[k][n]` 自然布局之后立刻 PASS。

这个教训值得写进肌肉记忆：**`ldmatrix.trans` 已经帮你转了一次，smem 里就存原始布局。** 而且 `[k][n]` 正好是 global 里 B 的自然布局，`cp.async` 可以直接搬（16×8 个 half 是连续 256 字节），不用手写转置拷贝——这也是我们最终版本的写法。

### 实测 PTX

```
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r21,%r22,%r23,%r24}, [%r25];
	ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r26,%r27}, [%r28];
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r38,%r39,%r40,%r41}, {%r21,%r22,%r23,%r24}, {%r26,%r27}, {%r38,%r39,%r40,%r41};
```

看操作数：`ldmatrix.x4` 的输出 `{%r21,%r22,%r23,%r24}` **原封不动**成了 `mma` 的 A 操作数；`ldmatrix.x2.trans` 的输出 `{%r26,%r27}` 成了 B。**中间没有任何搬运指令**——这就是 `ldmatrix` 的设计目的，也是它比手动 load 高效的原因。

（我们 v1 手动版就没有这个便利：它要么是 4 条 `ld.global.nc.b32`，要么是 2 条 `ld.global.nc.b16` + 一次打包，见 8.5 节。）

## 8.4 `cp.async` 精讲

```cpp
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"  ::"r"(s), "l"(gmem));
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group 0;\n");
```

### 名字拆解

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

| 变体 | 缓存行为 | 我们用它搬什么 |
| --- | --- | --- |
| `.ca` | cache all（L1 + L2） | B tile（8 字节/线程） |
| `.cg` | cache global（bypass L1） | A tile（16 字节/线程） |

对齐要求（踩过就记住了）：

- `16` 字节版本：源和目的都必须 16 字节对齐；
- `8` 字节版本：8 字节对齐；
- `4` 字节版本：4 字节对齐。

我们的 tile 尺寸刚好满足：A 每行 16 个 half = 32 字节，B 每行 8 个 half = 16 字节，起点都是 16 的倍数。

### 三步走：提交、等待、同步

```cpp
cp_async16(&As[r][cc], A + ...);      // 发起异步拷贝
cp_async8 (&Bs[r][cc], B + ...);
cp_async_commit();                    // commit_group: 把上面两条归成一组
cp_async_wait_all();                  // wait_group 0: 等所有组完成
__syncthreads();                      // 还要等全 block 的线程都到达，才能 ldmatrix
```

**为什么 `wait_group 0` 之后还要 `__syncthreads()`？** 因为 `cp.async` 是**线程级**的：每个线程只知道自己那 16/8 字节到位了，不知道别人（别的 warp）有没有写完。`bar.sync` 才保证整个 block 的共享内存都准备好了。

这两条在 PTX 里各有对应，在 SASS 里则是 `LDGDEPBAR` + `BAR.SYNC.DEFER_BLOCKING`（第 9 章）。

## 8.5 其它指令：一个个过

### `bar.sync 0;`

```
	bar.sync 0;
```

CTA 内 warp 间同步，barrier 0。来自 `llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)`（就是 `__syncthreads()`，第 3.5 节）。SASS 里是 `BAR.SYNC.DEFER_BLOCKING 0x0`。

### `ld.global.nc.*` —— 只读数据走非一致路径

```
	ld.global.nc.b32 	%r22, [%rd38+-16];
	ld.global.nc.b16 	%rs1, [%rd20];
```

`nc` = non-coherent，走只读数据缓存（等价于 `__ldg`）。**这个大有用处**：A/B 都是 `const __half* __restrict__`，编译器知道它们在本 kernel 内不会被写，于是用 `nc` 路径。传进 kernel 后如果被别的线程写过（数据竞争），结果未定义——但正常 GEMM 不会。

### `st.global.b32`

```
	st.global.b32 	[%rd31], %r33;
	st.global.b32 	[%rd31+4], %r34;
```

注意 `+4`：`%r33` 和 `%r34` 是同一行的相邻两列（`D[gid][2t]` 和 `D[gid][2t+1]`），所以地址差 4 字节。**这也验证了 C fragment 的布局**。

另外注意类型：存浮点用的是 `.b32` 而不是 `.f32`。因为这几个寄存器从 `mma` 的输出一路过来都是"32 位位模式"，写 `st.global.f32` 还是 `st.global.b32` 对硬件是同一件事——LLVM 选了后者。

### `cvt` / `cvta` / `mul.wide`

```
	cvta.to.global.u64 	%rd3, %rd8;         ; generic 指针 -> global 指针（第 6 章那条链）
	cvt.u32.u64 	%r19, %rd12;            ; 64 位地址截成 32 位供 shared 用
	mul.wide.s32 	%rd31, %r36, 2;         ; 32×32 -> 64 位乘法（算字节偏移）
```

`mul.wide` 是 32 位输入、64 位输出的乘法，专门用来算地址（避免溢出）。SASS 里它变成 `IMAD.WIDE`。

### `mov.b32 %r20, {%rs1,%rs2};` —— 打包两个 half

```
	ld.global.nc.b16 	%rs1, [%rd20];
	add.s64 	%rd24, %rd20, %rd23;
	ld.global.nc.b16 	%rs2, [%rd24];
	// begin inline asm
	{  mov.b32 %r20, {%rs1,%rs2};}
	// end inline asm
```

这就是 `__halves2half2` 落到 PTX 的样子。**它现在是一条 `mov`，但 ptxas 会把它变成 `PRMT`**（第 9 章），因为 SASS 里没有"把两个 16 位塞进一个 32 位"的 mov，得用字节置换。

## 8.6 clang 与 nvcc 的 PTX 对照

同一份源码，两条前端路径的 PTX 差别很大。数一下关键指令：

| 指令 | clang -O2 | nvcc -O3 |
| --- | --- | --- |
| `mma.sync` | 2 | 10 |
| `ldmatrix.sync` | 2 | 10 |
| `cp.async` | 4 | 20 |
| `bar.sync` | 2 | 10 |
| `ld.global` | 8 | 40 |
| `st.global` | 8 | 8 |

为什么差这么多？因为 **nvcc 在 PTX 层就把 k 循环展开了**（K=64 → 4 次迭代 + 收尾，每个 kernel 5 条 mma）。看 nvcc 的 manual kernel：

```
	mma.sync... {%f57,%f58,%f59,%f60}, {%r37,%r38,%r39,%r40}, {%r35,%r36}, {%f113,%f114,%f115,%f116};
	mma.sync... {%f65,%f66,%f67,%f68}, {%r45,%r46,%r47,%r48}, {%r43,%r44}, {%f57,%f58,%f59,%f60};
	mma.sync... {%f73,%f74,%f75,%f76}, {%r53,%r54,%r55,%r56}, {%r51,%r52}, {%f65,%f66,%f67,%f68};
	mma.sync... {%f113,%f114,%f115,%f116}, {%r61,%r62,%r63,%r64}, {%r59,%r60}, {%f73,%f74,%f75,%f76};
```

四条 mma 首尾相接，累加器 `%f57→%f65→%f73→%f113` 一路传下去，**没有循环**。而 clang 版本的 PTX 只有一条 mma 待在 `$L__BB0_4` 循环里。

**结论不是"谁更好"，而是分工不同**：

- nvcc 选择在自己的前端做展开（PTX 更长、更"手工"，但要占用更多寄存器）；
- clang 把展开留给 ptxas（PTX 短，SASS 里再由 ptxas 做 4x/2x/1x 展开）。

两条路最终都会到 SASS，而 SASS 会收敛得比较接近——这正是第 9 章要看的。

## 8.7 小结与衔接

- `mma.sync` 的名字把**形状、布局、类型**全写在名字里；操作数个数由"数据量 ÷ 32"决定（A=4、B=2、C/D=4）。
- `ldmatrix` 是"按 fragment 布局分发"的加载器，**它的输出可以直接喂给 mma**；`.trans` 已经包办转置，smem 里存原始布局。
- `cp.async` 的关键是"异步 + 分组（commit/wait）+ 仍需 `bar.sync`"。
- `ld.global.nc` 这种细节来自前端属性，不是后端"猜"的。
- clang 与 nvcc 的 PTX 差异主要来自**循环展开的时机**。

下一章我们把这些 PTX 喂给 `ptxas`，看它怎么变成 `HMMA.16816.F32`，以及 A/B/C/D fragment 在 SASS 里到底住在哪几个寄存器。
