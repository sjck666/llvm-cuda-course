# 第 9 章 · ptxas → SASS

## 9.0 本章目标

这是全课程你离硬件最近的一章。三件事：

1. 把 PTX 汇编成 cubin，取出 SASS，**逐条注释**；
2. 把 `HMMA.16816.F32` 的四个操作数彻底拆开，回答"A/B/C/D fragment 到底住在哪几个寄存器里"；
3. 看清楚 `ptxas` 在你看不见的地方替你做了多少事（循环展开、指令调度、指令合成）。

## 9.1 拿到 SASS

```bash
CUDA=/usr/local/cuda-12.8
export PATH=$CUDA/bin:$PATH

ptxas -arch=sm_89 -v dumps/03-clang-O2.ptx -o dumps/05-clang.cubin   # PTX -> cubin，看资源用量
cuobjdump -sass dumps/05-clang.cubin > dumps/06-sass-cuobjdump.txt
nvdisasm -c -hex dumps/05-clang.cubin > dumps/07-sass-nvdisasm.txt
bash llvm-cuda-course/tools/analyze-sass.sh                          # 洗干净 + 统计
```

资源用量：

```
ptxas info    : Compiling entry function 'mma_tc_ldmatrix' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 768 bytes smem, 388 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function 'mma_tc_manual' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers, 388 bytes cmem[0]
```

指令分布：

```
mma_tc_manual    HMMA=7   LDSM=0   LDGSTS=0   BAR=0  LDS=0   LDG=56 STG=4
mma_tc_ldmatrix  HMMA=7   LDSM=14  LDGSTS=14  BAR=14 LDS=21  LDG=7  STG=4
```

**v1 全靠 `LDG`（56 条全局加载），v2 主要靠 `LDSM`（14 条矩阵加载）+ `LDGSTS`（14 条异步拷贝），再加 14 条 `BAR`。** 两个 kernel 都是 7 个 HMMA 站点、4 条 STG。

## 9.2 SASS 格式速读

```
        /*03b0*/                   LDSM.16.MT88.2 R22, [R5] ;
        │                          │              │     └─ 地址操作数
        │                          │              └─────── 目标寄存器
        │                          └────────────────────── 操作码与修饰
        └───────────────────────────────────────────────── 指令地址（字节）
```

（这一章的地址、寄存器号、指令条数都是**这台机器、这个 ptxas 版本**跑出来的。换 `-maxrregcount`、换 ptxas 小版本，SASS 就会重新洗牌——但指令的**种类**和**数据流形状**基本稳定，那才是你要记住的东西。）

- **SASS 指令是 128 位定长编码**（16 字节），所以地址按 `0x0000 / 0x0010 / 0x0020 ...` 递增。`nvdisasm -hex` 会打两行十六进制（低 64 位 + 高 64 位）；
- 谓词写在最前面：`@!P0 BRA 0xdd0` = "P0 为假时跳转"；
- 寄存器种类：`R` 通用寄存器（32 位）、`UR` uniform 寄存器（warp 内共享的标量）、`P` 谓词、`SR` 特殊寄存器、`c[0x0][offset]` 常量存储；
- `.reuse` 是操作数复用缓存提示，`.X` 表示带进位链，`RZ` 是恒零寄存器（相当于 MIPS 的 `$zero`）。

## 9.3 `mma_tc_ldmatrix` 主循环逐条注释

我们把主循环里的一轮完整摘出来（`dumps/13-sass-clean.txt`，地址 0x360–0x0410）：

```
/*0360*/  DEPBAR.LE SB0, 0x0 ;                 ; 等异步拷贝的 scoreboard 满足
/*0370*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;        ; __syncthreads() 之一：等全 block 到齐
/*0380*/  LDSM.16.MT88.2 R22, [R2] ;           ; ldmatrix.x2.trans  -> B fragment（R22,R23）
/*0390*/  LDSM.16.M88.4  R16, [R4] ;           ; ldmatrix.x4        -> A fragment（R16..R19）
/*03a0*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;        ; __syncthreads() 之二：本轮 smem 用完
/*03b0*/  @!PT LDS RZ, [RZ] ;                  ; 占位/scoreboard 填充
/*03c0*/  @!PT LDS RZ, [RZ] ;
/*03d0*/  @!PT LDS RZ, [RZ] ;
/*03e0*/  LDGSTS.E.BYPASS.128 [R7], [R32.64+0x20] ;  ; cp.async.cg ... 16 字节（A tile）
/*03f0*/  LDGSTS.E.64 [R6], [R28.64] ;               ; cp.async.ca ...  8 字节（B tile）
/*0400*/  LDGDEPBAR ;                                ; cp.async.commit_group
/*0410*/  HMMA.16816.F32 R16, R16, R22, R12 ;        ; mma.sync（这一轮的累加）
```

### 逐条说清楚

**`DEPBAR.LE SB0, 0x0`** —— "dependency barrier"：等待 scoreboard 0 上登记的异步操作完成。对应 PTX 的 `cp.async.wait_group 0`。`ptxas` 用 scoreboard 硬件机制实现"等待未完成的内存操作"，而不是死等一个固定周期。

**`BAR.SYNC.DEFER_BLOCKING 0x0`** —— block 级 barrier（barrier 0）。`DEFER_BLOCKING` 是调度变体：让出调度槽而不完全阻塞，属于 ptxas 的微优化。

**`LDSM.16.MT88.2 R22, [R2]`** —— 这就是 `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r26,%r27}, [%r28]`：

| 片段 | 含义 |
| --- | --- |
| `LDSM` | Load Matrix |
| `.16` | 元素 16 位（b16） |
| `.MT88` | **M**atrix 8×8，**T** = transposed（对应 PTX 的 `.trans`） |
| `.2` | 装载 2 个矩阵（对应 `.x2`）→ 目标占 **2 个寄存器**（R22, R23） |
| `R22` | 目标起始寄存器 |
| `[R2]` | 共享内存地址（32 位偏移，来自那条 `cvt.u32.u64`） |

**`LDSM.16.M88.4 R16, [R4]`** —— 对应 `ldmatrix...x4.shared.b16 {%r21,%r22,%r23,%r24}, [%r25]`：`.M88` 不带 T（非转置），`.4` 装载 4 个矩阵 → 目标占 **4 个寄存器**（R16–R19）。

**`LDGSTS.E.BYPASS.128 [R7], [R32.64+0x20]`** —— 这是 `cp.async` 的真身：

| 片段 | 含义 |
| --- | --- |
| `LDGSTS` | **LD G**lobal + **ST**ore **S**hared，一条指令完成"global → shared" |
| `.E` | 扩展寻址（64 位地址） |
| `.BYPASS` | 绕过 L1 —— 对应 PTX 的 `.cg` |
| `.128` | 搬 128 位 = 16 字节 |
| `[R7]` | 目的：共享内存地址 |
| `[R32.64+0x20]` | 源：global 地址 + 立即数偏移 |

**`LDGSTS.E.64 [R6], [R28.64]`** —— 注意**没有 `.BYPASS`**，对应 PTX 的 `.ca`（走 L1），64 位 = 8 字节。**源码里 `.cg` / `.ca` 的选择，精确地映射到了 SASS 的 BYPASS 有无。**

**`LDGDEPBAR`** —— 登记依赖屏障，对应 `cp.async.commit_group`。

**`HMMA.16816.F32 R16, R16, R22, R12`** —— 主角。我们下一节专门拆它。

### 这一轮整体在干什么

把这 11 条指令按语义重排，你会看到 ptxas 的调度思路：

```
等上一次 cp.async 完成（DEPBAR）
  → 同步（BAR）
  → 用上一轮搬进来的数据装 fragment（LDSM × 2）
  → 再同步（BAR）
  → 发起下一轮的异步拷贝（LDGSTS × 2）      <- 注意：拷的是"下一轮"的数据
  → 提交（LDGDEPBAR）
  → 用刚装好的 fragment 做 mma（HMMA）
```

**它把"发起下一轮拷贝"塞在"本轮计算"之前——这就是软件流水（software pipelining）。** 拷贝是异步的，所以不会挡住后面的 HMMA；等到下一轮开头 `DEPBAR` 时才真正等它。你看源码里写的是一个直白的循环：

```cpp
for (int k0 = 0; k0 < K; k0 += 16) {
  cp_async16(...); cp_async8(...);      // 搬本轮数据
  cp_async_commit(); cp_async_wait_all();
  __syncthreads();
  ldmatrix_x4(...); ldmatrix_x2_trans(...);
  mma_m16n8k16(...);
  __syncthreads();
}
```

而 SASS 把"搬"提前到了上一轮的尾部。**这就是"编译器比你想得聪明"的地方**：`cp.async` 的异步语义给了 ptxas 重排的自由度，它把这个自由度用满了。

## 9.4 `HMMA.16816.F32` 操作数拆解

先看指令格式。手工注释的核心就是把四个操作数对应到 fragment：

```
        HMMA.16816.F32   R16,        R16,        R22,       R12 ;
                         │           │           │          │
                         │           │           │          └─ Rc（C fragment，4 个寄存器 R12–R15）
                         │           │           └──────────── Rb（B fragment，2 个寄存器 R22–R23）
                         │           └──────────────────────── Ra（A fragment，4 个寄存器 R16–R19）
                         └──────────────────────────────────── Rd（D fragment，4 个寄存器 R16–R19）
```

### 寄存器数怎么来的

| 操作数 | 名字 | 寄存器数 | 依据 |
| --- | --- | --- | --- |
| 第 1 个 | `Rd` | 4（R16,R17,R18,R19） | D 是 16×8 f32 = 128 个 float / 32 线程 = 4 |
| 第 2 个 | `Ra` | 4（R16..R19） | A 是 16×16 f16 = 256 half / 32 线程 = 8 half = 4 × b32 |
| 第 3 个 | `Rb` | 2（R22,R23） | B 是 16×8 f16 = 128 half / 32 = 4 half = 2 × b32 |
| 第 4 个 | `Rc` | 4（R12..R15） | C 是 16×8 f32，同 D |

`.16816` 就是 `m16n8k16` 的缩写（16-8-16），`.F32` 指累加器/输出的类型。

### 用数据流验证操作数角色

光看格式不够，我们**用前后指令的数据流把它钉死**。

**验证 Rd/Ra 是同一组寄存器。** `Rd == Ra == R16` 看着像笔误，其实是允许的"原地覆盖"：A 用完就没用了，结果直接写回 A 的寄存器。这一轮之后 `R16–R19` 就从"A fragment"变成了"累加器"。

**验证 Ra 确实是 A。** 在这条 HMMA 之前，`LDSM.16.M88.4 R16, [R4]` 正好把 4 个寄存器写进 R16–R19，而 A fragment 就是 4 个 b32。**一条 LDSM.x4 → 一个 Ra**，对得上。

**验证 Rb 确实是 B。** `LDSM.16.MT88.2 R22, [R2]` 写 R22–R23，正好 2 个寄存器，和 Rb 的宽度一致。

**验证 Rc 是上一轮的累加器。** 看主循环里连续三轮的 HMMA：

```
/*0410*/  HMMA.16816.F32 R16, R16, R22, R12 ;   ; D=R16..  C=R12..
/*04f0*/  HMMA.16816.F32 R8,  R8,  R20, R16 ;   ; D=R8..   C=R16..  ← 上一轮的 D
/*05b0*/  HMMA.16816.F32 R12, R12, R22, R8 ;    ; D=R12..  C=R8..   ← 上一轮的 D
/*0630*/  HMMA.16816.F32 R12, R16, R20, R12 ;   ; D=R12..  C=R12..（原地累加）
```

`C` 的位置上出现的，永远是**上一条 HMMA 的 D**。累加器在 `R12/R16/R8` 之间来回传递——这就是 `D = A*B + C` 在寄存器层面的样子。

### v1（手动版）里的同一件事

v1 没有 LDSM，A/B 都靠 `LDG` 取，但 HMMA 的形状完全一样：

```
/*03c0*/  LDG.E.CONSTANT R13, [R32.64+-0x10] ;   ┐
/*03d0*/  LDG.E.CONSTANT R12, [R30.64+-0x10] ;   │ A fragment：4 条 32 位只读加载
/*03e0*/  LDG.E.CONSTANT R14, [R30.64] ;         │  -> R12,R13,R14,R15
/*03f0*/  LDG.E.CONSTANT R15, [R32.64] ;         ┘
          ...
/*0420*/  PRMT R24, R27, 0x5410, R29 ;           ┐ B fragment：两条 16 位加载 + PRMT
/*0440*/  PRMT R25, R25, 0x5410, R35 ;           ┘  -> R24,R25
/*04b0*/  HMMA.16816.F32 R12, R12, R24, R8 ;     ; D=R12, A=R12, B=R24, C=R8
```

**这就是第 4 章那段 IR 的机器码对应**：

- A 的 `load i32` × 4 → `LDG.E.CONSTANT` × 4（`.CONSTANT` 来自参数的 `readonly`）；
- B 的两条 `load i16` + `mov.b32` 打包 → `LDG.E.U16.CONSTANT` × 2 + `PRMT`（ptxas 把 `mov.b32 {a,b}` 合成了一条 `PRMT R24, R34, 0x5410, R37`）；
- C 的 4 个 f32 → `R8..R11`（循环前用 `CS2R R8, SRZ` / `CS2R R10, SRZ` 清零）。

### fragment → SASS 寄存器映射表（v1 主循环）

| fragment | PTX 寄存器 | SASS 寄存器 | 数量 |
| --- | --- | --- | --- |
| A | `%r22..%r25` | `R12,R13,R14,R15` | 4 × b32 |
| B | `%r20,%r21` | `R24,R25`（后面几轮复用同一对） | 2 × b32 |
| C/D | `%r33..%r36` | `R8..R11`（另一轮是 `R12..R15`） | 4 × b32 |

**请记住这张表的性质**：PTX 名称到 SASS 名称的映射是 **ptxas 在寄存器分配时决定的**，不是 LLVM 决定的（第 7 章）。换一个 `-maxrregcount`、换一个 ptxas 版本，或者只是改一行代码，这张表就可能变。**唯一不变的是"数量关系"**：A 占 4 个、B 占 2 个、C/D 占 4 个。

## 9.5 `LDSM` 的两个变体：`M88` 与 `MT88`

```
/*03b0*/  LDSM.16.MT88.2 R22, [R5] ;   ; ldmatrix.x2.trans
/*03c0*/  LDSM.16.M88.4  R16, [R3] ;   ; ldmatrix.x4
```

| 字段 | `M88` | `MT88` |
| --- | --- | --- |
| `M` | Matrix | Matrix |
| `88` | 8×8 | 8×8 |
| `T` | 无 | **Transposed** |
| 对应 PTX | `.m8n8.x?.shared.b16` | `.m8n8.x?.trans.shared.b16` |
| 我们用它装 | A fragment | B fragment |

后缀数字 = 装载矩阵个数 = 目标寄存器个数（`.2` → R22,R23；`.4` → R16..R19）。

地址操作数 `[R2]` / `[R4]` 是**共享内存的 32 位偏移**——它从哪来？回到 PTX 里那几条：

```
	mov.b64 	%rd9, _ZZ15mma_tc_ldmatrixE2As;
	add.s64 	%rd10, %rd9, %rd8;
	mul.wide.u32 	%rd11, %r11, 2;
	add.s64 	%rd12, %rd10, %rd11;
	cvt.u32.u64 	%r19, %rd12;          <- 64 位符号地址 -> 32 位偏移
```

也就是说：**符号地址（64 位）→ 加法算偏移 → `cvt` 截成 32 位 → 喂给 LDSM**。这一步之所以能安全截断，是因为共享内存空间在 sm_89 上只有 48 位地址空间、且相对于基址的偏移很小。

## 9.6 ptxas 的循环展开：4x + 2x + 1x

回到第 5 章那个悬念：IR 层没展开的循环，在 SASS 里变成什么样了？

我们把 `mma_tc_manual` 的控制流骨架抽出来（`tools/analyze-sass.sh`）：

```
/*01e0*/  ISETP.GT.AND P1, PT, R20, 0x30, PT ;   ; K > 48 ?
/*0290*/  @!P1 BRA 0x870 ;
[主循环 0x2c0 – 0x860]
/*0860*/  @!P1 BRA 0x2c0 ;                       ; 回到主循环（每次前进 0x40 = 4 个 k-step）
[中间循环 0x870 – 0xbd0]
/*0880*/  ISETP.GT.AND P1, PT, R12, 0x10, PT ;
/*0890*/  @!P1 BRA 0xbd0 ;
/*0b70*/  IADD3 R5, R5, 0x20, RZ ;               ; k0 += 32
[收尾 0xbd0 – 0xde0]
/*0d60*/  HMMA.16816.F32 R8, R20, R14, R8 ;      ; 最后一次累加
```

配合 HMMA 的地址分布：

```
主循环（4 条）：0x4b0  0x5f0  0x730  0x820      每轮跨过 4 个 k-step
中间循环（2 条）：0xa50  0xbb0                  每轮跨过 2 个
收尾（1 条）：0xd60                              1 个
```

**这是教科书式的 "unroll by 4 / by 2 / by 1" 分解**：任意 trip count 都能被拆成 `4a + 2b + c`。K=64 时走 4 次主循环 + 0 次中间 + 0 次收尾，正好 64。K=48 时走 3 次主循环（48）…… 总之不管 K 是多少都能对上。

内存指令的分布也印证了这一点：

```
mma_tc_manual    HMMA=7   LDG=56  STG=4
```

56 条 LDG 里，主循环一组 20 条左右（4 个 A 的 32 位加载 + 8 个 B 的 16 位加载 + 地址计算），乘上展开因子 4，再加上中间循环和收尾。

**所以第 5 章那句话现在可以补全了**：

> `#pragma unroll` 能让 LLVM 在 **IR/PTX 层**展开；不写 pragma 时，**ptxas 会在 SASS 层**用 4x/2x/1x 的方式展开。两条路都能到终点，区别只在于"展开发生在谁的阶段"，以及**展开后 ptxas 有多少自由度去调度**。

## 9.7 ptxas 替你合成的指令

对比 PTX 和 SASS，你会看到一堆 PTX 里没有的指令。挑几个说：

| SASS | PTX 里对应什么 | 说明 |
| --- | --- | --- |
| `PRMT R24, R27, 0x5410, R29` | `mov.b32 %r20, {%rs1,%rs2}` | PTX 没有"拼两个 16 位"的 mov，SASS 用字节置换实现 |
| `IMAD.WIDE R26, R27, 0x2, R22` | `mul.wide.s32 %rd19, %r26, 2` + `add.s64` | 乘法与加法合成一条（乘加） |
| `CS2R R8, SRZ` | `mov.b32 %r38, 0f00000000` ×4 | 一条 CS2R 写两个寄存器，清零 4 个累加器只要 2 条 |
| `IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]` | （PTX 里没有） | 栈指针初始化，编译器内部约定 |
| `S2R R0, SR_TID.X` | `mov.u32 %r1, %tid.x;` | 读特殊寄存器 |
| `ULDC.64 UR6, c[0x0][0x118]` | （PTX 里是直接引用参数） | 常量加载到 uniform 寄存器 |

`PRMT` 那条特别值得记：**PTX 是"虚拟 ISA"，允许存在语义清楚但硬件没有直接对应的指令；ptxas 的职责之一就是把这些"理想指令"落地成真实指令。** 同理还有 `LDSM`（PTX 有 `ldmatrix`，硬件有 LDSM，一对一的运气好）和 `LDGSTS`（cp.async 直接对应）。

## 9.8 小结与衔接

本章四条结论：

1. **SASS 里一条 `HMMA.16816.F32` 的四个操作数就是 D、A、B、C**，寄存器数是 4 / 4 / 2 / 4，我们用前后指令的数据流钉死了这个对应关系。
2. **PTX 名 → SASS 名的映射是 ptxas 决定的**，会变；不变的是"数量关系"。
3. **`LDSM.16.M88.4` / `MT88.2` 就是 `ldmatrix.x4` / `x2.trans`**，`T` = 转置，数字 = 目标寄存器数。
4. **ptxas 做了三件你看不见的事**：把运行期 K 的 k 循环按 4x/2x/1x 展开、把 cp.async 提前一轮做软件流水、把 `mov.b32 {a,b}` 合成 `PRMT`。

到这里，"`.cu` → SASS"的完整链路就走完了。最后一张第 10 章，我们把整条链反向用一遍：**给你一条 CUDA 硬件里有、但 LLVM IR 里还没有的新指令，你该怎么从零把它接进来**——从 intrinsic 定义、TableGen pattern、到 Clang 前端识别、再到测试。
