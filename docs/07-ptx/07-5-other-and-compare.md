# 07-5 · 其它指令与 clang/nvcc 对照

第 7 部分最后一讲，做两件事：**把剩下的 PTX 指令过一遍**，
然后**把 clang 和 nvcc 两条路径的 PTX 摆在一起比一比**。

## 一、`bar.sync 0;`

```
	bar.sync 	0;
```

CTA 内同步，barrier 编号 0。来自 `llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)`
（也就是 `__syncthreads()`，第 02-5 讲）。SASS 里是 `BAR.SYNC.DEFER_BLOCKING 0x0`。

## 二、`ld.global.nc.*`：只读数据走非一致路径

```
	ld.global.nc.b32 	%r22, [%rd38+-16];
	ld.global.nc.b16 	%rs1, [%rd20];
```

`nc` = **non-coherent**，走只读数据缓存（等价于 `__ldg`）。

**这个 `nc` 从哪来？** 来自源码里的 `const __half *__restrict__`：

```
源码的 const / __restrict__  →  IR 的 readonly + noalias 属性
                             →  PTX 的 ld.global.nc（非一致路径）
                             →  SASS 的 LDG.E.CONSTANT
```

**一条源码属性，最终改变了硬件走哪条缓存路径。** 这是"属性不是注释"最有力的证据。

（第 06-3 讲我们看到，机器 opcode 的名字里甚至直接写着 `NC`：`LD_GLOBAL_NC_i32`。）

## 三、`st.global.b32`：写回 D

```
	st.global.b32 	[%rd31], %r33;
	st.global.b32 	[%rd31+4], %r34;
	st.global.b32 	[%rd35], %r35;
	st.global.b32 	[%rd35+4], %r36;
```

注意两点：

1. **`+4`**：`%r33` 和 `%r34` 是同一行的相邻两列（`D[gid][2t]` 和 `D[gid][2t+1]`），
   所以地址差 4 字节——**这验证了 C fragment 的布局**（第 01-4 讲）。
2. **存浮点用的是 `.b32` 而不是 `.f32`**：这几个寄存器从 mma 输出一路过来都是
   "32 位位模式"，写 `.f32` 还是 `.b32` 对硬件是同一件事，LLVM 选了后者。

## 四、`cvt` / `cvta` / `mul.wide`

```
	cvta.to.global.u64 	%rd3, %rd8;         ; generic 指针 → global 指针
	cvt.u32.u64 	%r19, %rd12;            ; 64 位地址截成 32 位供 shared 用
	mul.wide.s32 	%rd31, %r36, 2;         ; 32×32 → 64 位乘法（算字节偏移）
```

三者的分工：

| 指令 | 干什么 | 来自哪里 |
| --- | --- | --- |
| `cvta.to.global` | 地址空间转换 | 第 06-2 讲那条证据链 |
| `cvt.u32.u64` | 地址截断 | `__cvta_generic_to_shared` 被折叠后的残留（第 02-3 讲） |
| `mul.wide` | 32×32→64 位乘法 | 算地址时避免溢出（SASS 里是 `IMAD.WIDE`） |

## 五、`mov.b32 %r20, {%rs1,%rs2};`：打包两个 half

```
	ld.global.nc.b16 	%rs1, [%rd20];
	add.s64 	%rd24, %rd20, %rd23;
	ld.global.nc.b16 	%rs2, [%rd24];
	// begin inline asm
	{  mov.b32 %r20, {%rs1,%rs2};}
	// end inline asm
```

这就是 `__halves2half2` 落到 PTX 的样子（第 02-7 讲）。

**它现在是一条 `mov`，但 ptxas 会把它变成 `PRMT`**（第 8 部分），
因为 SASS 里没有"把两个 16 位塞进一个 32 位"的 mov，得用字节置换。

## 六、clang vs nvcc：数一数就知道差别在哪

```bash
for pat in "mma.sync" "ldmatrix.sync" "cp.async" "bar.sync" "ld.global" "st.global"; do
  printf "%-16s clang=%-4s nvcc=%s\n" "$pat" \
    "$(grep -c "$pat" dumps/03-clang-O2.ptx)" \
    "$(grep -c "$pat" dumps/04-nvcc-O3.ptx)"
done
```

```
mma.sync         clang=2    nvcc=10
ldmatrix.sync    clang=2    nvcc=10
cp.async         clang=4    nvcc=20
bar.sync         clang=2    nvcc=10
ld.global        clang=8    nvcc=40
st.global        clang=8    nvcc=8
```

**除了 `st.global`，其它都是 5 倍关系。** 为什么？

因为 **nvcc 在 PTX 层就把 k 循环展开了**（K=64 → 4 圈 + 1 次收尾 = 5 次，
而 clang 把循环原样留着，PTX 里只有 1 份）。

看 nvcc 的 manual kernel：

```
	mma.sync... {%f57,%f58,%f59,%f60}, {%r37,%r38,%r39,%r40}, {%r35,%r36}, {%f113,%f114,%f115,%f116};
	mma.sync... {%f65,%f66,%f67,%f68}, {%r45,%r46,%r47,%r48}, {%r43,%r44}, {%f57,%f58,%f59,%f60};
	mma.sync... {%f73,%f74,%f75,%f76}, {%r53,%r54,%r55,%r56}, {%r51,%r52}, {%f65,%f66,%f67,%f68};
	mma.sync... {%f113,%f114,%f115,%f116}, {%r61,%r62,%r63,%r64}, {%r59,%r60}, {%f73,%f74,%f75,%f76};
```

**四条 mma 首尾相接，累加器 `%f57 → %f65 → %f73 → %f113` 一路传下去，没有循环。**

而 clang 版本的 PTX 里只有一条 mma 待在循环里。

### 这个差别说明什么

**不是"谁更好"，而是"活在哪一层做"：**

```
nvcc：在自己的前端（PTX 层）就展开
      → PTX 更长、更"手工"，但占用更多寄存器

clang：把展开留给下游
      → PTX 短；展开由 ptxas 在生成 SASS 时做（4x/2x/1x 分解）
```

**两条路都能到终点**，而最终的 SASS 会收敛得比较接近——
这一点我们在第 01-5 讲用 cubin 大小验证过（11.2 KB vs 10.6 KB，
而 PTX 是 8.4 KB vs 16.1 KB）。

**这也是这一讲的"总结性经验"：**

> **同一份源码，不同的前端可能在"哪一层做了优化"上分歧很大。
> 比较两者时，先看它们在**哪一层**做了决定，再看结果。**

## 七、第 7 部分总结

```
[07-1] PTX 文件结构：版本头 / 函数头 / .reg / .shared / 指令，
       每一部分都有明确来源；IR 与 PTX 几乎一一对应
[07-2] mma.sync：名字逐段拆解、四个操作数、D-A-B-C 的类型顺序
[07-3] ldmatrix：按 fragment 布局分发、地址怎么给、.trans 只转一次
[07-4] cp.async：异步直搬、.ca/.cg、commit/wait、为什么要再加 barrier
[07-5] 其它指令 + clang/nvcc 对照：差异来自"在哪一层展开"
```

现在你手里有了完整的三层对照：

```
IR（第 3 部分） → MIR（第 6 部分） → PTX（第 7 部分）
                                       ↓ 接下来
                                      SASS（第 8 部分）
```

## 八、动手题

1. 自己数一遍那张对照表（用上面的命令），确认你得到的是同样的数字。
2. 在 nvcc 的 PTX 里找出它的循环结构（如果还有的话），
   对比 clang 版本里 `$L__BB0_4` 那个循环。**说说两者的"循环"分别在哪一层消失的。**

下一部分我们走完最后一站：**PTX → SASS，看 ptxas 替我们做了多少事。**
