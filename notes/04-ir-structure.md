# 第 4 章 · LLVM IR 结构精讲

## 本章要干什么

上一章我们回答了"CUDA 的东西变成了什么"。这一章换个姿势：**把 `dumps/02-device-O2.ll` 从头读到尾**，一句一句说清楚它为什么长这样。

目标不是让你背语法，而是让你以后看到任何一份 `.ll`，都能在五分钟内回答三个问题：

1. 这段代码对应源码的哪一块？
2. 它为什么是这种形状（是前端产生的，还是某个 pass 改出来的）？
3. 它会怎么落到机器指令上？

我们只读 `mma_tc_manual` 这一个函数——`mma_tc_ldmatrix` 的形状一样，只是多了地址空间和 `barrier`。

## 4.1 读 IR 之前：四层结构 + 两个概念

LLVM IR 的结构是严格的四层嵌套：

```
Module                  <- 一个编译单元（对应一个 .cu 的 device 部分）
 └── Function           <- 一个函数（对应一个 __global__ / __device__ 函数）
      └── BasicBlock    <- 一段直线代码，只有一个入口、一个出口
           └── Instruction
```

再记住两个概念，剩下的都顺了：

**SSA（静态单赋值）**：每个变量（`%0`、`%1`……）**只被赋值一次**。所以你会看到 `%12 = lshr i32 %7, 2` 这种"算一次、用到底"的风格，永远不会出现 `%12 = ...` 第二次。循环里那些跨迭代变化的量，必须用 `phi` 节点在基本块入口"选一个值"——第 4.4 节会看到四个 `phi`。

**类型系统是显式的**：每个值都带类型，指针也带（`ptr`、`ptr addrspace(3)`），`getelementptr` 的第一个参数就是元素类型。C 里一句 `A[i]`，在 IR 里是三件事：算出字节偏移（算术）、`getelementptr` 得到指针、`load` 取值。

## 4.2 Module 头部：datalayout 与全局变量

```llvm
; ModuleID = 'tc_mma.cu'
source_filename = "tc_mma.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__half = type { i16 }

@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
@_ZZ15mma_tc_ldmatrixE2Bs = internal addrspace(3) global [16 x [8 x %struct.__half]] undef, align 16
```

### datalayout 每一段在说什么

`e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64` 用 `-` 分隔，逐段读：

| 段 | 含义 |
| --- | --- |
| `e` | little-endian（大小端）。PTX 在 sm_89 上就是小端 |
| `p6:32:32` | 地址空间 6（NVPTX 的 `tensor`，`NVPTXAddrSpace.h` 里的 `ADDRESS_SPACE_TENSOR`）里的指针是 32 位、32 位对齐。这一条 sm_89 上用不到，是给 sm_90+ 的 tensor memory 留的 |
| `i64:64` | `i64` 的对齐是 64 位（8 字节） |
| `i128:128` | `i128` 对齐 16 字节 |
| `i256:256` | `i256` 对齐 32 字节 |
| `v16:16` | 16 位向量（比如 `<2 x i8>`）对齐 16 位 |
| `v32:32` | 32 位向量（比如 `<4 x i8>`、`<2 x i16>`）对齐 32 位 |
| `n16:32:64` | "native" 整数宽度是 16/32/64 位 |

注意 `v32:32` 这一条对我们**很关键**：`<4 x i8>`、`<2 x i16>` 都是 32 位对齐——因为它们本来就住在 32 位寄存器里。这不是巧合，是 DSP/向量寄存器模型决定的（回忆一下我们研究 MIPS DSP 时那个 `DSPR : GPR32Class<[v4i8, v2i16, i32]>`，思路一模一样）。

### 全局变量里的地址空间

```llvm
@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
```

拆开读：

| 部分 | 含义 |
| --- | --- |
| `@_ZZ15mma_tc_ldmatrixE2As` | 名字，Itanium ABI 的"函数内静态变量"命名（第 3 章讲过） |
| `internal` | 链接属性：模块私有 |
| `addrspace(3)` | **共享内存** |
| `global` | 这是个全局对象（不是常量，不是别名） |
| `[16 x [16 x %struct.__half]]` | 类型：16×16 个 half |
| `undef` | 初值未定义（共享内存不需要初始化） |
| `align 16` | 对齐 16 字节（我们写的 `__align__(16)`，也是 `cp.async` 16B 的前提） |

### `%struct.__half = type { i16 }`

`__half` 就是个 16 位结构体。这点很重要：**IR 里没有"半精度浮点"这个类型**。f16 在 LLVM IR 里主要靠 `half` 类型存在（用于计算），而 CUDA 的 `__half` 因为是"存储用"的结构体，前端把它做成了 `{ i16 }`。

所以我们的 fragment 代码用 `uint32_t` 视角去看它（一次 `load i32` 拿两个 half），在 IR 层就表现为 `load i32, ptr ... !tbaa !13`——后面读到那里再说。

## 4.3 函数签名与属性组

```llvm
; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_tc_manual(
       ptr noalias nofree noundef readonly captures(none) %0,
       ptr noalias nofree noundef readonly captures(none) %1,
       ptr noalias nofree noundef writeonly captures(none) %2,
       i32 noundef %3, i32 noundef %4, i32 noundef %5)
    local_unnamed_addr #0 {
```

先看参数上的属性，这些全是 `__restrict__` 和 const 语义推出来的：

| 属性 | 来自 | 作用 |
| --- | --- | --- |
| `noalias` | `__restrict__` | 这两个指针不指向同一块内存，可以放心重排 |
| `captures(none)` | 参数没被存到别处 | 函数不会把指针泄漏出去（老写法是 `nocapture`，LLVM 24 换成了更细的 `captures(...)` 形式） |
| `nofree` | 分析出来的 | 不会释放这个指针，重排更自由 |
| `readonly` / `writeonly` | `const __half*` / `float*` | 只读 / 只写 |
| `noundef` | C++ 语义 | 传进来的值不能是 undef |

`readonly` + `noalias` 这两条一起，就是 SASS 里那些 `LDG.E.CONSTANT`（走常量缓存路径的只读加载）的合法性来源——编译器敢把 A、B 的 load 提到前面去、敢用非一致的加载路径。

再看函数属性 `#0`：

```llvm
attributes #0 = { convergent mustprogress noinline norecurse nounwind
                  "frame-pointer"="all" "no-trapping-math"="true"
                  "stack-protector-buffer-size"="8"
                  "target-cpu"="sm_89" "target-features"="+ptx87"
                  "uniform-work-group-size" }
```

| 属性 | 谁加的 | 含义 |
| --- | --- | --- |
| `convergent` | CUDA 前端（kernel 里有 convergent 操作） | 不许在它周围做破坏收敛性的变换 |
| `mustprogress` | C++11 起默认 | 循环要么终止要么有副作用 |
| `noinline` | NVPTX 后端（kernel 不许内联） | `__global__` 函数不会被内联进调用者 |
| `norecurse` | 分析出来的 | 不递归 |
| `nounwind` | CUDA 默认 | 不抛异常 |
| `"frame-pointer"="all"` | 驱动默认（`-fno-omit-frame-pointer` 风格） | 保留帧指针 |
| `"target-cpu"="sm_89"` | `--cuda-gpu-arch=sm_89` | 目标 CPU |
| `"target-features"="+ptx87"` | 同上 | **PTX ISA 8.7**——第 8 章 PTX 里的 `.version 8.7` / `.target sm_89` 就是它 |
| `"uniform-work-group-size"` | CUDA | block 大小均匀（给优化器信息）。布尔字符串属性在新版里可以只写名字，不写 `="true"` |

## 4.4 循环体逐行精讲

先看循环前的地址准备（`%23` 块），再进循环（`%58` 块）。

### 4.4.1 循环前的准备

```llvm
  %7  = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8  = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %9  = shl nuw nsw i32 %8, 4               ; tile_m = blockIdx.y * 16
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %11 = shl i32 %10, 3                      ; tile_n = blockIdx.x * 8
  %12 = lshr i32 %7, 2                      ; gid = lane >> 2
  %13 = add nuw nsw i32 %12, 8              ; gid + 8
  %14 = shl nuw nsw i32 %7, 1               ; lane * 2
  %15 = and i32 %14, 6                      ; (lane * 2) & 6  == tig * 2
  %16 = or disjoint i32 %15, 8              ; tig * 2 + 8
  %17 = icmp sgt i32 %5, 0                  ; K > 0 ?
  %18 = add nuw nsw i32 %9, %12             ; tile_m + gid
  br i1 %17, label %23, label %19
```

这里有三个值得停下来看的地方。

**第一，`lshr` 而不是 `ashr`。** 源码里 `lane` 是 `int`，`lane >> 2` 按 C 语义本该是算术右移；但 LLVM 能证明 `threadIdx.x` 非负，InstCombine 就把它换成了**逻辑**右移 `lshr`——语义更弱、后端更好优化。老版本 LLVM 在这里留的是 `ashr`，这一处变化正是"优化器多知道了一点"的体现。后面 `%14 = shl nuw nsw i32 %7, 1` 再 `and 6`，则是 InstCombine 把 `(lane & 3) * 2` 变成的**移位+掩码**。

顺带注意这些 `nuw` / `nsw`：`shl nuw nsw`、`add nuw nsw` 都是 `InstCombine` / `ValueTracking` 推出来的"无符号/有符号都不溢出"承诺。它们不是装饰，后端据此可以把两条算术合成一条 `IADD3`，或者直接用有限的位宽做地址计算。**源码里 `%` 和 `&` 写得越"干净"，这类标记就越多**——这也是手写 kernel 时值得注意的一个习惯。

**第二，`or disjoint`。** `disjoint` 是个很有用的提示：说明这两个操作数的置位区间不重叠，`or` 不会产生进位。它是 InstCombine 推出来的，后端可以据此把它当加法处理（SASS 里我们确实看到 `IADD3` 或者直接合并进地址计算）。

**第三，前半段就分了岔路。** `br i1 %17, label %23, label %19` —— `K > 0` 才进主循环，否则直接跳到 %38（写回块），把 4 个 0 存回去。这是 `-O2` 里的 loop-rotate 干的，好处是主循环里少一次判断。SASS 里对应的就是 `ISETP.LT.AND P0, PT, RZ, c[0x0][0x180]` + `@!P0 BRA 0xd80`。

接着 `%23` 块算地址：

```llvm
  %24 = mul nuw nsw i32 %5, %18             ; (tile_m + gid) * K
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %25      ; &A[row][0]
  %27 = add nuw nsw i32 %13, %9
  %28 = mul nuw nsw i32 %5, %27
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %29      ; &A[row+8][0]
  %31 = zext nneg i32 %15 to i64            ; tig*2  (无符号扩展)
  %32 = zext nneg i32 %16 to i64            ; tig*2+8
  %33 = sext i32 %11 to i64
  %34 = getelementptr [2 x i8], ptr %1, i64 %33                  ; &B[0][tile_n]
  %35 = zext nneg i32 %12 to i64
  %36 = getelementptr [2 x i8], ptr %34, i64 %35                 ; &B[0][tile_n + gid]
  %37 = sext i32 %4 to i64                  ; N (用来跨行)
```

**这就是 GEP 的精髓**：`getelementptr inbounds nuw [2 x i8], ptr %0, i64 %25` 的意思是"以 `[2 x i8]`（也就是一个 `__half`）为单位，往前走 `%25` 个元素"，这一步隐含了 `× 2`。**在 IR 里你永远看不到显式的 `× 2`**，这就是"类型化指针"的代价和便利。

注意 GEP 的元素类型写的是 `[2 x i8]` 而不是 `%struct.__half`：`-O2` 会把"元素类型"规格化成它认为最方便的形式，只要**大小**不变，语义就不变（`%struct.__half` = `{ i16 }` = 2 字节 = `[2 x i8]`）。读 IR 时别被这个换了马甲的类型骗到——**看尺寸，不看名字**。

注意 `%36 = getelementptr [2 x i8], ptr %34, i64 %35` —— 这里 `inbounds` 没了，因为后面循环里会用它加上 `%37`(=`N`) 去取 `B[k+1][n]`，编译器算不出一定不越界。

### 4.4.2 循环体：四个 phi 就是 C fragment

```llvm
58:                                               ; preds = %23, %58
  %59 = phi i32   [ 0, %23 ],         [ %96, %58 ]       ; k0
  %60 = phi float [ 0.000000e+00, %23 ], [ %92, %58 ]    ; c[0]
  %61 = phi float [ 0.000000e+00, %23 ], [ %93, %58 ]    ; c[1]
  %62 = phi float [ 0.000000e+00, %23 ], [ %94, %58 ]    ; c[2]
  %63 = phi float [ 0.000000e+00, %23 ], [ %95, %58 ]    ; c[3]
```

五个 `phi`，正好对应源码里"跨迭代存活"的五个变量：循环变量 `k0` 和 4 个累加器 `c[0..3]`。

**这里能看出一个关键事实：`float c[4]` 这个数组在 O2 之后彻底不存在了**，它变成了 4 个 `phi` + 一组寄存器。第 4.5 节我们看 O0 是怎么写的，形成对照。

接着是 A fragment 的加载：

```llvm
  %64 = zext nneg i32 %59 to i64            ; k0
  %65 = getelementptr inbounds nuw [2 x i8], ptr %26, i64 %64     ; &A[row][k0]
  %66 = getelementptr inbounds nuw [2 x i8], ptr %30, i64 %64     ; &A[row+8][k0]
  %67 = getelementptr inbounds nuw [2 x i8], ptr %65, i64 %31     ; + tig*2
  %68 = load i32, ptr %67, align 4, !tbaa !13                     ; a[0]
  %69 = getelementptr inbounds nuw [2 x i8], ptr %66, i64 %31
  %70 = load i32, ptr %69, align 4, !tbaa !13                     ; a[1]
  %71 = getelementptr inbounds nuw [2 x i8], ptr %65, i64 %32     ; + tig*2+8
  %72 = load i32, ptr %71, align 4, !tbaa !13                     ; a[2]
  %73 = getelementptr inbounds nuw [2 x i8], ptr %66, i64 %32
  %74 = load i32, ptr %73, align 4, !tbaa !13                     ; a[3]
```

四条 `load i32`，对应 A fragment 的 4 个 `.b32`。**这是"用 32 位视角看 half"的胜利**：如果按 `__half` 逐个取，会是 8 条 `load i16`；我们写成 `*reinterpret_cast<const uint32_t*>(p)`，前端就发一条 `load i32`。

到了 SASS 里，这四条就是 `LDG.E.CONSTANT R13, [R32.64+-0x10]` 这种 4 字节只读加载（`CONSTANT` 后缀来自参数的 `readonly`）。

然后是 B fragment，画风突变：

```llvm
  %75 = or disjoint i32 %59, %15                  ; k0 + tig*2   （低位不重叠，可以当加法）
  %76 = mul nsw i32 %75, %4
  %77 = sext i32 %76 to i64
  %78 = getelementptr [2 x i8], ptr %36, i64 %77  ; &B[k0 + tig*2][n]
  %79 = or disjoint i32 %59, %16                  ; k0 + tig*2 + 8
  %80 = mul nsw i32 %79, %4
  %81 = sext i32 %80 to i64
  %82 = getelementptr [2 x i8], ptr %36, i64 %81
  %83 = load i16, ptr %78, align 2, !tbaa !14     ; B[k][n]
  %84 = getelementptr inbounds [2 x i8], ptr %78, i64 %37   ; + N
  %85 = load i16, ptr %84, align 2, !tbaa !14     ; B[k+1][n]
  %86 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %83, i16 %85) #3, !srcloc !16
```

**两条 16 位加载 + 一条汇编打包**。为什么 A 是一条 32 位加载、B 就得两条 16 位？回看第 2 章的 fragment 表：A 的同一寄存器里是**同一行相邻的两个 k**（内存连续），B 的同一寄存器里是**同一列相邻的两个 k**（内存隔了 N 个元素）。所以 B 必须分开取、再打包。

这就是 Tensor Core 编程里那个著名的"B 要转置"问题的 IR 形态。到了 SASS，`mov.b32` 不再是独立指令，而是变成了一条 `PRMT R24, R27, 0x5410, R29`——`PRMT` 是字节置换指令，`0x5410` 选择字节，正好干"两个 16 位拼成一个 32 位"的活。

顺带看 `%75 = or disjoint i32 %59, %15`：`%59` 是 `k0`（16 的倍数，低 4 位为 0），`%15` 是 `tig*2`（小于 16），两者低位不重叠，所以 `or` 等价于 `add`——前端用 `or` 表达"这里没有进位"，后端就能放心地把它折进地址计算。

接着是本节的压轴：

```llvm
  %91 = tail call contract { float, float, float, float } asm sideeffect
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
        (i32 %68, i32 %70, i32 %72, i32 %74, i32 %86, i32 %90,
         float %60, float %61, float %62, float %63) #4, !srcloc !17
  %92 = extractvalue { float, float, float, float } %91, 0
  %93 = extractvalue { float, float, float, float } %91, 1
  %94 = extractvalue { float, float, float, float } %91, 2
  %95 = extractvalue { float, float, float, float } %91, 3
  %96 = add nuw nsw i32 %59, 16
  %97 = icmp slt i32 %96, %5
  br i1 %97, label %58, label %38, !llvm.loop !18
```

**注意这里最强的那个信号**：`mma` 的 A 操作数是 `%68, %70, %72, %74`，正好是前面 4 条 `load i32` 的结果；B 操作数是 `%86, %90`，两条 `mov.b32` 的结果；C 是 `%60..%63` 四个 phi；输出经过 `extractvalue` 又回到 `%92..%95`，而它们正是下一轮 `%60..%63` 的 phi 输入。**这个"数据绕一圈回到 phi"的闭环，就是 `D = A*B + C` 的累加形式在 SSA 下的样子。**

再看这条 `br`：`%96 = k0 + 16`，`%97 = (k0+16) < K`，满足则回 `%58`。这就是我们源码里 `for (int k0 = 0; k0 < K; k0 += 16)` 的完整形态——**它还在，没有展开**。为什么不展开？第 5 章我们用三组实验回答。

## 4.5 O0 与 O2 对照：数组去哪了

同一段代码，O0 是这样（`dumps/01-device-O0.ll:262` 附近）：

```llvm
  %199 = load i32, ptr %198, align 4          ; a[0] 从 alloca 里 load
  %200 = load ptr, ptr %8, align 8
  %201 = getelementptr inbounds i32, ptr %200, i64 1
  %202 = load i32, ptr %201, align 4          ; a[1]
  ...
  %214 = load ptr, ptr %10, align 8
  %215 = load float, ptr %214, align 4        ; c[0] 从 alloca 里 load
  ...
  %225 = call contract { float, float, float, float } asm sideeffect "mma.sync..."(...) #7, !srcloc !6
```

O0 里 `uint32_t a[4]` 是**真的数组**：4 个 `alloca`，每个元素算完就 `store` 进去，调 `asm` 之前再从 `alloca` 里 `load` 出来。整个函数光 `alloca` 就有六十多个。

O2 里这些全没了。中间发生的事：

```
O0: alloca + store/load            (796 行)
  │  SROA + mem2reg
  ▼
   标量 SSA 值                      (353 行，实验见第 5 章)
  │  InstCombine / GVN / ...
  ▼
   紧凑的 SSA 形式                  (291 行 = 02-device-O2.ll)
```

这不是"文本压缩"，是**语义上的改变**：`alloca` 代表"内存对象"，`phi`/寄存器代表"值"。前者必须按内存语义处理（可能被取地址、可能越界），后者可以直接进寄存器。第 5 章我们用 `opt` 亲手跑一遍这个过程给你看。

## 4.6 元数据速查

IR 里那些 `!xxx` 不是注释，是机器可读的元数据。我们这份文件里出现的：

| 元数据 | 出现位置 | 含义 |
| --- | --- | --- |
| `!srcloc !17` | 汇编调用后面 | 指向源码位置，`-Rpass` / 调试用 |
| `!tbaa !13` / `!14` | `load` / `store` 后面 | 类型化别名分析信息，告诉优化器这两个访问是不同类型、不会别名 |
| `!llvm.loop !18` | 循环的 `br` 后面 | 循环元数据（unroll/vectorize 指令等都挂在这里） |
| `!llvm.module.flags` | 文件末尾 | 模块级开关（SDK 版本、FTZ、frame-pointer 等） |
| `!llvm.ident` | 文件末尾 | 编译器身份字符串（clang 版本，以及 libdevice 自带的那个 `clang version 3.8.0`） |
| `!nvvmir.version` | 文件末尾 | NVVM IR 版本，`nvcc`/驱动用它判断兼容性 |

老版本文件末尾还有一个 `!nvvm.annotations` 用来标 kernel，**现在已经没有了**——kernel 的身份改由签名上的 `ptx_kernel` calling convention 表达（第 3.2 节）。

`!llvm.loop` 值得单独说：如果你在源码里写 `#pragma unroll 4` 或者 `#pragma nounroll`，clang 会在循环的元数据里挂上 `llvm.loop.unroll.count` / `llvm.loop.unroll.disable`。第 5 章讲 LoopUnroll 时，我们可以手工加这个元数据来做实验——这是 LLVM 调优时最常用的"不改代码改 IR"的技巧。

## 4.7 小结与衔接

本章读完，你应该能独立回答这几个问题：

- **`float c[4]` 去哪了？** 变成了 4 个 `phi` + `extractvalue`，SSA 形式下没有"数组"，只有值。
- **为什么 A 用 `load i32` 而 B 用两条 `load i16`？** fragment 布局决定的，A 的 k 连续、B 的 k 跨行。
- **`inbounds` 什么时候会消失？** 当编译器无法证明偏移不越界时（比如 `&B[k][n]` 加上 `N`）。
- **`disjoint`、`nuw`、`nsw` 这些标记哪来的？** 优化 pass（InstCombine / ValueTracking）推出来的，它们是后端能做更激进优化的依据。

一句话总结：**IR 是"值的世界"，不是"变量的世界"**。你在 C 里写的数组、指针、循环，到了 O2 之后都变形了，但变形是**保语义**的——每一处变形后面都有个 pass 在负责。

下一章我们就去看那些 pass 到底是谁、干了什么、又为什么有些事它们干不了（比如给这个 k 循环展开、比如把 `mma` 挪到别的地方）。
