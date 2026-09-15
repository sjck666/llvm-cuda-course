# 03-5 · 元数据速查与 O0/O2 对照

第 3 部分最后一讲。我们做两件事：**把 IR 里那些 `!xxx` 说清楚**，
然后**把 O0 和 O2 摆在一起**，看数组到底是怎么消失的。

## 一、元数据不是注释

IR 里到处都是 `!` 开头的东西。它们不是"给人看的注释"，而是**机器可读的附加信息**，
可以影响优化和代码生成。

我们这份文件里出现的几类：

| 元数据 | 出现在哪 | 含义 |
| --- | --- | --- |
| `!srcloc !17` | 汇编调用后面 | 指向源码位置，调试和 `-Rpass` 用 |
| `!tbaa !13` / `!14` | `load` / `store` 后面 | **类型化别名分析**：告诉优化器这些访问的类型不同、不会互相别名 |
| `!llvm.loop !18` | 循环的 `br` 后面 | **循环元数据**：unroll / vectorize 的指令都挂在这里 |
| `!llvm.module.flags` | 文件末尾 | 模块级开关（SDK 版本、FTZ、frame-pointer 等） |
| `!llvm.ident` | 文件末尾 | 编译器身份字符串 |
| `!nvvmir.version` | 文件末尾 | NVVM IR 版本，驱动用它判断兼容性 |

### 元数据里最有用的一类：`!llvm.loop`

如果你在源码里写 `#pragma unroll 4` 或者 `#pragma nounroll`，
clang 会在循环的元数据上挂 `llvm.loop.unroll.count` / `llvm.loop.unroll.disable`。
我们案例里只有默认的：

```llvm
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
```

`distinct` 表示"这个循环元数据是这个循环独有的"（不能和别人合并）；
`mustprogress` 对应 C++ 语义里的"必须向前推进"。

**手工改元数据是调优时很常用的手段**：不改源码，只改 IR，
看某个 pragma 到底有没有效果。第 4 部分讲循环展开时就靠它做实验。

### 一个消失了的元数据：`!nvvm.annotations`

老版本的 IR 末尾有：

```llvm
!nvvm.annotations = !{!4, !5}
!4 = !{ptr @mma_tc_manual, !"kernel", i32 1}
```

**这一版不生成它了**，kernel 的身份改由签名上的 `ptx_kernel` 表达（第 02-2 讲）。
你可以自己验证：

```bash
grep -c nvvm.annotations dumps/01-device-O0.ll    # 0
```

文件末尾现在长这样：

```llvm
!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!nvvmir.version = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 8]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 24.0.0git (ssh://git@... 677a4c33...)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}   ← 这条很有意思，见下
!5 = !{i32 2, i32 0}
```

**`!4` 那条 `clang version 3.8.0` 是从哪来的？** 它是 **libdevice 自带的**标识——
第 02-1 讲说过，clang 会把 NVIDIA 提供的 libdevice bitcode 链进你的模块，
而那份 bitcode 是用 clang 3.8 编的，它的 ident 就跟着进来了。
**这是"IR 里能看出外部输入"的一个有趣例子。**

## 二、O0 vs O2：一次完整的"数组消失"复盘

现在把两份 IR 摆在一起看。

### 先看规模

```bash
wc -l dumps/01-device-O0.ll dumps/02-device-O2.ll
```

```
 796 dumps/01-device-O0.ll
 291 dumps/02-device-O2.ll
```

**少了 500 多行。** 但这些行不是"被压掉"的，是**语义变了**。

### O0 里的数组是"真的数组"

O0 的 `mma_tc_manual` 一开头是一串 `alloca`：

```llvm
  %7  = alloca %struct.__half2, align 4
  %8  = alloca %struct.__half, align 2
  %9  = alloca %struct.__half, align 2
  %10 = alloca %struct.__half2, align 4
  ...
```

用的时候要先 `load` 出来：

```llvm
  %199 = load i32, ptr %198, align 4          ; a[0] 从 alloca 里 load
  %215 = load float, ptr %214, align 4        ; c[0] 从 alloca 里 load
  ...
  %225 = call contract { float, float, float, float } asm sideeffect "mma.sync..."(...) #7, !srcloc !6
```

**光 `mma_tc_manual` 这一个函数里就有 53 条 `alloca`**（另一个 kernel 46 条）。
这就是"内存世界"：
每个变量都是栈上一块内存，每次用都要 load，算完再 store 回去。

### O2 里它们全消失了

O2 里没有一条 `alloca`（你可以 `grep -c alloca` 验证）。
A 的四次访问变成四条 `load i32`，直接读 global；
`c[4]` 变成四个 phi；`a[4]`、`b[2]` 这两个临时数组彻底不存在了。

### 中间发生了什么

```
O0: alloca + store/load                    (796 行)
  │  SROA + mem2reg
  ▼
   标量 SSA 值                              (353 行)
  │  InstCombine / GVN / ...
  ▼
   紧凑的 SSA 形式                          (291 行)
```

（353 行那个数字来自第 4 部分的手工实验：用 `opt -passes='sroa,mem2reg'`
单独跑一遍，得到 `dumps/15-after-sroa-mem2reg.ll`。）

**这不是"文本压缩"，是语义上的改变**：

```
alloca 代表"内存对象"  —— 可能被取地址、可能越界，必须按内存语义处理
phi/SSA 值代表"值"     —— 可以直接进寄存器，优化自由度大得多
```

### 一个可以直接观察的痕迹

在 `dumps/15-after-sroa-mem2reg.ll` 里，你会看到这样的名字：

```
float %.sroa.097.0, float %.sroa.499.0, ...
```

**`.sroa.` 前缀就是"这个值是从数组里拆出来的"的痕迹。**
再过几个 pass，这些名字也会被丢掉，变成普通的 `%xx`。

读 IR 时看到 `.sroa.`、`.mem2reg` 之类的名字，你就知道"这里刚刚发生过什么变换"。

## 三、小结

1. IR 里的 `!xxx` 是机器可读的元数据：`!llvm.loop` 管循环优化、
   `!tbaa` 管别名分析、`!srcloc` 管调试；`!nvvm.annotations` **已经不再生成**。
2. 文件末尾那条 `clang version 3.8.0` 是 libdevice 自带的 ident——
   一个"从 IR 里能看出外部输入"的有趣证据。
3. O0 → O2 的核心变化是"**从内存世界搬到值世界**"：
   `alloca` + load/store 变成 SSA 值 + phi；796 行 → 291 行不是压缩，是语义改变。

到这里第 3 部分结束了。你现在应该能做到：**拿到任何一份 `.ll`，
五分钟内说出它大致在干什么、哪些是源码的痕迹、哪些是优化器的手笔。**

下一部分我们进入中端：**这些变化到底是哪些 pass 干的，它们又为什么有时收手。**

## 四、动手题

1. 验证一下 O2 里确实没有 `alloca`：

   ```bash
   grep -c alloca dumps/01-device-O0.ll        # 很多
   grep -c alloca dumps/02-device-O2.ll        # 0
   grep -c '\.sroa\.' dumps/15-after-sroa-mem2reg.ll   # SROA 的痕迹
   ```
2. 找出 `!llvm.loop` 链向的元数据内容，说说 `mustprogress` 在 C++ 语义里
   对应什么。（提示：想想"编译器凭什么假设循环一定会结束"。）
