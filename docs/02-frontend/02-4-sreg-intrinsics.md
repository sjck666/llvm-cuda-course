# 02-4 · `threadIdx` / `blockIdx` 与 sreg

这一讲讲一件很朴素的事：**你写的 `threadIdx.x`，在 IR 里变成了什么。**

答案很短：变成了一次 intrinsic 调用。但这里有几个细节值得单独讲一讲，
其中一个是"你会在 IR 里看到几个你可能想删掉的全局变量，千万别删"。

## 一、`threadIdx.x` 在 IR 里长这样

打开 `dumps/01-device-O0.ll`，找 kernel 开头的几行（第 79 行附近）：

```llvm
%60 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%61 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
%63 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
```

分别对应源码里的：

```cpp
const int lane   = threadIdx.x;        // → .tid.x
const int tile_m = blockIdx.y * 16;    // → .ctaid.y（乘 16 在 IR 里另算）
const int tile_n = blockIdx.x * 8;     // → .ctaid.x
```

注意 `blockIdx.y * 16` 里那个乘 16 **不在 intrinsic 里**，它是后面一条
`shl i32 %8, 4`（左移 4 位等于乘 16）。intrinsic 只负责"把那个数读出来"。

## 二、命名规律：`llvm.nvvm.read.ptx.sreg.<特殊寄存器名>`

这些 intrinsic 的名字是机械拼接出来的：

```
llvm.nvvm.read.ptx.sreg.tid.x
└────┬───┘ └─┬─┘ └┬┘ └──┬──┘
     │       │    │      └── PTX 里的特殊寄存器名：tid.x
     │       │    └───────── 读操作
     │       └────────────── PTX 这个虚拟 ISA
     └────────────────────── NVVM（NVIDIA 给 LLVM 的扩展）
```

常用的一张对照表（你以后写 IR 或者读 IR 都用得上）：

| CUDA 里写的 | IR intrinsic | PTX 里最终是 |
| --- | --- | --- |
| `threadIdx.x` | `llvm.nvvm.read.ptx.sreg.tid.x()` | `%tid.x` |
| `threadIdx.y` | `...sreg.tid.y()` | `%tid.y` |
| `blockIdx.x` | `...sreg.ctaid.x()` | `%ctaid.x` |
| `blockIdx.y` | `...sreg.ctaid.y()` | `%ctaid.y` |
| `blockDim.x` | `...sreg.ntid.x()` | `%ntid.x` |
| `gridDim.x` | `...sreg.nctaid.x()` | `%nctaid.x` |
| （warp 内编号） | `...sreg.laneid()` | `%laneid` |
| （warp 编号） | `...sreg.warpid()` | `%warpid` |

**注意命名的不对称**：`blockIdx` 变成了 `ctaid`（**CTA** ID，CTA 是 block 的正式叫法），
`blockDim` 变成 `ntid`（number of threads in block），`gridDim` 变成 `nctaid`。
这套名字是 PTX 的约定，不是 LLVM 自己编的。

## 三、这些 intrinsic 是"纯值读取"

看它们的函数属性（`dumps/01-device-O0.ll` 里的 `attributes #4`）：

```llvm
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
```

重点是 **`memory(none)`**（等价于老写法的 `readnone`）：

> 这个函数**既不读内存也不写内存**。

这个信息对优化器非常有用。它意味着：

- 读 `threadIdx.x` 两次，等价于读一次（可以做 CSE）；
- 读 `threadIdx.x` 和"往 global memory 写数据"之间没有顺序依赖（可以重排）；
- 它可以被当成一个"纯值"搬来搬去。

你能在 IR 里验证这一点：看 `-O2` 之后的 IR，`%7` 这个值被用了很多次，
而不是每次都重新调一次 intrinsic。**这就是"纯值"带来的好处。**

## 四、别被那几个全局变量骗了

现在看一个容易让人困惑的东西。在 `dumps/01-device-O0.ll` 的第 13、14 行：

```llvm
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@blockIdx  = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t,  align 1
```

你可能会想："这不是说 `threadIdx` 是个全局变量吗？那前面那些 intrinsic 是怎么回事？"

**这俩是"占位符"，不是真在用的东西。** 三个特征说明它们只是摆设：

1. **`extern_weak`**：弱外部符号声明——如果没人定义它，链接时就当它不存在；
2. **`addrspace(1)`**：声明在全局内存空间里（历史上 CUDA 确实提供过这样的全局变量）；
3. **空结构体类型**：`%struct.__cuda_builtin_threadIdx_t` 只有 `x/y/z` 字段，
   但从来不会被真正读写。

为什么还要声明它们？因为 CUDA 头文件里 `threadIdx` 这个名字是**有地址的**
（理论上你能写 `&threadIdx`），有些老代码依赖这一点。
clang 的做法是：**类型上留着声明，取值时直接走 sreg intrinsic**。

你可以验证：在 `dumps/01-device-O0.ll` 里搜 `@threadIdx`，只会搜到那一行声明，
**没有任何一条指令引用它**。

> **实践建议**：读 IR 时看到这类 `extern_weak` 的占位全局变量，直接跳过。
> 真正干活的是那些 `llvm.nvvm.read.ptx.sreg.*` 调用。

## 五、一路向下：从 IR 到 SASS

这条链很短，我们把它走完（第 6、7、8 部分会重复用到，这里先给预告）：

| 层 | 形态 |
| --- | --- |
| CUDA C++ | `threadIdx.x` |
| LLVM IR | `call i32 @llvm.nvvm.read.ptx.sreg.tid.x()` |
| Machine IR（ISel 之后） | `%3:b32 = INT_PTX_SREG_TID_x` |
| PTX | `mov.u32 %r1, %tid.x;` |
| SASS | `S2R R0, SR_TID.X ;` |

一路看下来你会发现：**每一层都对得上，而且越往下越"具体"**。
IR 里一次"函数调用"，到机器层变成了一条读特殊寄存器的伪指令，
最后在 SASS 里变成一条真正的硬件指令 `S2R`（**S**ource **R**egister read）。

顺带说一句：`INT_PTX_SREG_TID_x` 里的 `SREG` 就是 "special register"。
第 6 部分会看到，这一版 NVPTX 后端里只有 sreg 读取和 barrier 还保留着
`INT_PTX_` 前缀——其它访存指令都改名了。

## 六、小结

1. `threadIdx` / `blockIdx` / `blockDim` / `gridDim` 都编译成
   `llvm.nvvm.read.ptx.sreg.*` 的 intrinsic 调用，名字后段直接用 PTX 的 sreg 名
   （`tid.x`、`ctaid.x`、`ntid.x`、`nctaid.x`…）。
2. 这些 intrinsic 是 `memory(none) speculatable` 的**纯值读取**，
   所以优化器可以随意复制、合并、重排它们。
3. IR 里那几个 `@threadIdx` / `@blockIdx` 全局变量是**占位符**（`extern_weak`），
   没有任何指令引用它们，读 IR 时直接跳过。

## 七、动手题

1. 在 IR 里找出"`blockIdx.y * 16`"这个乘法：

   ```bash
   grep -n "ctaid.y" dumps/01-device-O0.ll | head -3
   grep -n "ctaid.y" dumps/02-device-O2.ll | head -3
   ```

   O0 里 intrinsic 的返回值后面跟着什么？O2 里它被简化成了什么？
2. 想一想：如果把 `threadIdx.x` 存到一个 `int` 变量里，然后把这个变量改掉：

   ```cpp
   int t = threadIdx.x;
   t = 100;
   ```

   改完之后 `t` 还是"纯值"吗？这对优化有影响吗？
   （这题没有标准答案，你能说清"哪个值是可优化的纯值、哪个不是"就行。）

下一讲我们看同步：`__syncthreads()` 变成了什么，以及为什么它身上带着一个
会一路影响到 SASS 的标记。
