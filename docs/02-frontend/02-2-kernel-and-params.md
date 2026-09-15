# 02-2 · kernel 的身份与参数

上一讲我们看清了 clang 怎么把一份 `.cu` 分成两遍来编。这一讲我们看设备端口那一遍的
第一件事：**`__global__ void mma_tc_manual(...)` 这个函数，在 IR 里长什么样。**

看起来是个很小的问题，但它牵扯到一个很有意思的事实：**"什么是一个 kernel"这件事，
LLVM 的表达方式换过好几茬。**

## 一、先看现状：IR 里 kernel 长什么样

打开 `dumps/01-device-O0.ll`（未优化版本，更适合看原始形态），
搜 `mma_tc_manual`，第一条就是它：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(ptr noalias noundef %0,
                                                ptr noalias noundef %1,
                                                ptr noalias noundef %2,
                                                i32 noundef %3, i32 noundef %4,
                                                i32 noundef %5) #0 {
```

优化之后（`dumps/02-device-O2.ll`）属性变多了，但那个关键字还在：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(
    ptr noalias nofree noundef readonly captures(none) %0, ...
) local_unnamed_addr #0 {
```

**`ptx_kernel` 就是这一讲的主角。** 它是 NVPTX 目标注册的一个 calling convention
（调用约定）。在 LLVM 的源码里，它对应 `CallingConv::PTX_Kernel`。

## 二、为什么叫"calling convention"？这就是"身份"的表达方式

在 LLVM IR 里，函数的 calling convention 位置本来就用来表达"这个函数应该怎么被调用"。
普通 C 函数是 `ccc`（默认，通常省略不写）、Windows 上有 `x86_stdcallcc`，
还有 `fastcc`、`coldcc` 等等。

NVPTX 目标就往这个位置上注册了一个自己的：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(...)   ; 这是个 kernel，能被 <<<>>> 启动
define dso_local void       @mma_tc_manual(...)        ; 这只是个普通设备函数
```

**一句话：`ptx_kernel` 就是"这是个 kernel"的权威标记。**

## 三、有意思的地方：这个标记以前不是这么表达的

这里我要停一下，因为这是本课里"版本差异"最好的一个例子。

### 老写法（我们原来的笔记里就是这么写的）

```llvm
define dso_local void @mma_tc_manual(...)    ; 看起来就是个普通函数

!nvvm.annotations = !{!4, !5}                ; kernel 的身份藏在元数据里
!4 = !{ptr @mma_tc_manual, !"kernel", i32 1}
!5 = !{ptr @mma_tc_ldmatrix, !"kernel", i32 1}
```

老版本里，函数本身看起来跟普通设备函数没区别，**"我是 kernel"这件事写在模块末尾的
`!nvvm.annotations` 里**。

### 新写法（本机这一版）

```llvm
define dso_local ptx_kernel void @mma_tc_manual(...)
define dso_local ptx_kernel void @mma_tc_ldmatrix(...)
```

而 `!nvvm.annotations` 呢？你来验证一下：

```bash
$ grep -c "nvvm.annotations" dumps/01-device-O0.ll
0
```

**它已经彻底消失了。** kernel 的身份完全由签名上的 `ptx_kernel` 承担。

### 这个变化对读代码的人意味着什么

三条实际的差别：

1. **写 IR 的人**：你手工写 IR 做实验时，不再需要写元数据了，
   但**必须在签名上写 `ptx_kernel`**。忘了写，函数就只是普通设备函数，
   宿主代码启动不了它（第 02-8 讲会解释为什么）。
2. **做分析的人**：如果想找"这个模块里有哪些 kernel"，
   老办法是查元数据，新办法是遍历函数看 calling convention。
3. **优化行为**：元数据是"可以被优化掉的附加信息"，而 calling convention 是
   函数类型的一部分。这也是为什么优化之后 `ptx_kernel` 还在，
   而 `!nvvm.annotations` 在新版本里干脆不生成。

顺便说：你如果读老文章或者老版本的 LLVM 输出，看到 `!nvvm.annotations`
不要觉得奇怪，那是历史的写法。**这份教材后面的所有内容都以 `ptx_kernel` 为准。**

## 四、参数：6 个标量，一个不多一个不少

看签名里的参数：

```cpp
__global__ void mma_tc_manual(const __half *A, const __half *B,
                              float *D, int M, int N, int K)
```

```llvm
define dso_local ptx_kernel void @mma_tc_manual(
    ptr noalias noundef %0,      ; A
    ptr noalias noundef %1,      ; B
    ptr noalias noundef %2,      ; D
    i32 noundef %3,              ; M
    i32 noundef %4,              ; N
    i32 noundef %5)              ; K
```

6 个参数原样搬过来，每个都是一个标量。这里有三个可以讲的点：

**第一，指针参数是 `ptr`（地址空间 0，generic），不是 `ptr addrspace(1)`。**

为什么？因为 host 侧 `cudaMalloc` 拿到的指针在设备端是"通用地址"，
CUDA 允许你把它当 generic 指针用，需要的时候再自己 `cvta` 转换。
所以前端统一按 generic 处理。**这个 generic → global 的转换发生在后端**，
第 6 部分会详细讲（那是一段很漂亮的"地址空间推断"的故事）。

**第二，`noalias` 是从 `__restrict__` 来的。**

我们源码里写的是 `const __half *__restrict__ A`。这个 `__restrict__` 不是装饰，
它会变成 IR 参数属性里的 `noalias`，意思是"这两个指针不会指向同一块内存"。
后端据此可以放心地把 A 的 load 提到前面去——第 8 部分你会看到
`LDG.E.CONSTANT`（走只读缓存路径）就是这么来的。

**第三，`M` 这个参数其实没被用到。**

看我们的 kernel：v1 里用了 `N` 和 `K`（算地址），但 `M` 什么都没干。
那为什么它还留在签名里？

因为 **clang 不会替你去删参数**。函数签名是 `extern "C"` 的 ABI 约定，
而且 host 侧的 `<<<>>>` 启动要用 `M` 这个实参去对位。所以它会一路留到最后。

你可以在优化后的 IR 里验证这一点：数一数 `%3`（也就是 M）出现了几次。
**这是初学者读 IR 时一个很容易迷惑的点**：参数出现在签名里，不代表函数体里用它。

## 五、`extern "C"` 为什么要写

我们的 kernel 是这么声明的：

```cpp
extern "C" __global__ void mma_tc_manual(...)
```

如果不写 `extern "C"`，C++ 会对函数名做名称修饰（mangling），
你会在 IR 里看到这样的东西：

```llvm
define dso_local ptx_kernel void @_Z14mma_tc_manualPK6__halfS1_Pfiii(...)
```

而对照的 PTX 和 SASS 里也会是这串名字：

```
.visible .entry _Z14mma_tc_manualPK6__halfS1_Pfiii
```

不是不能用，但是**读起来非常痛苦**——尤其是第 8 部分读 SASS 的时候，
你会在 `cuobjdump` 的输出里找函数边界。所以本课的 kernel 一律写 `extern "C"`。

（例外情况：如果你在做 C++ 模板 kernel，那 mangling 是必须的，
因为不同的模板实例本来就是不同的 kernel。）

## 六、从 IR 到 PTX：`ptx_kernel` 变成了什么

这条信息最终会落到 PTX 的函数声明上：

```
.visible .entry mma_tc_manual(      ← kernel：可以被 host 启动
	.param .u64 .ptr .align 1 mma_tc_manual_param_0, ...
)
```

而普通的 `__device__` 函数会变成 `.func`：

```
.visible .func (.param .b32 func_retval0) arch_id()   ← 普通设备函数
```

**`.entry` 和 `.func` 的区别，就是 `ptx_kernel` 与否的区别。**
你在 cubin 里也能看到它：kernel 会被放进 `.text.<名字>` 段，
并在 `.nv.info` 里留下记录，宿主驱动就是靠这些信息找到它能启动哪些函数。

## 七、一个真实的坑：手工写 IR 时忘了 `ptx_kernel`

我们做实验经常手工写 IR。如果你忘了写 `ptx_kernel`：

```llvm
define dso_local void @my_kernel() {    ; 忘了 ptx_kernel
  ret void
}
```

编译**不会报错**，它会老老实实生成一个 `.func`：

```
.visible .func () my_kernel()
```

然后你拿这个 cubin 去启动，会得到"找不到符号"之类的运行时错误。
**"IR 能编译"和"这东西能被启动"是两件事**——这和我们在第 9 部分要讲的
intrinsic 名字坑是一个性质的问题。

## 八、小结

1. kernel 的身份写在**函数签名上**：`define dso_local ptx_kernel void @name(...)`。
   老版本的 `!nvvm.annotations` 已经不再生成。
2. 参数会原样搬成 6 个标量；`__restrict__` 变成 `noalias`；
   **签名里出现的参数不代表函数体里会用**（比如我们的 `M`）。
3. 写 `extern "C"` 能让名字可读；从 IR 到 PTX 的映射是
   `ptx_kernel` → `.visible .entry`，普通函数 → `.visible .func`。

## 九、动手题

1. 用下面两条命令各跑一次，比较一下：

   ```bash
   grep -c "ptx_kernel" dumps/01-device-O0.ll
   grep -cE "^\s*\.visible\s+\.entry" dumps/03-clang-O2.ptx
   grep -cE "^\s*\.visible\s+\.func"  dumps/03-clang-O2.ptx
   ```

   IR 里有几个 `ptx_kernel`？PTX 里有几个 `.entry`？对得上吗？
2. 在 `dumps/02-device-O2.ll` 里搜 `%3`（也就是 `M`）。它出现了几次？
   分别在哪？想一想：如果我想让 `M` 完全从代码里消失，该怎么改源码？

下一讲我们讲地址空间——CUDA 的内存模型是怎么落到 IR 里的。
这是 CUDA 前端在 IR 上留下的最深的脚印。
