# 03-3 · 函数签名与属性组

这一讲读函数签名和它后面那一串属性。**属性是 IR 里最容易被跳过、但最有价值的部分**：
它们不产生指令，却决定了优化器敢做什么。

## 一、签名：属性和参数

`dumps/02-device-O2.ll:12`：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(
    ptr noalias nofree noundef readonly  captures(none) %0,
    ptr noalias nofree noundef readonly  captures(none) %1,
    ptr noalias nofree noundef writeonly captures(none) %2,
    i32 noundef %3, i32 noundef %4, i32 noundef %5)
    local_unnamed_addr #0 {
```

我们分两半看：**参数上的属性**和**函数上的属性（`#0`）**。

## 二、参数属性：全是前端推出来的

抛开 `ptx_kernel`（第 02-2 讲）、`dso_local`、`local_unnamed_addr` 这些链接属性，
真正有信息量的是每个参数后面那串：

| 属性 | 来自哪里 | 含义 |
| --- | --- | --- |
| `noalias` | 源码里的 `__restrict__` | 这个指针不会和别的指针指向同一块内存 |
| `readonly` | `const __half *`（参数 0、1） | 只读，不会被写 |
| `writeonly` | `float *D`（参数 2） | 只写，不会被读 |
| `noundef` | C++ 语义 | 传进来的值不能是 undef（未定义值） |
| `nofree` | 分析出来的 | 函数不会释放这个指针 |
| `captures(none)` | 分析出来的 | 函数不会把这个指针存到别处去 |

### `noalias` 和 `readonly` 的价值：它决定了走哪条缓存路径

这两个属性一起，让后端敢把 A、B 的 load 提到很前面、也敢用"非一致加载"。
你会在第 7、8 部分看到结果：

```
ld.global.nc.b32    %r22, [%rd38+-16];     ; nc = non-coherent，只读数据缓存路径
LDG.E.CONSTANT      R13, [R32.64+-0x10] ;  ; SASS 里连 opcode 都写着 NC/CONSTANT
```

**一条源码里的 `__restrict__`，最后变成了机器指令走哪条缓存路径。**
这是"属性不是注释"最有力的证据。

### 一个小变化：`nocapture` 现在写成了 `captures(none)`

老版本这里写的是 `nocapture`。新版 LLVM 把它扩展成了更一般的 `captures(...)`
形式（可以表达"只捕获到某几个参数"这种更细的语义），`captures(none)` 就是"什么都不捕获"。

你在老资料里看到 `nocapture` 不用惊讶，是同一个意思。

## 三、函数属性 `#0`：一大串，逐条说

```llvm
attributes #0 = { convergent mustprogress noinline norecurse nounwind
                  "frame-pointer"="all" "no-trapping-math"="true"
                  "stack-protector-buffer-size"="8"
                  "target-cpu"="sm_89" "target-features"="+ptx87"
                  "uniform-work-group-size" }
```

| 属性 | 谁加的 | 含义 |
| --- | --- | --- |
| `convergent` | CUDA 前端 | 函数里有收敛操作，不许做破坏收敛性的变换 |
| `mustprogress` | C++ 语义（C++11 起默认） | 循环要么终止、要么有副作用 |
| `noinline` | NVPTX 后端 | **kernel 不许被内联进任何调用者** |
| `norecurse` | 分析出来的 | 不递归 |
| `nounwind` | CUDA 默认 | 不抛异常 |
| `"frame-pointer"="all"` | 驱动默认 | 保留帧指针（调试友好） |
| `"no-trapping-math"="true"` | CUDA 默认 | 浮点运算不会触发陷阱（trapping） |
| `"stack-protector-buffer-size"="8"` | 默认 | 栈保护相关 |
| `"target-cpu"="sm_89"` | `--cuda-gpu-arch=sm_89` | 目标 CPU |
| `"target-features"="+ptx87"` | 同上 | PTX ISA 8.7 |
| `"uniform-work-group-size"` | CUDA | block 大小均匀，给优化器的提示 |

### `noinline` 为什么会出现在这个函数上

这是个值得单独讲一下的细节。

在我们的源码里，kernel 是 `__global__` 的；但 `__global__` 只是 CUDA 的语义，
IR 里并没有"这个函数不能被内联"这种属性。**这个 `noinline` 是后端加的**：

因为 kernel 是被**驱动**启动的，不是被别的函数调用的。如果某个 pass 把它内联进调用者，
那它作为独立入口点就没了，宿主就没法启动它了。所以 NVPTX 的流水线会给 kernel
加上 `noinline`。

你可以在 `notes/06-nvptx-backend-1.md` 里看到它的证据：
对比 `01-device-O0.ll`（clang 直接产出，**没有** `noinline`）和
`02-device-O2.ll`（clang 的 `-O2`，也没有）——那它是谁加的？

答案在喂给 ISel 的那份 IR 里（`dumps/26-isel.mir` 头部嵌的那一段）：

```llvm
; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_tc_manual(ptr noalias noundef %0, ...) #0 {
```

它有 `noinline`。**这就证明了 `noinline` 是后端流水线在 codegen 阶段加的。**

（这个例子很好地说明了一件事：**"IR" 不是一份，而是很多份**——
clang 出来的是一份，中间每过几个 pass 又是一份，进入后端时还有一份。
读 IR 时永远要问：**这是哪个阶段的 IR？**）

### `"uniform-work-group-size"` 为什么没有 `="true"`

这是个小的语法细节：新版本的 IR 里，布尔字符串属性可以只写名字，
不写 `="true"`。老版本会写成 `"uniform-work-group-size"="true"`。
两种写法意思一样，看到不用困惑。

## 四、属性怎么"用"

属性不是给编译器看的注释，而是**规则**。举个我们案例里的例子：

因为参数有 `readonly`，优化器知道"这两个指针指向的内存在这函数里不会被写"，
于是它可以：

1. 把 load 提到循环外面（如果地址不依赖循环变量）；
2. 用"非一致加载"（`ld.global.nc`），走只读数据缓存；
3. 做更激进的 CSE（两次读同一地址可以合并）。

而如果某个属性**缺失**（比如你没写 `__restrict__`），这些优化就都不做了，
你会看到 IR 里多出一堆重复 load、PTX 里 `nc` 消失、SASS 里 `CONSTANT` 后缀没了。

**所以"给编译器足够的信息"这件事不是玄学，它直接决定生成什么指令。**

## 五、小结

1. 参数属性来自源码语义（`__restrict__` → `noalias`、`const` → `readonly`），
   它们是"能做什么优化"的依据；`nocapture` 现在写作 `captures(none)`。
2. 函数属性里，`convergent` 来自 CUDA 前端、`noinline` 来自 NVPTX 后端——
   **同一个属性组里的东西，来源可以完全不同**。
3. 属性决定的不只是"能不能优化"，还决定**走哪条缓存路径**
   （`noalias` + `readonly` → `ld.global.nc` → `LDG.E.CONSTANT`）。

## 六、动手题

1. 对比三份 IR 的函数签名，找出 `noinline` 是什么时候出现的：

   ```bash
   grep -n "^attributes #0" dumps/01-device-O0.ll dumps/02-device-O2.ll
   grep -n "^attributes #0" dumps/26-isel.mir
   ```

   三份的差异说明了什么？
2. 把源码里的 `__restrict__` 去掉一个，重新生成 PTX，看看
   `ld.global.nc` 有没有变化。（提示：先去 `code/tc_mma.cu` 备份一份原文件。）

下一讲我们把循环体逐行读一遍——这是整部分的重头戏。
