# 02-8 · host 侧：`<<<grid, block>>>` 变成了什么

前面七讲都在讲设备端。这一讲我们把镜头转过来，看看 **host 那一遍编译**产出了什么。

为什么值得一讲？因为"kernel 怎么被启动"这件事，直接解释了
**为什么 kernel 函数名必须保持可见**，以及**为什么手工写 IR 时不能忘了 `ptx_kernel`**。

## 一、先拿到 host 侧的 IR

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 --cuda-host-only \
        -O0 -S -emit-llvm -o dumps/12-host-O0.ll code/tc_mma.cu
```

注意 `--cuda-host-only`：只跑 host 那一遍。产出的 IR 里：

- 三元组是 `x86_64-unknown-linux-gnu`（不是 nvptx64）；
- **两个 kernel 函数都不在里面**（它们是设备代码）；
- 取而代之的是 `main`、两个 stub、以及三个 CUDA runtime 调用。

## 二、`<<<>>>` 被拆成了三部分

源码里这一行：

```cpp
mma_tc_manual<<<grid, block>>>(dA, dB, dD, M, N, K);
```

被拆成三部分：**调用点**、**stub 函数**、**runtime 调用**。

### 第一部分：调用点做的是"存配置"

在 `main` 里（`dumps/12-host-O0.ll:385` 附近）：

```llvm
%151 = call i32 @__cudaPushCallConfiguration(i64 %144,   ; gridDim，打包成 64 位
                                            i32 %146,    ; gridDim 的高半/低半
                                            i64 %148,    ; blockDim，打包
                                            i32 %150,
                                            i64 noundef 0,      ; sharedMem = 0
                                            ptr noundef null)   ; stream = 默认流
call void @__device_stub__mma_tc_manual(ptr %154, ptr %155, ptr %156,
                                        i32 noundef 256, i32 noundef 128,
                                        i32 noundef 64) #9
```

两件事：

1. `__cudaPushCallConfiguration` 把 `<<<grid, block, sharedMem, stream>>>`
   这四个参数**存到一个线程本地的配置槽里**。
2. 然后调用 **自动生成的 stub**，把 kernel 的参数原样传进去。

为什么 `gridDim` 和 `blockDim` 各占两个参数（`i64` + `i32`）？
因为 `dim3` 有 x/y/z 三个字段，clang 把它们打包进一个 64 位整型（`i64`），
外加一个 `i32`。这是我们不需要深究的 ABI 细节，知道"是打包传递"就够了。

### 第二部分：stub 负责"把参数摆成数组"

clang 会为每个 kernel 自动生成一个 `__device_stub__<名字>` 函数
（`dumps/12-host-O0.ll:34`）：

```llvm
define dso_local void @__device_stub__mma_tc_manual(
    ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2,
    i32 noundef %3, i32 noundef %4, i32 noundef %5) #0 {
  ...
  %26 = call i32 @__cudaPopCallConfiguration(ptr %13, ptr %14, ptr %15, ptr %16)
  ...
  %37 = call noundef i32 @cudaLaunchKernel(ptr noundef @__device_stub__mma_tc_manual,
                                           i64 %30, i32 %32,   ; gridDim
                                           i64 %34, i32 %36,   ; blockDim
                                           ptr noundef %19,    ; kernelParams
                                           i64 noundef %27,    ; sharedMemBytes
                                           ptr noundef %28)    ; stream
```

它干的事：

1. `__cudaPopCallConfiguration` 把刚才存的配置取回来；
2. 把 6 个参数**一个个拷进一个数组**（`kernelParams`）——因为 `cudaLaunchKernel`
   只接受 `void**` 这种"参数数组"的形式；
3. 调 `cudaLaunchKernel`，把 kernel 函数的**地址**传进去。

### 第三部分：runtime 去找真正的 kernel

```llvm
declare i32 @cudaLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)
```

这是 CUDA runtime 的 API。它拿到 kernel 函数地址后，去 fatbinary / cubin 里
找对应的设备函数。

## 三、于是"kernel 名字必须可见"这件事就解释清楚了

看那个关键参数：

```llvm
ptr noundef @__device_stub__mma_tc_manual
```

**host 代码需要拿到 kernel 的地址。** 这意味着：

1. kernel 函数**不能被优化掉**（哪怕没人调用它）；
2. 它的名字**必须能在符号表里找到**（所以 `extern "C"` 很实用，
   而且别加 `static` 之类的链接属性）；
3. 设备端那边的 cubin 里，必须有同名（经过 mangling 之后）的 `.entry`。

这就把第 02-2 讲的那个坑解释清楚了：**如果你手工写 IR 时忘了 `ptx_kernel`，
设备端生成的是 `.func` 而不是 `.entry`，`cudaLaunchKernel` 就找不到它。**

## 四、一个常见的误解：`<<<>>>` 不是语法糖那么简单

很多人以为 `a<<<g,b>>>(args)` 就是"调一个函数"。从 IR 上看，它其实是：

```
push 配置（grid/block/shared/stream）
   ↓
调 stub（把参数打包成数组）
   ↓
cudaLaunchKernel（异步启动，不等待）
```

三个后果值得记住：

1. **它是异步的**：`<<<>>>` 返回时 kernel 还没跑完。我们的案例里必须
   `cudaDeviceSynchronize()` 才能读结果——`build-and-run.sh` 里就是这么做的。
2. **参数是"拷贝"进去的**：stub 会把参数复制到 `kernelParams` 数组里。
   所以你不能把"临时对象的地址"传进 kernel 然后期望它有效。
3. **kernel 名是"运行时字符串"**：驱动是靠符号名找函数的，
   所以 mangling、链接属性这些"编译期细节"会影响运行期行为。

## 五、两遍编译是怎么合起来的

我们再回到整体看一眼。`clang++ tc_mma.cu -o tc_mma` 的时候，clang 其实是：

```
1. host 那一遍     → 生成 host 目标文件（含 main、stub、cudaLaunchKernel 调用）
2. device 那一遍   → 生成 PTX → ptxas 汇编成 cubin
3. 把 cubin 打包进 fatbinary（一段嵌在可执行文件里的数据）
4. 链接 host 目标文件 + libcudart + fatbinary
```

而 `--cuda-device-only` / `--cuda-host-only` 就是让你"只要其中一段"。
我们这门课研究的是第 2 步的产物；第 1、3、4 步只要知道它们存在就够了。

## 六、小结

1. host 那一遍产出的是：`main` + 每个 kernel 一个 `__device_stub__<名字>` +
   `__cudaPushCallConfiguration` / `__cudaPopCallConfiguration` / `cudaLaunchKernel`。
2. `<<<grid, block>>>` 的语义是"**存配置 → 打包参数 → 异步启动**"，
   它不是一个普通函数调用。
3. 正因为 `cudaLaunchKernel` 要拿 kernel 的**地址**，kernel 必须：名字可见、
   不被优化掉、且在设备端生成 `.entry`（也就是必须带 `ptx_kernel`）。

到这里，Clang 前端这一部分就讲完了。我们把设备端和 host 端的账都算清楚了。

下一部分我们进入 IR 本身：把 `dumps/02-device-O2.ll` 从头到尾读一遍，
搞清楚每一行为什么长这样。

## 七、动手题

1. 在 `dumps/12-host-O0.ll` 里找出 `__device_stub__mma_tc_ldmatrix`，
   数一数它内部有多少条指令，和 `__device_stub__mma_tc_manual` 比一比。
   （提示：kernel 参数一样多，所以应该差不多。）
2. 想一想：为什么 `--cuda-device-only` 生成的 IR 里**没有** `cudaLaunchKernel`
   这些东西？（提示：设备端不需要"启动自己"。）
