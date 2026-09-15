# 第 2 部分 · Clang 前端：CUDA 的东西怎么变成 LLVM IR

前一部分我们把案例和 fragment 布局钉死了。从这一部分开始，我们正式进入编译链路，
**第一站是 Clang 前端**。

这一部分要回答的问题只有一个，但可以拆成八个小问题：

> 我写的那几行 CUDA 特有的东西，在 LLVM IR 里到底变成了什么？

具体是这些：

```
__global__ void k(...)   → ？
threadIdx.x / blockIdx.y → ？
__shared__ __half As[]   → ？
__syncthreads()          → ？
asm volatile("mma...")   → ？为什么它带了 convergent？
__half / __halves2half2  → ？
kernel<<<grid, block>>>  → ？（host 侧）
```

每一讲解决一两个，配真实 IR 和真实命令。

| 讲 | 标题 | 你会拿到什么 |
| --- | --- | --- |
| [02-1](02-1-cc1-command-line.md) | 一次编译、两套前端 | clang 的 `-cc1` 命令行逐段解读 |
| [02-2](02-2-kernel-and-params.md) | kernel 的身份与参数 | `ptx_kernel`、参数拆成标量、kernel 标记的变迁 |
| [02-3](02-3-address-spaces.md) | 地址空间：CUDA 内存模型落到 IR | `addrspace(1/3/101)`、`__shared__` 提成全局变量 |
| [02-4](02-4-sreg-intrinsics.md) | `threadIdx` / `blockIdx` 与 sreg | intrinsic 命名规律、那些"占位全局变量" |
| [02-5](02-5-syncthreads.md) | `__syncthreads()` 与 barrier | intrinsic 改名、为什么必须 convergent |
| [02-6](02-6-inline-asm.md) | 内联汇编的三层信息 | 约束字符串、返回结构体、`convergent` 的来源与摘除 |
| [02-7](02-7-half-glue.md) | `__half` 的胶水代码 | `mov.b32` 打包、`__cvta_generic_to_shared` 的真身 |
| [02-8](02-8-host-side.md) | host 侧：`<<<>>>` 变成了什么 | `__cudaPushCallConfiguration` + stub + `cudaLaunchKernel` |
