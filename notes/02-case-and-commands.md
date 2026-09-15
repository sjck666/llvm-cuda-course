# 第 2 章 · 案例代码与编译命令

## 本章你要拿到的东西

三样东西，缺一不可：

1. 一个**能编译、能跑、算得对**的 Tensor Core 矩阵乘 kernel（`code/tc_mma.cu`）；
2. 一张**fragment 布局表**——这是个地基，布局搞错了，第 8、9 章你看 `HMMA` 的操作数会一头雾水；
3. 一整套**编译 / dump 命令**，后面每一章都从这套命令产出的文件里取材料。

## 2.1 案例是怎么设计的，以及为什么这么设计

先把需求摆清楚。我们要的不是一个"快"的 GEMM，而是一个**足够小、又完整覆盖 Tensor Core 数据通路**的 GEMM。

### 为什么选 `m16n8k16`

Ada 这一代（sm_89）上，f16 输入的 mma 有好几种形状，我们挑 `m16n8k16`：

- 它是 sm_80 引入的经典形状，sm_89 完整支持，PTX ISA 8.x 里文档最全；
- 一个 warp 一条指令算 16×8 的输出、吃 16 深度的 K，**四个 fragment 的寄存器数刚好是 4 / 2 / 4**（A 用 4 个 `.b32`，B 用 2 个 `.b32`，C/D 用 4 个 `.f32`），寄存器账很好算；
- 对应的 SASS 是 `HMMA.16816.F32`，第 9 章可以直接逐字段拆。

顺带说一句，sm_89 还支持 `mma.m16n8k16` 的 `bf16`、`tf32` 变体，以及 `mma.m16n8k8.f16`。形状小的（k8）每线程持有的元素更少，形状大的（k16）一个 warp 干的活更多。选 k16 是因为它在"寄存器数量"和"指令数量"之间最平衡，最容易把账算清楚。

### 为什么写两个 kernel

```cpp
mma_tc_manual    // v1: 手动从 global memory 组装 fragment，不碰 shared memory
mma_tc_ldmatrix  // v2: cp.async 搬 tile 到 shared memory + ldmatrix 装 fragment
```

这不是为了炫技，是因为这两条路径在**编译链路里长得完全不一样**：

- v1 里 fragment 是普通的整数/浮点标量，`mma` 是唯一一条"带外挂语义"的汇编；你看到的 IR 就是普通的 load + `call asm`。
- v2 里出现了 `__shared__`（地址空间 3）、`cp.async`、`ldmatrix`、`__syncthreads`（`llvm.nvvm.barrier.cta.sync.aligned.all`），**同步和地址空间**这两件事只有 v2 才能讲清楚。

后面第 3 章讲地址空间、第 8 章讲 PTX、第 9 章讲 `LDSM` 与 `LDGSTS`，都要靠 v2 撑场面。

### 为什么 M/N/K 取 256/128/64

三个约束：

- 必须满足 `M % 16 == 0`、`N % 8 == 0`、`K % 16 == 0`——这样 fragment 加载不用做边界判断，IR 干净，SASS 也短；
- `K = 64` 让 k 循环正好 4 圈（每圈 16），第 5 章讲循环展开时你能直接数出圈数；
- 矩阵小到能在 CPU 上跑一份 fp32 参考实现做对拍，省得你怀疑"是不是我 matrix 太大了所以对不上"。

于是 grid = `dim3(N/8, M/16)` = `(16, 16)`，block = `dim3(32)`——**一个 warp 一个 block，一个 block 算一个 16×8 的 tile**。慢，但是账目清楚。

### 为什么不用 CUTLASS / CuTe

因为我们这门课要看的是 `mma` 从 C++ 一路变成 `HMMA` 的过程。CUTLASS 会在中间加进大量模板元编程和它的抽象层，最后 dump 出来的东西是"CUTLASS 的 lowering + LLVM 的 lowering"，初学时会分不清哪一层是哪一层的锅。手写 60 行 fragment 代码，换来的是**每一行 IR 你都能对上自己的源码**。

## 2.2 先把 fragment 布局钉死

这是全课最重要的表。PTX ISA 文档里对 `mma.m16n8k16.row.col.f32.f16.f16.f32` 的 fragment 有正式定义，但文档是图，不好翻。我们直接写成公式。

warp 里 32 个线程，`lane = threadIdx.x`，定义两个身份：

```
gid = lane >> 2     // groupID  : 0..7   —— 决定"哪一行 / 哪一列"
tig = lane &  3     // 组内编号 : 0..3   —— 决定"k 方向上的偏移"
```

### A fragment：16×16 的 f16，每线程 4 个 `.b32`

`a[0..3]` 是 4 个 32 位寄存器，每个装 2 个 f16。A 是 row-major，元素位置：

| 寄存器 | 第 0 个 half | 第 1 个 half |
| --- | --- | --- |
| `a[0]` | `A[gid][2*tig]` | `A[gid][2*tig+1]` |
| `a[1]` | `A[gid+8][2*tig]` | `A[gid+8][2*tig+1]` |
| `a[2]` | `A[gid][2*tig+8]` | `A[gid][2*tig+9]` |
| `a[3]` | `A[gid+8][2*tig+8]` | `A[gid+8][2*tig+9]` |

换个角度看，16×16 的 A 被切成四个 8×8 子块，每个子块分配给 32 个线程里的每一个各 2 个元素：

```
              k = 0..7        k = 8..15
          ┌───────────────┬───────────────┐
  row 0-7 │     a[0]      │     a[2]      │   a[0] 属于: 第 gid 行, 第 2*tig/2*tig+1 列
          ├───────────────┼───────────────┤
  row 8-15│     a[1]      │     a[3]      │   a[1] 属于: 第 gid+8 行, ...
          └───────────────┴───────────────┘
```

注意 `a[1]` 和 `a[2]` 的顺序——**它跟 `ldmatrix.x4` 四条地址的自然顺序不一样**，第 8 章讲 `ldmatrix` 时你会看到我们是怎么把地址顺序掰过来的。这个"反直觉"的顺序是很多人手写 mma 第一次算错的原因。

### B fragment：16×8 的 f16，每线程 2 个 `.b32`

B 是"col-major"，也就是按 `k` 方向成对出现：

| 寄存器 | 第 0 个 half | 第 1 个 half |
| --- | --- | --- |
| `b[0]` | `B[k=2*tig][n=gid]` | `B[k=2*tig+1][n=gid]` |
| `b[1]` | `B[k=2*tig+8][n=gid]` | `B[k=2*tig+9][n=gid]` |

看仔细：**同一个寄存器里的两个 half，是同一个输出列上相邻的两个 k**。我们的 `B` 在 global memory 里是 row-major（`B[k][n]`，`n` 连续），所以这两个 half 在内存里并不连续，得分别取出来再拼：

```cpp
b[0] = pack2_half(pb0[0], pb0[N]);   // 跨了 N 个元素
```

这就是为什么 v1 里 B 的装载比 A 别扭。v2 里我们用 `ldmatrix.x2.trans` 让硬件去干这件事。

### C / D fragment：16×8 的 f32，每线程 4 个 `.f32`

| 寄存器 | 元素 |
| --- | --- |
| `c[0]` | `D[gid][2*tig]` |
| `c[1]` | `D[gid][2*tig+1]` |
| `c[2]` | `D[gid+8][2*tig]` |
| `c[3]` | `D[gid+8][2*tig+1]` |

C 和 D 是同一套布局（`mma` 的累加形式就是 `D = A*B + C`，我们直接把 `c` 同时当输入和输出）。

### 这三张表不是"抄来的"，是验证过的

写进 kernel 之后，我们在真机上跑：

```
[v1 manual  ] max |D - ref| = 4.29153e-06   -> PASS
[v2 ldmatrix] max |D - ref| = 4.29153e-06   -> PASS
```

对拍对象是同一个 CPU 上跑的 fp32 三重循环。误差 `4.29e-06` 是 f16 输入、f32 累加的合理量级（f16 的机器 epsilon 约 `9.8e-4`，累加 64 项后量级相符）。**如果布局错了，误差不会是 1e-6 这个量级，而是直接几个数量级地炸开**——我们在调试 v2 的时候就见过一次 `max |D - ref| = 13.9497`，原因就是 `ldmatrix` 的 `.trans` 配错了布局（详见第 8 章）。

## 2.3 完整源码

文件名：`code/tc_mma.cu`（270 行，可直接 `nvcc`/`clang++` 编译）。

```cpp
// ============================================================================
// Tensor Core 矩阵乘核心案例  ——  LLVM/Clang 编译 .cu 全链路教材配套代码
//
//   D[M,N] = A[M,K] * B[K,N]      A/B: half   D: float
//
//   硬件假设: RTX 4070 (sm_89), CUDA 12.8, LLVM/Clang 24
//   指令:     mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
//   两个 kernel:
//     mma_tc_manual   : 手动从 global 组装 fragment, 不碰 shared memory
//     mma_tc_ldmatrix : cp.async 搬 tile 到 shared memory + ldmatrix 装 fragment
//
//   约定: M % 16 == 0, N % 8 == 0, K % 16 == 0 (host 侧 assert)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   A: 16x16 f16 (row major),  4 x .b32 寄存器
//   B: 16x8  f16 (col major),  2 x .b32 寄存器
//   C/D: 16x8 f32,             4 x .f32 寄存器
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16(float *d, const uint32_t *a,
                                             const uint32_t *b, const float *c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// 把 memory 里连续的两个 half 打包成一个 .b32(供 A fragment 用)
__device__ __forceinline__ uint32_t pack_half2(const __half *p) {
  return *reinterpret_cast<const uint32_t *>(p);
}

// 把两个不相邻的 half 打包(供 B fragment 用: 同一列相邻两行 k)
__device__ __forceinline__ uint32_t pack2_half(__half lo, __half hi) {
  __half2 h = __halves2half2(lo, hi);
  return *reinterpret_cast<uint32_t *>(&h);
}

// ---------------------------------------------------------------------------
// Kernel v1: 手动组装 fragment, 只用 global memory
// ---------------------------------------------------------------------------
extern "C" __global__ void mma_tc_manual(const __half *__restrict__ A,
                                         const __half *__restrict__ B,
                                         float *__restrict__ D, int M, int N,
                                         int K) {
  const int lane = threadIdx.x;             // 0..31
  const int tile_m = blockIdx.y * 16;       // 本 warp 负责的 16 行
  const int tile_n = blockIdx.x * 8;        // 本 warp 负责的 8 列

  const int gid = lane >> 2;                // groupID   0..7: 行 / 列索引
  const int tig = lane & 3;                 // 组内编号  0..3: k 方向偏移

  // A fragment: 16x16, 每线程 4 个 .b32 = 8 个 half
  const int a_r0 = gid, a_r1 = gid + 8;     // reg0/reg2 用 a_r0, reg1/reg3 用 a_r1
  const int a_c0 = tig * 2, a_c1 = tig * 2 + 8;  // k 方向: 低 8 / 高 8

  // B fragment: 16x8, 每线程 2 个 .b32 = 4 个 half
  const int b_k0 = tig * 2, b_k1 = tig * 2 + 8;
  const int b_n = gid;                      // 输出列

  // C fragment: 16x8 f32, 每线程 4 个 float
  const int c_r0 = gid, c_r1 = gid + 8, c_c = tig * 2;

  float c[4] = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4], b[2];

    const __half *pa0 = A + (tile_m + a_r0) * K + k0;
    const __half *pa1 = A + (tile_m + a_r1) * K + k0;
    a[0] = pack_half2(pa0 + a_c0);   // row gid,    k 2t .. 2t+1
    a[1] = pack_half2(pa1 + a_c0);   // row gid+8,  k 2t .. 2t+1
    a[2] = pack_half2(pa0 + a_c1);   // row gid,    k 2t+8 .. 2t+9
    a[3] = pack_half2(pa1 + a_c1);   // row gid+8,  k 2t+8 .. 2t+9

    const __half *pb0 = B + (k0 + b_k0) * N + tile_n + b_n;
    const __half *pb1 = B + (k0 + b_k1) * N + tile_n + b_n;
    b[0] = pack2_half(pb0[0], pb0[N]);   // k 2t, 2t+1 在 n 列上
    b[1] = pack2_half(pb1[0], pb1[N]);   // k 2t+8, 2t+9

    mma_m16n8k16(c, a, b, c);
  }

  float *pd0 = D + (tile_m + c_r0) * N + tile_n + c_c;
  float *pd1 = D + (tile_m + c_r1) * N + tile_n + c_c;
  pd0[0] = c[0];
  pd0[1] = c[1];
  pd1[0] = c[2];
  pd1[1] = c[3];
}

// ---------------------------------------------------------------------------
// 异步拷贝 + ldmatrix 辅助
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
               "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void *smem, const void *gmem) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t *r, const void *addr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(addr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t *r,
                                                  const void *addr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(addr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}

// ---------------------------------------------------------------------------
// Kernel v2: cp.async 搬 tile 到 shared memory, ldmatrix 装 fragment
// ---------------------------------------------------------------------------
extern "C" __global__ void mma_tc_ldmatrix(const __half *__restrict__ A,
                                           const __half *__restrict__ B,
                                           float *__restrict__ D, int M, int N,
                                           int K) {
  // As: A 的 16x16 tile, 行主序
  // Bs: B 的 16x8 tile, 也保持行主序 [k][n] —— 转置交给 ldmatrix.trans 做
  __shared__ __align__(16) __half As[16][16];
  __shared__ __align__(16) __half Bs[16][8];

  const int lane = threadIdx.x;
  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * 8;

  const int gid = lane >> 2;
  const int tig = lane & 3;
  const int c_r0 = gid, c_r1 = gid + 8, c_c = tig * 2;
  float c[4] = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += 16) {
    // ---- 搬运 A tile: 16x16 half = 512B, 32 线程 x 16B 一轮搬完 ----
    {
      int idx = lane * 8;            // 每线程 8 个 half
      int r = idx >> 4, cc = idx & 15;
      cp_async16(&As[r][cc], A + (tile_m + r) * K + k0 + cc);
    }
    // ---- 搬运 B tile: 16x8 half = 256B, 32 线程 x 8B 一轮搬完 ----
    {
      int idx = lane * 4;            // 每线程 4 个 half
      int r = idx >> 3, cc = idx & 7;
      cp_async8(&Bs[r][cc], B + (k0 + r) * N + tile_n + cc);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // ---- ldmatrix 装 fragment ----
    // A: x4, 四个 8x8 子块 -> reg0(row0-7,k0-7) reg1(row8-15,k0-7)
    //                              reg2(row0-7,k8-15) reg3(row8-15,k8-15)
    const int grp = lane >> 3;                 // 0..3
    const int arow = (lane & 7) + ((grp & 1) ? 8 : 0);
    const int acol = (grp >= 2) ? 8 : 0;
    uint32_t a[4];
    ldmatrix_x4(a, &As[arow][acol]);

    // B: x2.trans, 地址给的是 [k][n] 里 k=0..15 各行的行首;
    //    转置之后每线程拿到 (k=2t,2t+1) 在 n=gid 上, 正是 col-major B fragment
    const int krow = lane & 15;
    uint32_t b[2];
    ldmatrix_x2_trans(b, &Bs[krow][0]);

    mma_m16n8k16(c, a, b, c);
    __syncthreads();               // 本轮 smem 用完再生效下一轮覆盖
  }

  float *pd0 = D + (tile_m + c_r0) * N + tile_n + c_c;
  float *pd1 = D + (tile_m + c_r1) * N + tile_n + c_c;
  pd0[0] = c[0];
  pd0[1] = c[1];
  pd1[0] = c[2];
  pd1[1] = c[3];
}

// ---------------------------------------------------------------------------
// Host 侧: 启动 + 与 CPU 参考实现比对
// ---------------------------------------------------------------------------
static void check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}

static float frand(unsigned &s) {   // 简单 LCG, 避免依赖 rand 的实现差异
  s = s * 1103515245u + 12345u;
  return ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

int main() {
  const int M = 256, N = 128, K = 64;
  printf("GEMM: M=%d N=%d K=%d  (sm_89, mma.m16n8k16)\n", M, N, K);

  __half *hA = (__half *)malloc(sizeof(__half) * M * K);
  __half *hB = (__half *)malloc(sizeof(__half) * K * N);
  float *hD = (float *)malloc(sizeof(float) * M * N);
  float *ref = (float *)malloc(sizeof(float) * M * N);

  unsigned seed = 12345;
  for (int i = 0; i < M * K; ++i) hA[i] = __float2half(frand(seed));
  for (int i = 0; i < K * N; ++i) hB[i] = __float2half(frand(seed));

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k)
        acc += __half2float(hA[m * K + k]) * __half2float(hB[k * N + n]);
      ref[m * N + n] = acc;
    }

  __half *dA, *dB;
  float *dD;
  check(cudaMalloc(&dA, sizeof(__half) * M * K), "malloc A");
  check(cudaMalloc(&dB, sizeof(__half) * K * N), "malloc B");
  check(cudaMalloc(&dD, sizeof(float) * M * N), "malloc D");
  check(cudaMemcpy(dA, hA, sizeof(__half) * M * K, cudaMemcpyHostToDevice), "H2D A");
  check(cudaMemcpy(dB, hB, sizeof(__half) * K * N, cudaMemcpyHostToDevice), "H2D B");

  dim3 grid(N / 8, M / 16);
  dim3 block(32);

  for (int variant = 0; variant < 2; ++variant) {
    check(cudaMemset(dD, 0, sizeof(float) * M * N), "memset D");
    if (variant == 0)
      mma_tc_manual<<<grid, block>>>(dA, dB, dD, M, N, K);
    else
      mma_tc_ldmatrix<<<grid, block>>>(dA, dB, dD, M, N, K);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "kernel sync");
    check(cudaMemcpy(hD, dD, sizeof(float) * M * N, cudaMemcpyDeviceToHost), "D2H D");

    double maxerr = 0.0;
    for (int i = 0; i < M * N; ++i)
      maxerr = fmax(maxerr, fabs((double)hD[i] - (double)ref[i]));
    printf("[%s] max |D - ref| = %.6g   -> %s\n",
           variant == 0 ? "v1 manual  " : "v2 ldmatrix",
           maxerr, maxerr < 1e-2 ? "PASS" : "FAIL");
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dD);
  free(hA); free(hB); free(hD); free(ref);
  return 0;
}
```

几个值得提前点出来的地方：

- `extern "C"` 是为了让 kernel 名字不被 C++ mangling。第 3 章你会看到，`mma_tc_manual` 在 `.ll` 里就长这样，SASS 里也是这个名字。
- `__restrict__` 会变成 IR 里的 `noalias`，影响 `ldg`/重排序，第 4 章讲。
- `__shared__` 数组是**函数作用域里的静态变量**，会被提升成模块级的 `addrspace(3) global`，第 3 章给证据。
- 三个 `int M, int N, int K` 参数没有被用到（v1 里用了 `N/K`，`M` 没用），不影响正确性，但会体现在 IR 里：`%3`（就是 `M`）在 `-O2` 之后可能被优化掉/只留在签名里。

## 2.4 编译命令

### 路径 A：clang/LLVM 编译（本教材主线）

```bash
CUDA=/usr/local/cuda-12.8
CFLAGS="--cuda-path=$CUDA --cuda-gpu-arch=sm_89"

# 只出设备端 IR
clang++ -x cuda $CFLAGS --cuda-device-only -O2 -S -emit-llvm -o tc_mma.ll tc_mma.cu

# 只出设备端 PTX
clang++ -x cuda $CFLAGS --cuda-device-only -O2 -S -o tc_mma.ptx tc_mma.cu

# host + device 一起编译成可执行文件（需要 libcudart）
clang++ -x cuda $CFLAGS -O3 -L$CUDA/lib64 -lcudart -o tc_mma tc_mma.cu
```

参数逐条解释：

| 参数 | 作用 | 不写会怎样 |
| --- | --- | --- |
| `-x cuda` | 让 clang 走 CUDA 前端（认 `__global__`、`__shared__`、`<<<>>>`） | 当成普通 C++，`__global__` 不认识 |
| `--cuda-path=` | 指定 CUDA 安装位置，找 `cuda_fp16.h`、libdevice、`ptxas` | 找不到 CUDA，报 "cannot find CUDA installation" |
| `--cuda-gpu-arch=sm_89` | 指定设备端目标架构 | **默认 `sm_52`**，编译能过但 PTX 里的 `.target` 不是你想要的（例如用不到 sm_89 的 `cp.async`/`ldmatrix` 之外的特性、或指令直接编译失败） |
| `--cuda-device-only` | 只编译设备端，不做 host 编译、不链接 | 会走完整 host+device 流程，你需要 libcudart |
| `-S -emit-llvm` | 输出 LLVM IR 而不是 PTX | 出 PTX |
| `-S`（不带 `-emit-llvm`） | 输出 PTX 文本 | — |
| `-L.../lib64 -lcudart` | 链接 CUDA runtime（host 侧才需要） | 链接失败，`undefined reference to cudaMalloc` |

两个容易踩的坑：

1. `-arch=sm_89` 是 **nvcc** 的写法。clang 不认，直接报错退出（exit=1，不产出任何文件）：

   ```
   clang++: error: unknown argument '-arch=sm_89'; did you mean '-march=sm_89'?
   ```

   这条错还算友好。**真正阴险的是忘写 `--cuda-gpu-arch`**：clang 不报错，安安静静按默认的
   `sm_52` 出一份 PTX（`.target sm_52`），你要到第 8 章逐条读 PTX 时才会发现架构不对。
   正确写法是 `--cuda-gpu-arch=sm_89`，clang 也接受 `--offload-arch=sm_89`（两条都实测过，产出同样的 `target-cpu="sm_89"`）。
2. `-nocudainc -nocudalib` 是"我不想用 CUDA 头文件和库"的意思，**我们的代码要 `cuda_fp16.h`，不能加这两个**。

### 路径 B：nvcc 编译（对照组）

```bash
nvcc -arch=sm_89 -O3 -ptx  -o tc_mma.nvcc.ptx tc_mma.cu   # 只要 PTX
nvcc -arch=sm_89 -O3       -o tc_mma.nvcc     tc_mma.cu   # 可执行文件
```

nvcc 在这里的角色是**对照组**：同样一份源码，两家前端的 PTX 长得不一样（第 8 章我们会并排看），SASS 却会收敛得很接近——因为最后都是 `ptxas` 干的活。这个对比能让你判断"某段代码是不是 LLVM 特有的怪癖"。

## 2.5 运行验证（实测输出）

```bash
$ bash llvm-cuda-course/tools/build-and-run.sh
===== [1] nvcc -arch=sm_89 -O3 =====
GEMM: M=256 N=128 K=64  (sm_89, mma.m16n8k16)
[v1 manual  ] max |D - ref| = 4.29153e-06   -> PASS
[v2 ldmatrix] max |D - ref| = 4.29153e-06   -> PASS
nvcc 路径 rc=0

===== [2] clang++ 作为 CUDA 编译器 (host+device) =====
GEMM: M=256 N=128 K=64  (sm_89, mma.m16n8k16)
[v1 manual  ] max |D - ref| = 4.29153e-06   -> PASS
[v2 ldmatrix] max |D - ref| = 4.29153e-06   -> PASS
clang 路径 rc=0
```

顺便验证一下 **clang 编出来的可执行文件里到底嵌了什么**（这一步很关键，它证明 clang 是真的自己走完了 "IR → PTX → ptxas → cubin" 全流程，不是把活甩给 nvcc）：

```bash
$ cuobjdump --list-elf tc_mma.clang
ELF file    1: tc_mma.1.sm_89.cubin
$ cuobjdump --list-ptx tc_mma.clang
PTX file    1: tc_mma.1.sm_89.ptx

$ cuobjdump --list-elf tc_mma.nvcc
ELF file    1: tc_mma.1.sm_89.cubin
ELF file    2: tc_mma.2.sm_89.cubin
```

clang 产物里是**一个 sm_89 cubin + 一份 sm_89 PTX**（v1、v2 在同一个编译单元里）；nvcc 拆成了两个 cubin（每个 kernel 一个 `.text` 段）。两者都能跑，性能差异不在这门课讨论范围内。

## 2.6 一份 dump 命令清单

下面这些命令全部在 `tools/dump-all.sh` 里，跑一次就全都落到 `dumps/`：

```bash
bash llvm-cuda-course/tools/dump-all.sh
```

| 想看的中间产物 | 命令 | 落到哪个文件 |
| --- | --- | --- |
| 未优化 IR | `clang++ -x cuda $CFLAGS --cuda-device-only -O0 -S -emit-llvm` | `dumps/01-device-O0.ll` |
| 优化后 IR | 同上 `-O2` | `dumps/02-device-O2.ll` |
| clang 出的 PTX | `clang++ -x cuda $CFLAGS --cuda-device-only -O2 -S` | `dumps/03-clang-O2.ptx` |
| nvcc 出的 PTX | `nvcc -arch=sm_89 -O3 -ptx` | `dumps/04-nvcc-O3.ptx` |
| cubin + 资源用量 | `ptxas -arch=sm_89 -v` | `dumps/05-clang.cubin`、`dumps/05-ptxas-verbose.txt` |
| SASS | `cuobjdump -sass` | `dumps/06-sass-cuobjdump.txt` |
| SASS（带机器码字） | `nvdisasm -c -hex` | `dumps/07-sass-nvdisasm.txt` |
| 寄存器/共享内存占用 | `cuobjdump -res-usage` | `dumps/08-res-usage.txt` |
| 指令选择后 MIR | `llc -stop-after=finalize-isel` | `dumps/09-mir-after-isel.mir` |
| 编码前 MIR（前导/后记之后） | `llc -stop-after=nvptx-prolog-epilog` | `dumps/10-mir-after-regalloc.mir` |
| 驱动完整日志 | `clang++ ... -v` | `dumps/11-clang-verbose.log` |
| 每个 pass 之后的 IR/MIR | `llc -print-after-all` | 直接打到 stderr |

`ptxas -v` 和 `cuobjdump -res-usage` 这两条**一定要看**，它们是你判断"寄存器压力大不大、有没有 spill"的唯一依据：

```
$ ptxas -arch=sm_89 -v 03-clang-O2.ptx -o 05-clang.cubin
ptxas info    : Compiling entry function 'mma_tc_ldmatrix' for 'sm_89'
ptxas info    : Function properties for mma_tc_ldmatrix
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 768 bytes smem, 388 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compiling entry function 'mma_tc_manual' for 'sm_89'
ptxas info    : Function properties for mma_tc_manual
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers, 388 bytes cmem[0]
```

读法：

- **v2 用了 40 个寄存器、768 字节共享内存、1 个 barrier**（`__syncthreads` 换来的），
- v1 也用了 40 个寄存器，但没有共享内存、没有 barrier；
- 两个都 **0 spill**——这对教学案例非常重要，一旦 spill，SASS 里会插一堆 `LDL/STL`，你就看不清 `HMMA` 周围的数据流了。

顺带一个"专业习惯"：`-maxrregcount` 或者 `__launch_bounds__` 会改变寄存器分配，进而改变 SASS 的形状。如果你发现自己的 SASS 和教程对不上，先确认没加这些限制。

## 2.7 踩坑清单

按我们实际踩到的顺序排：

1. **`uint32_t` 在 nvcc 下找不到**：`cuda_fp16.h` 在 clang 的头文件组合里会顺带带进 `<stdint.h>`，nvcc 的组合不一定。老老实实 `#include <cstdint>`。
2. **`ldmatrix.x2.trans` 的地址该指向谁**：第一版我们把 B 在 smem 里存成 `[n][k]`（转置过）再去 `ldmatrix`，结果 `max|D-ref| = 13.9497`。正确做法是**保持 `[k][n]` 原样**，让 `.trans` 去做转置，地址指向 k 行。第 8 章完整复盘。
3. **`cp.async` 的对齐要求**：16 字节版本要求源和目的都 16 字节对齐，8 字节版本要求 8 字节对齐。我们的 `A` 每行 16 个 half（32B）、`B` 每行 8 个 half（16B），配合 `K % 16 == 0`、`N % 8 == 0` 刚好满足。改矩阵尺寸时这条最容易炸。
4. **`__syncthreads()` 的位置**：v2 的 k 循环里有两个 `__syncthreads()`——一个在 `cp.async.wait_group` 之后（等数据到齐才能 `ldmatrix`），一个在 `mma` 之后（本轮 smem 用完才能被下一轮覆盖）。少一个就是数据竞争，而且**在小矩阵上可能仍然 PASS**（因为时序刚好），改大尺寸才暴露。
5. **`-debug-only=isel` 在 Release 构建里不存在**：详见第 1 章的说明，本教程用 `-print-after-all` + MIR dump 替代。

## 2.8 本章与后续章节的接口

从这里开始，后面每一章都只做一件事：**把本章某个产物切开看**。

```
本章产出                          下一章接手
─────────────────────────────────────────────────────────────
code/tc_mma.cu          ──────►  第 3 章：这份 C++ 怎么变成 IR
dumps/01-device-O0.ll   ──────►  第 3、4 章：逐段读 IR
dumps/02-device-O2.ll   ──────►  第 4、5 章：逐段读优化后的 IR / 看 pass 干了什么
dumps/09-mir-*.mir      ──────►  第 6、7 章：看后端怎么把它变成机器指令
dumps/03-clang-O2.ptx   ──────►  第 8 章：逐条读 PTX
dumps/06-sass-*.txt     ──────►  第 9 章：逐条读 SASS
```

下一章我们从 `tc_mma.cu` 出发，回答第一个硬问题：**`__global__`、`__shared__`、`__syncthreads()`、`asm volatile` 这四样东西，在 LLVM IR 里各自变成了什么？**
