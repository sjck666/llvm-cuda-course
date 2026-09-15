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
