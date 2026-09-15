// 与 unroll_demo.cu 完全相同的代码, 只在循环前多一行 #pragma unroll 4
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void mma_demo2(float *d, const uint32_t *a,
                                          const uint32_t *b, const float *c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

extern "C" __global__ void mma_kconst_pragma(const __half *__restrict__ A,
                                             const __half *__restrict__ B,
                                             float *__restrict__ D) {
  constexpr int K = 64, N = 128;
  const int lane = threadIdx.x;
  const int tile_m = blockIdx.y * 16, tile_n = blockIdx.x * 8;
  const int gid = lane >> 2, tig = lane & 3;
  const int a_r0 = gid, a_r1 = gid + 8;
  const int a_c0 = tig * 2, a_c1 = tig * 2 + 8;
  const int b_k0 = tig * 2, b_k1 = tig * 2 + 8, b_n = gid;
  const int c_r0 = gid, c_r1 = gid + 8, c_c = tig * 2;
  float c[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll 4
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4], b[2];
    const __half *pa0 = A + (tile_m + a_r0) * K + k0;
    const __half *pa1 = A + (tile_m + a_r1) * K + k0;
    a[0] = *reinterpret_cast<const uint32_t *>(pa0 + a_c0);
    a[1] = *reinterpret_cast<const uint32_t *>(pa1 + a_c0);
    a[2] = *reinterpret_cast<const uint32_t *>(pa0 + a_c1);
    a[3] = *reinterpret_cast<const uint32_t *>(pa1 + a_c1);
    const __half *pb0 = B + (k0 + b_k0) * N + tile_n + b_n;
    const __half *pb1 = B + (k0 + b_k1) * N + tile_n + b_n;
    __half2 h0 = __halves2half2(pb0[0], pb0[N]);
    __half2 h1 = __halves2half2(pb1[0], pb1[N]);
    b[0] = *reinterpret_cast<const uint32_t *>(&h0);
    b[1] = *reinterpret_cast<const uint32_t *>(&h1);
    mma_demo2(c, a, b, c);
  }

  float *pd0 = D + (tile_m + c_r0) * N + tile_n + c_c;
  float *pd1 = D + (tile_m + c_r1) * N + tile_n + c_c;
  pd0[0] = c[0]; pd0[1] = c[1];
  pd1[0] = c[2]; pd1[1] = c[3];
}
