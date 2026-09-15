// 实验: 能不能让某一条内联汇编不被标 convergent?
//   (1) __attribute__((noconvergent)) 放在 asm 前   -- 语法报错
//   (2) [[clang::noconvergent]]       放在 asm 前   -- 警告属性被忽略，无效
//   (3) [[clang::noconvergent]] { asm ...; }        -- 有效: 属性挂在语句块上
// 对应 clang/lib/CodeGen/CGStmt.cpp 里的 attr::NoConvergent 分支
#include <cstdint>
#include <cuda_runtime.h>

extern "C" __global__ void with_convergent(float *out, float *in) {
  float x = in[threadIdx.x];
  asm volatile("add.f32 %0, %1, %1;" : "+f"(x) : "f"(x));
  out[threadIdx.x] = x;
}

extern "C" __global__ void with_noconvergent(float *out, float *in) {
  float x = in[threadIdx.x];
  [[clang::noconvergent]]
  asm volatile("add.f32 %0, %1, %1;" : "+f"(x) : "f"(x));
  out[threadIdx.x] = x;
}

// (3) 把属性挂在"语句"上: AttributedStmt 才会被 CGStmt.cpp 的
//     InNoConvergentAttributedStmt 接住, 这条 asm 才能真正摘掉 convergent
extern "C" __global__ void with_noconvergent_stmt(float *out, float *in) {
  float x = in[threadIdx.x];
  [[clang::noconvergent]] {
    asm volatile("add.f32 %0, %1, %1;" : "+f"(x) : "f"(x));
  }
  out[threadIdx.x] = x;
}
