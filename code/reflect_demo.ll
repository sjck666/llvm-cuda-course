; NVVMReflect 演示: __nvvm_reflect 在 NVPTX 后端流水线里被折成常数
; 对应 CUDA 里的 __nvvm_reflect("__CUDA_FTZ") / __nvvm_reflect("__CUDA_ARCH")
; 跑法见 tools/dump-all.sh 或教材第 5 章

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@ftz  = private unnamed_addr addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"
@arch = private unnamed_addr addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"

declare i32 @__nvvm_reflect(ptr)

; 用 __CUDA_FTZ 选择 FTZ / 非 FTZ 两条路径
define float @fmul_ftz(float %a, float %b, float %c) {
entry:
  %r = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @ftz to ptr))
  %is_ftz = icmp ne i32 %r, 0
  br i1 %is_ftz, label %ftz, label %nftz

ftz:
  %m1 = fmul float %a, %b
  %s1 = fadd float %m1, %c
  ret float %s1

nftz:
  %m2 = fmul float %a, %b
  %s2 = fadd float %m2, %c
  ret float %s2
}

; 用 __CUDA_ARCH 选择代码路径
define i32 @arch_id() {
entry:
  %a = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @arch to ptr))
  ret i32 %a
}

; CUDA 工具链用这个 module flag 传 FTZ 设置 (cuda 默认 -ftz=true)
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"nvvm-reflect-ftz", i32 1}
