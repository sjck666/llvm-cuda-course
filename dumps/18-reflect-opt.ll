; ModuleID = 'code/reflect_demo.ll'
source_filename = "code/reflect_demo.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@ftz = private unnamed_addr addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"
@arch = private unnamed_addr addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"

define float @fmul_ftz(float %a, float %b, float %c) #0 {
entry:
  br label %ftz

ftz:                                              ; preds = %entry
  %m1 = fmul float %a, %b
  %s1 = fadd float %m1, %c
  ret float %s1

nftz:                                             ; No predecessors!
  %m2 = fmul float %a, %b
  %s2 = fadd float %m2, %c
  ret float %s2
}

define i32 @arch_id() #0 {
entry:
  ret i32 0
}

attributes #0 = { "target-cpu"="sm_89" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"nvvm-reflect-ftz", i32 1}
