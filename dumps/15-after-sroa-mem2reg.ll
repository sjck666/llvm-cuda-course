; ModuleID = 'dumps/01b-device-O0-nooptnone.ll'
source_filename = "code/tc_mma.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__half = type { i16 }
%struct.__half2 = type { %struct.__half, %struct.__half }

$__nv_cvta_generic_to_shared_impl = comdat any

@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
@_ZZ15mma_tc_ldmatrixE2Bs = internal addrspace(3) global [16 x [8 x %struct.__half]] undef, align 16

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_tc_manual(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #0 {
  %7 = alloca %struct.__half2, align 4
  %8 = alloca %struct.__half, align 2
  %9 = alloca %struct.__half, align 2
  %10 = alloca %struct.__half2, align 4
  %11 = alloca %struct.__half, align 2
  %12 = alloca %struct.__half, align 2
  %13 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %14 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %15 = mul i32 %14, 16
  %16 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %17 = mul i32 %16, 8
  %18 = ashr i32 %13, 2
  %19 = and i32 %13, 3
  %20 = add nsw i32 %18, 8
  %21 = mul nsw i32 %19, 2
  %22 = mul nsw i32 %19, 2
  %23 = add nsw i32 %22, 8
  %24 = mul nsw i32 %19, 2
  %25 = mul nsw i32 %19, 2
  %26 = add nsw i32 %25, 8
  %27 = add nsw i32 %18, 8
  %28 = mul nsw i32 %19, 2
  br label %29

29:                                               ; preds = %85, %6
  %.sroa.10.0 = phi float [ 0.000000e+00, %6 ], [ %84, %85 ]
  %.sroa.7.0 = phi float [ 0.000000e+00, %6 ], [ %83, %85 ]
  %.sroa.499.0 = phi float [ 0.000000e+00, %6 ], [ %82, %85 ]
  %.sroa.097.0 = phi float [ 0.000000e+00, %6 ], [ %81, %85 ]
  %.0 = phi i32 [ 0, %6 ], [ %86, %85 ]
  %30 = icmp slt i32 %.0, %5
  br i1 %30, label %31, label %87

31:                                               ; preds = %29
  %32 = add nsw i32 %15, %18
  %33 = mul nsw i32 %32, %5
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds %struct.__half, ptr %0, i64 %34
  %36 = sext i32 %.0 to i64
  %37 = getelementptr inbounds %struct.__half, ptr %35, i64 %36
  %38 = add nsw i32 %15, %20
  %39 = mul nsw i32 %38, %5
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds %struct.__half, ptr %0, i64 %40
  %42 = sext i32 %.0 to i64
  %43 = getelementptr inbounds %struct.__half, ptr %41, i64 %42
  %44 = sext i32 %21 to i64
  %45 = getelementptr inbounds %struct.__half, ptr %37, i64 %44
  %46 = load i32, ptr %45, align 4
  %47 = sext i32 %21 to i64
  %48 = getelementptr inbounds %struct.__half, ptr %43, i64 %47
  %49 = load i32, ptr %48, align 4
  %50 = sext i32 %23 to i64
  %51 = getelementptr inbounds %struct.__half, ptr %37, i64 %50
  %52 = load i32, ptr %51, align 4
  %53 = sext i32 %23 to i64
  %54 = getelementptr inbounds %struct.__half, ptr %43, i64 %53
  %55 = load i32, ptr %54, align 4
  %56 = add nsw i32 %.0, %24
  %57 = mul nsw i32 %56, %4
  %58 = sext i32 %57 to i64
  %59 = getelementptr inbounds %struct.__half, ptr %1, i64 %58
  %60 = sext i32 %17 to i64
  %61 = getelementptr inbounds %struct.__half, ptr %59, i64 %60
  %62 = sext i32 %18 to i64
  %63 = getelementptr inbounds %struct.__half, ptr %61, i64 %62
  %64 = add nsw i32 %.0, %26
  %65 = mul nsw i32 %64, %4
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds %struct.__half, ptr %1, i64 %66
  %68 = sext i32 %17 to i64
  %69 = getelementptr inbounds %struct.__half, ptr %67, i64 %68
  %70 = sext i32 %18 to i64
  %71 = getelementptr inbounds %struct.__half, ptr %69, i64 %70
  %72 = getelementptr inbounds %struct.__half, ptr %63, i64 0
  %.sroa.09.0.copyload = load i16, ptr %72, align 2
  %73 = sext i32 %4 to i64
  %74 = getelementptr inbounds %struct.__half, ptr %63, i64 %73
  %.sroa.07.0.copyload = load i16, ptr %74, align 2
  store i16 %.sroa.09.0.copyload, ptr %8, align 2
  store i16 %.sroa.07.0.copyload, ptr %9, align 2
  call void @_ZL14__halves2half26__halfS_(ptr dead_on_unwind writable sret(%struct.__half2) align 4 %7, ptr noundef byval(%struct.__half) align 2 %8, ptr noundef byval(%struct.__half) align 2 %9) #6
  %75 = load i32, ptr %7, align 4
  %76 = getelementptr inbounds %struct.__half, ptr %71, i64 0
  %.sroa.05.0.copyload = load i16, ptr %76, align 2
  %77 = sext i32 %4 to i64
  %78 = getelementptr inbounds %struct.__half, ptr %71, i64 %77
  %.sroa.0.0.copyload = load i16, ptr %78, align 2
  store i16 %.sroa.05.0.copyload, ptr %11, align 2
  store i16 %.sroa.0.0.copyload, ptr %12, align 2
  call void @_ZL14__halves2half26__halfS_(ptr dead_on_unwind writable sret(%struct.__half2) align 4 %10, ptr noundef byval(%struct.__half) align 2 %11, ptr noundef byval(%struct.__half) align 2 %12) #6
  %79 = load i32, ptr %10, align 4
  %80 = call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %46, i32 %49, i32 %52, i32 %55, i32 %75, i32 %79, float %.sroa.097.0, float %.sroa.499.0, float %.sroa.7.0, float %.sroa.10.0) #7, !srcloc !6
  %81 = extractvalue { float, float, float, float } %80, 0
  %82 = extractvalue { float, float, float, float } %80, 1
  %83 = extractvalue { float, float, float, float } %80, 2
  %84 = extractvalue { float, float, float, float } %80, 3
  br label %85

85:                                               ; preds = %31
  %86 = add nsw i32 %.0, 16
  br label %29, !llvm.loop !7

87:                                               ; preds = %29
  %88 = add nsw i32 %15, %18
  %89 = mul nsw i32 %88, %4
  %90 = sext i32 %89 to i64
  %91 = getelementptr inbounds float, ptr %2, i64 %90
  %92 = sext i32 %17 to i64
  %93 = getelementptr inbounds float, ptr %91, i64 %92
  %94 = sext i32 %28 to i64
  %95 = getelementptr inbounds float, ptr %93, i64 %94
  %96 = add nsw i32 %15, %27
  %97 = mul nsw i32 %96, %4
  %98 = sext i32 %97 to i64
  %99 = getelementptr inbounds float, ptr %2, i64 %98
  %100 = sext i32 %17 to i64
  %101 = getelementptr inbounds float, ptr %99, i64 %100
  %102 = sext i32 %28 to i64
  %103 = getelementptr inbounds float, ptr %101, i64 %102
  %104 = getelementptr inbounds float, ptr %95, i64 0
  store float %.sroa.097.0, ptr %104, align 4
  %105 = getelementptr inbounds float, ptr %95, i64 1
  store float %.sroa.499.0, ptr %105, align 4
  %106 = getelementptr inbounds float, ptr %103, i64 0
  store float %.sroa.7.0, ptr %106, align 4
  %107 = getelementptr inbounds float, ptr %103, i64 1
  store float %.sroa.10.0, ptr %107, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_tc_ldmatrix(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #0 {
  %7 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %9 = mul i32 %8, 16
  %10 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %11 = mul i32 %10, 8
  %12 = ashr i32 %7, 2
  %13 = and i32 %7, 3
  %14 = add nsw i32 %12, 8
  %15 = mul nsw i32 %13, 2
  br label %16

16:                                               ; preds = %88, %6
  %.sroa.10.0 = phi float [ 0.000000e+00, %6 ], [ %87, %88 ]
  %.sroa.7.0 = phi float [ 0.000000e+00, %6 ], [ %86, %88 ]
  %.sroa.487.0 = phi float [ 0.000000e+00, %6 ], [ %85, %88 ]
  %.sroa.085.0 = phi float [ 0.000000e+00, %6 ], [ %84, %88 ]
  %.0 = phi i32 [ 0, %6 ], [ %89, %88 ]
  %17 = icmp slt i32 %.0, %5
  br i1 %17, label %18, label %90

18:                                               ; preds = %16
  %19 = mul nsw i32 %7, 8
  %20 = ashr i32 %19, 4
  %21 = and i32 %19, 15
  %22 = sext i32 %20 to i64
  %23 = getelementptr inbounds [16 x [16 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 0, i64 %22
  %24 = sext i32 %21 to i64
  %25 = getelementptr inbounds [16 x %struct.__half], ptr %23, i64 0, i64 %24
  %26 = add nsw i32 %9, %20
  %27 = mul nsw i32 %26, %5
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds %struct.__half, ptr %0, i64 %28
  %30 = sext i32 %.0 to i64
  %31 = getelementptr inbounds %struct.__half, ptr %29, i64 %30
  %32 = sext i32 %21 to i64
  %33 = getelementptr inbounds %struct.__half, ptr %31, i64 %32
  %34 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %25) #6
  %35 = trunc i64 %34 to i32
  call void asm sideeffect "cp.async.cg.shared.global [$0], [$1], 16;\0A", "r,l"(i32 %35, ptr %33) #7, !srcloc !9
  %36 = mul nsw i32 %7, 4
  %37 = ashr i32 %36, 3
  %38 = and i32 %36, 7
  %39 = sext i32 %37 to i64
  %40 = getelementptr inbounds [16 x [8 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 0, i64 %39
  %41 = sext i32 %38 to i64
  %42 = getelementptr inbounds [8 x %struct.__half], ptr %40, i64 0, i64 %41
  %43 = add nsw i32 %.0, %37
  %44 = mul nsw i32 %43, %4
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds %struct.__half, ptr %1, i64 %45
  %47 = sext i32 %11 to i64
  %48 = getelementptr inbounds %struct.__half, ptr %46, i64 %47
  %49 = sext i32 %38 to i64
  %50 = getelementptr inbounds %struct.__half, ptr %48, i64 %49
  %51 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %42) #6
  %52 = trunc i64 %51 to i32
  call void asm sideeffect "cp.async.ca.shared.global [$0], [$1], 8;\0A", "r,l"(i32 %52, ptr %50) #7, !srcloc !10
  call void asm sideeffect "cp.async.commit_group;\0A", ""() #7, !srcloc !11
  call void asm sideeffect "cp.async.wait_group 0;\0A", ""() #7, !srcloc !12
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %53 = ashr i32 %7, 3
  %54 = and i32 %7, 7
  %55 = and i32 %53, 1
  %56 = icmp ne i32 %55, 0
  %57 = zext i1 %56 to i64
  %58 = select i1 %56, i32 8, i32 0
  %59 = add nsw i32 %54, %58
  %60 = icmp sge i32 %53, 2
  %61 = zext i1 %60 to i64
  %62 = select i1 %60, i32 8, i32 0
  %63 = sext i32 %59 to i64
  %64 = getelementptr inbounds [16 x [16 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 0, i64 %63
  %65 = sext i32 %62 to i64
  %66 = getelementptr inbounds [16 x %struct.__half], ptr %64, i64 0, i64 %65
  %67 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %66) #6
  %68 = trunc i64 %67 to i32
  %69 = call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A", "=r,=r,=r,=r,r"(i32 %68) #7, !srcloc !13
  %70 = extractvalue { i32, i32, i32, i32 } %69, 0
  %71 = extractvalue { i32, i32, i32, i32 } %69, 1
  %72 = extractvalue { i32, i32, i32, i32 } %69, 2
  %73 = extractvalue { i32, i32, i32, i32 } %69, 3
  %74 = and i32 %7, 15
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds [16 x [8 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 0, i64 %75
  %77 = getelementptr inbounds [8 x %struct.__half], ptr %76, i64 0, i64 0
  %78 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %77) #6
  %79 = trunc i64 %78 to i32
  %80 = call { i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0A", "=r,=r,r"(i32 %79) #7, !srcloc !14
  %81 = extractvalue { i32, i32 } %80, 0
  %82 = extractvalue { i32, i32 } %80, 1
  %83 = call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %70, i32 %71, i32 %72, i32 %73, i32 %81, i32 %82, float %.sroa.085.0, float %.sroa.487.0, float %.sroa.7.0, float %.sroa.10.0) #7, !srcloc !6
  %84 = extractvalue { float, float, float, float } %83, 0
  %85 = extractvalue { float, float, float, float } %83, 1
  %86 = extractvalue { float, float, float, float } %83, 2
  %87 = extractvalue { float, float, float, float } %83, 3
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %88

88:                                               ; preds = %18
  %89 = add nsw i32 %.0, 16
  br label %16, !llvm.loop !15

90:                                               ; preds = %16
  %91 = add nsw i32 %9, %12
  %92 = mul nsw i32 %91, %4
  %93 = sext i32 %92 to i64
  %94 = getelementptr inbounds float, ptr %2, i64 %93
  %95 = sext i32 %11 to i64
  %96 = getelementptr inbounds float, ptr %94, i64 %95
  %97 = sext i32 %15 to i64
  %98 = getelementptr inbounds float, ptr %96, i64 %97
  %99 = add nsw i32 %9, %14
  %100 = mul nsw i32 %99, %4
  %101 = sext i32 %100 to i64
  %102 = getelementptr inbounds float, ptr %2, i64 %101
  %103 = sext i32 %11 to i64
  %104 = getelementptr inbounds float, ptr %102, i64 %103
  %105 = sext i32 %15 to i64
  %106 = getelementptr inbounds float, ptr %104, i64 %105
  %107 = getelementptr inbounds float, ptr %98, i64 0
  store float %.sroa.085.0, ptr %107, align 4
  %108 = getelementptr inbounds float, ptr %98, i64 1
  store float %.sroa.487.0, ptr %108, align 4
  %109 = getelementptr inbounds float, ptr %106, i64 0
  store float %.sroa.7.0, ptr %109, align 4
  %110 = getelementptr inbounds float, ptr %106, i64 1
  store float %.sroa.10.0, ptr %110, align 4
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: convergent mustprogress noinline nounwind
define internal void @_ZL14__halves2half26__halfS_(ptr dead_on_unwind noalias writable sret(%struct.__half2) align 4 %0, ptr noundef byval(%struct.__half) align 2 %1, ptr noundef byval(%struct.__half) align 2 %2) #5 {
  %4 = load i16, ptr %1, align 2
  %5 = load i16, ptr %2, align 2
  %6 = call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %4, i16 %5) #8, !srcloc !16
  store i32 %6, ptr %0, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline nounwind
define internal noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %0) #5 {
  %2 = call i64 @__nv_cvta_generic_to_shared_impl(ptr noundef %0) #6
  ret i64 %2
}

; Function Attrs: convergent mustprogress noinline nounwind
define linkonce_odr dso_local i64 @__nv_cvta_generic_to_shared_impl(ptr noundef %0) #5 comdat {
  %2 = addrspacecast ptr %0 to ptr addrspace(3)
  %3 = ptrtoint ptr addrspace(3) %2 to i64
  ret i64 %3
}

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx87" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent mustprogress noinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx87" "uniform-work-group-size" }
attributes #6 = { convergent nounwind "uniform-work-group-size" }
attributes #7 = { convergent nounwind }
attributes #8 = { convergent nounwind memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!nvvmir.version = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 8]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 24.0.0git (ssh://git@ssh.github.com:443/llvm/llvm-project.git 677a4c33ba942fe7aec6a1be15ba388f1b74d855)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!5 = !{i32 2, i32 0}
!6 = !{i64 1374}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{i64 4728}
!10 = !{i64 4959}
!11 = !{i64 5095}
!12 = !{i64 5195}
!13 = !{i64 5376}
!14 = !{i64 5740}
!15 = distinct !{!15, !8}
!16 = !{i64 2156917663}
