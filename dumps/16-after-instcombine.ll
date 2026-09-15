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
  %15 = shl nuw nsw i32 %14, 4
  %16 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %17 = shl i32 %16, 3
  %18 = lshr i32 %13, 2
  %19 = and i32 %13, 3
  %20 = add nuw nsw i32 %18, 8
  %21 = shl nuw nsw i32 %19, 1
  %22 = shl nuw nsw i32 %19, 1
  %23 = or disjoint i32 %22, 8
  %24 = shl nuw nsw i32 %19, 1
  %25 = shl nuw nsw i32 %19, 1
  %26 = or disjoint i32 %25, 8
  br label %27

27:                                               ; preds = %77, %6
  %.sroa.10.0 = phi float [ 0.000000e+00, %6 ], [ %78, %77 ]
  %.sroa.7.0 = phi float [ 0.000000e+00, %6 ], [ %79, %77 ]
  %.sroa.499.0 = phi float [ 0.000000e+00, %6 ], [ %80, %77 ]
  %.sroa.097.0 = phi float [ 0.000000e+00, %6 ], [ %81, %77 ]
  %.0 = phi i32 [ 0, %6 ], [ %82, %77 ]
  %28 = icmp slt i32 %.0, %5
  br i1 %28, label %29, label %83

29:                                               ; preds = %27
  %30 = add nuw nsw i32 %15, %18
  %31 = mul nsw i32 %30, %5
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds [2 x i8], ptr %0, i64 %32
  %34 = zext nneg i32 %.0 to i64
  %35 = getelementptr inbounds nuw [2 x i8], ptr %33, i64 %34
  %36 = add nuw nsw i32 %15, %20
  %37 = mul nsw i32 %36, %5
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds [2 x i8], ptr %0, i64 %38
  %40 = zext nneg i32 %.0 to i64
  %41 = getelementptr inbounds nuw [2 x i8], ptr %39, i64 %40
  %42 = zext nneg i32 %21 to i64
  %43 = getelementptr inbounds nuw [2 x i8], ptr %35, i64 %42
  %44 = load i32, ptr %43, align 4
  %45 = zext nneg i32 %21 to i64
  %46 = getelementptr inbounds nuw [2 x i8], ptr %41, i64 %45
  %47 = load i32, ptr %46, align 4
  %48 = zext nneg i32 %23 to i64
  %49 = getelementptr inbounds nuw [2 x i8], ptr %35, i64 %48
  %50 = load i32, ptr %49, align 4
  %51 = zext nneg i32 %23 to i64
  %52 = getelementptr inbounds nuw [2 x i8], ptr %41, i64 %51
  %53 = load i32, ptr %52, align 4
  %54 = or disjoint i32 %.0, %24
  %55 = mul nsw i32 %54, %4
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds [2 x i8], ptr %1, i64 %56
  %58 = sext i32 %17 to i64
  %59 = getelementptr inbounds [2 x i8], ptr %57, i64 %58
  %60 = zext nneg i32 %18 to i64
  %61 = getelementptr inbounds nuw [2 x i8], ptr %59, i64 %60
  %62 = or disjoint i32 %.0, %26
  %63 = mul nsw i32 %62, %4
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds [2 x i8], ptr %1, i64 %64
  %66 = sext i32 %17 to i64
  %67 = getelementptr inbounds [2 x i8], ptr %65, i64 %66
  %68 = zext nneg i32 %18 to i64
  %69 = getelementptr inbounds nuw [2 x i8], ptr %67, i64 %68
  %.sroa.09.0.copyload = load i16, ptr %61, align 2
  %70 = sext i32 %4 to i64
  %71 = getelementptr inbounds [2 x i8], ptr %61, i64 %70
  %.sroa.07.0.copyload = load i16, ptr %71, align 2
  store i16 %.sroa.09.0.copyload, ptr %8, align 2
  store i16 %.sroa.07.0.copyload, ptr %9, align 2
  call void @_ZL14__halves2half26__halfS_(ptr dead_on_unwind nonnull writable sret(%struct.__half2) align 4 %7, ptr noundef nonnull byval(%struct.__half) align 2 %8, ptr noundef nonnull byval(%struct.__half) align 2 %9) #6
  %72 = load i32, ptr %7, align 4
  %.sroa.05.0.copyload = load i16, ptr %69, align 2
  %73 = sext i32 %4 to i64
  %74 = getelementptr inbounds [2 x i8], ptr %69, i64 %73
  %.sroa.0.0.copyload = load i16, ptr %74, align 2
  store i16 %.sroa.05.0.copyload, ptr %11, align 2
  store i16 %.sroa.0.0.copyload, ptr %12, align 2
  call void @_ZL14__halves2half26__halfS_(ptr dead_on_unwind nonnull writable sret(%struct.__half2) align 4 %10, ptr noundef nonnull byval(%struct.__half) align 2 %11, ptr noundef nonnull byval(%struct.__half) align 2 %12) #6
  %75 = load i32, ptr %10, align 4
  %76 = call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %44, i32 %47, i32 %50, i32 %53, i32 %72, i32 %75, float %.sroa.097.0, float %.sroa.499.0, float %.sroa.7.0, float %.sroa.10.0) #7, !srcloc !6
  br label %77

77:                                               ; preds = %29
  %78 = extractvalue { float, float, float, float } %76, 3
  %79 = extractvalue { float, float, float, float } %76, 2
  %80 = extractvalue { float, float, float, float } %76, 1
  %81 = extractvalue { float, float, float, float } %76, 0
  %82 = add nuw nsw i32 %.0, 16
  br label %27, !llvm.loop !7

83:                                               ; preds = %27
  %84 = shl nuw nsw i32 %19, 1
  %85 = add nuw nsw i32 %18, 8
  %86 = add nuw nsw i32 %15, %18
  %87 = mul nsw i32 %86, %4
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds [4 x i8], ptr %2, i64 %88
  %90 = sext i32 %17 to i64
  %91 = getelementptr inbounds [4 x i8], ptr %89, i64 %90
  %92 = zext nneg i32 %84 to i64
  %93 = getelementptr inbounds nuw [4 x i8], ptr %91, i64 %92
  %94 = add nuw nsw i32 %15, %85
  %95 = mul nsw i32 %94, %4
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds [4 x i8], ptr %2, i64 %96
  %98 = sext i32 %17 to i64
  %99 = getelementptr inbounds [4 x i8], ptr %97, i64 %98
  %100 = zext nneg i32 %84 to i64
  %101 = getelementptr inbounds nuw [4 x i8], ptr %99, i64 %100
  store float %.sroa.097.0, ptr %93, align 4
  %102 = getelementptr inbounds nuw i8, ptr %93, i64 4
  store float %.sroa.499.0, ptr %102, align 4
  store float %.sroa.7.0, ptr %101, align 4
  %103 = getelementptr inbounds nuw i8, ptr %101, i64 4
  store float %.sroa.10.0, ptr %103, align 4
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
  %9 = shl nuw nsw i32 %8, 4
  %10 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %11 = shl i32 %10, 3
  br label %12

12:                                               ; preds = %71, %6
  %.sroa.10.0 = phi float [ 0.000000e+00, %6 ], [ %72, %71 ]
  %.sroa.7.0 = phi float [ 0.000000e+00, %6 ], [ %73, %71 ]
  %.sroa.487.0 = phi float [ 0.000000e+00, %6 ], [ %74, %71 ]
  %.sroa.085.0 = phi float [ 0.000000e+00, %6 ], [ %75, %71 ]
  %.0 = phi i32 [ 0, %6 ], [ %76, %71 ]
  %13 = icmp slt i32 %.0, %5
  br i1 %13, label %14, label %77

14:                                               ; preds = %12
  %15 = shl nuw nsw i32 %7, 3
  %16 = lshr i32 %7, 1
  %17 = and i32 %15, 8
  %18 = zext nneg i32 %16 to i64
  %19 = getelementptr inbounds nuw [32 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 %18
  %20 = zext nneg i32 %17 to i64
  %21 = getelementptr inbounds nuw [2 x i8], ptr %19, i64 %20
  %22 = add nuw nsw i32 %9, %16
  %23 = mul nsw i32 %22, %5
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds [2 x i8], ptr %0, i64 %24
  %26 = zext nneg i32 %.0 to i64
  %27 = getelementptr inbounds nuw [2 x i8], ptr %25, i64 %26
  %28 = zext nneg i32 %17 to i64
  %29 = getelementptr inbounds nuw [2 x i8], ptr %27, i64 %28
  %30 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %21) #6
  %31 = trunc i64 %30 to i32
  call void asm sideeffect "cp.async.cg.shared.global [$0], [$1], 16;\0A", "r,l"(i32 %31, ptr %29) #7, !srcloc !9
  %32 = shl nuw nsw i32 %7, 2
  %33 = lshr i32 %7, 1
  %34 = and i32 %32, 4
  %35 = zext nneg i32 %33 to i64
  %36 = getelementptr inbounds nuw [16 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 %35
  %37 = zext nneg i32 %34 to i64
  %38 = getelementptr inbounds nuw [2 x i8], ptr %36, i64 %37
  %39 = add nuw nsw i32 %.0, %33
  %40 = mul nsw i32 %39, %4
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds [2 x i8], ptr %1, i64 %41
  %43 = sext i32 %11 to i64
  %44 = getelementptr inbounds [2 x i8], ptr %42, i64 %43
  %45 = zext nneg i32 %34 to i64
  %46 = getelementptr inbounds nuw [2 x i8], ptr %44, i64 %45
  %47 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %38) #6
  %48 = trunc i64 %47 to i32
  call void asm sideeffect "cp.async.ca.shared.global [$0], [$1], 8;\0A", "r,l"(i32 %48, ptr %46) #7, !srcloc !10
  call void asm sideeffect "cp.async.commit_group;\0A", ""() #7, !srcloc !11
  call void asm sideeffect "cp.async.wait_group 0;\0A", ""() #7, !srcloc !12
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %49 = and i32 %7, 15
  %50 = icmp samesign ugt i32 %7, 15
  %51 = select i1 %50, i64 8, i64 0
  %52 = zext nneg i32 %49 to i64
  %53 = getelementptr inbounds nuw [32 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 %52
  %54 = getelementptr inbounds nuw [2 x i8], ptr %53, i64 %51
  %55 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %54) #6
  %56 = trunc i64 %55 to i32
  %57 = call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A", "=r,=r,=r,=r,r"(i32 %56) #7, !srcloc !13
  %58 = extractvalue { i32, i32, i32, i32 } %57, 0
  %59 = extractvalue { i32, i32, i32, i32 } %57, 1
  %60 = extractvalue { i32, i32, i32, i32 } %57, 2
  %61 = extractvalue { i32, i32, i32, i32 } %57, 3
  %62 = and i32 %7, 15
  %63 = zext nneg i32 %62 to i64
  %64 = getelementptr inbounds nuw [16 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 %63
  %65 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %64) #6
  %66 = trunc i64 %65 to i32
  %67 = call { i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0A", "=r,=r,r"(i32 %66) #7, !srcloc !14
  %68 = extractvalue { i32, i32 } %67, 0
  %69 = extractvalue { i32, i32 } %67, 1
  %70 = call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %58, i32 %59, i32 %60, i32 %61, i32 %68, i32 %69, float %.sroa.085.0, float %.sroa.487.0, float %.sroa.7.0, float %.sroa.10.0) #7, !srcloc !6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %71

71:                                               ; preds = %14
  %72 = extractvalue { float, float, float, float } %70, 3
  %73 = extractvalue { float, float, float, float } %70, 2
  %74 = extractvalue { float, float, float, float } %70, 1
  %75 = extractvalue { float, float, float, float } %70, 0
  %76 = add nuw nsw i32 %.0, 16
  br label %12, !llvm.loop !15

77:                                               ; preds = %12
  %78 = shl nuw nsw i32 %7, 1
  %79 = and i32 %78, 6
  %80 = lshr i32 %7, 2
  %81 = add nuw nsw i32 %80, 8
  %82 = add nuw nsw i32 %9, %80
  %83 = mul nsw i32 %82, %4
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds [4 x i8], ptr %2, i64 %84
  %86 = sext i32 %11 to i64
  %87 = getelementptr inbounds [4 x i8], ptr %85, i64 %86
  %88 = zext nneg i32 %79 to i64
  %89 = getelementptr inbounds nuw [4 x i8], ptr %87, i64 %88
  %90 = add nuw nsw i32 %9, %81
  %91 = mul nsw i32 %90, %4
  %92 = sext i32 %91 to i64
  %93 = getelementptr inbounds [4 x i8], ptr %2, i64 %92
  %94 = sext i32 %11 to i64
  %95 = getelementptr inbounds [4 x i8], ptr %93, i64 %94
  %96 = zext nneg i32 %79 to i64
  %97 = getelementptr inbounds nuw [4 x i8], ptr %95, i64 %96
  store float %.sroa.085.0, ptr %89, align 4
  %98 = getelementptr inbounds nuw i8, ptr %89, i64 4
  store float %.sroa.487.0, ptr %98, align 4
  store float %.sroa.7.0, ptr %97, align 4
  %99 = getelementptr inbounds nuw i8, ptr %97, i64 4
  store float %.sroa.10.0, ptr %99, align 4
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
