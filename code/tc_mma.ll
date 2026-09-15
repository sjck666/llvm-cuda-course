; ModuleID = 'tc_mma.cu'
source_filename = "tc_mma.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__half = type { i16 }

@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
@_ZZ15mma_tc_ldmatrixE2Bs = internal addrspace(3) global [16 x [8 x %struct.__half]] undef, align 16

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_tc_manual(ptr noalias nofree noundef readonly captures(none) %0, ptr noalias nofree noundef readonly captures(none) %1, ptr noalias nofree noundef writeonly captures(none) %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %9 = shl nuw nsw i32 %8, 4
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %11 = shl i32 %10, 3
  %12 = lshr i32 %7, 2
  %13 = add nuw nsw i32 %12, 8
  %14 = shl nuw nsw i32 %7, 1
  %15 = and i32 %14, 6
  %16 = or disjoint i32 %15, 8
  %17 = icmp sgt i32 %5, 0
  %18 = add nuw nsw i32 %9, %12
  br i1 %17, label %23, label %19

19:                                               ; preds = %6
  %20 = sext i32 %11 to i64
  %21 = zext nneg i32 %15 to i64
  %22 = add nuw nsw i32 %13, %9
  br label %38

23:                                               ; preds = %6
  %24 = mul nuw nsw i32 %5, %18
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %25
  %27 = add nuw nsw i32 %13, %9
  %28 = mul nuw nsw i32 %5, %27
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %29
  %31 = zext nneg i32 %15 to i64
  %32 = zext nneg i32 %16 to i64
  %33 = sext i32 %11 to i64
  %34 = getelementptr [2 x i8], ptr %1, i64 %33
  %35 = zext nneg i32 %12 to i64
  %36 = getelementptr [2 x i8], ptr %34, i64 %35
  %37 = sext i32 %4 to i64
  br label %58

38:                                               ; preds = %58, %19
  %39 = phi i32 [ %22, %19 ], [ %27, %58 ]
  %40 = phi i64 [ %21, %19 ], [ %31, %58 ]
  %41 = phi i64 [ %20, %19 ], [ %33, %58 ]
  %42 = phi float [ 0.000000e+00, %19 ], [ %95, %58 ]
  %43 = phi float [ 0.000000e+00, %19 ], [ %94, %58 ]
  %44 = phi float [ 0.000000e+00, %19 ], [ %93, %58 ]
  %45 = phi float [ 0.000000e+00, %19 ], [ %92, %58 ]
  %46 = mul nsw i32 %4, %18
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds [4 x i8], ptr %2, i64 %47
  %49 = getelementptr inbounds [4 x i8], ptr %48, i64 %41
  %50 = getelementptr inbounds nuw [4 x i8], ptr %49, i64 %40
  %51 = mul nsw i32 %4, %39
  %52 = sext i32 %51 to i64
  %53 = getelementptr inbounds [4 x i8], ptr %2, i64 %52
  %54 = getelementptr inbounds [4 x i8], ptr %53, i64 %41
  %55 = getelementptr inbounds nuw [4 x i8], ptr %54, i64 %40
  store float %45, ptr %50, align 4, !tbaa !11
  %56 = getelementptr inbounds nuw i8, ptr %50, i64 4
  store float %44, ptr %56, align 4, !tbaa !11
  store float %43, ptr %55, align 4, !tbaa !11
  %57 = getelementptr inbounds nuw i8, ptr %55, i64 4
  store float %42, ptr %57, align 4, !tbaa !11
  ret void

58:                                               ; preds = %23, %58
  %59 = phi i32 [ 0, %23 ], [ %96, %58 ]
  %60 = phi float [ 0.000000e+00, %23 ], [ %92, %58 ]
  %61 = phi float [ 0.000000e+00, %23 ], [ %93, %58 ]
  %62 = phi float [ 0.000000e+00, %23 ], [ %94, %58 ]
  %63 = phi float [ 0.000000e+00, %23 ], [ %95, %58 ]
  %64 = zext nneg i32 %59 to i64
  %65 = getelementptr inbounds nuw [2 x i8], ptr %26, i64 %64
  %66 = getelementptr inbounds nuw [2 x i8], ptr %30, i64 %64
  %67 = getelementptr inbounds nuw [2 x i8], ptr %65, i64 %31
  %68 = load i32, ptr %67, align 4, !tbaa !13
  %69 = getelementptr inbounds nuw [2 x i8], ptr %66, i64 %31
  %70 = load i32, ptr %69, align 4, !tbaa !13
  %71 = getelementptr inbounds nuw [2 x i8], ptr %65, i64 %32
  %72 = load i32, ptr %71, align 4, !tbaa !13
  %73 = getelementptr inbounds nuw [2 x i8], ptr %66, i64 %32
  %74 = load i32, ptr %73, align 4, !tbaa !13
  %75 = or disjoint i32 %59, %15
  %76 = mul nsw i32 %75, %4
  %77 = sext i32 %76 to i64
  %78 = getelementptr [2 x i8], ptr %36, i64 %77
  %79 = or disjoint i32 %59, %16
  %80 = mul nsw i32 %79, %4
  %81 = sext i32 %80 to i64
  %82 = getelementptr [2 x i8], ptr %36, i64 %81
  %83 = load i16, ptr %78, align 2, !tbaa !14
  %84 = getelementptr inbounds [2 x i8], ptr %78, i64 %37
  %85 = load i16, ptr %84, align 2, !tbaa !14
  %86 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %83, i16 %85) #3, !srcloc !16
  %87 = load i16, ptr %82, align 2, !tbaa !14
  %88 = getelementptr inbounds [2 x i8], ptr %82, i64 %37
  %89 = load i16, ptr %88, align 2, !tbaa !14
  %90 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %87, i16 %89) #3, !srcloc !16
  %91 = tail call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %68, i32 %70, i32 %72, i32 %74, i32 %86, i32 %90, float %60, float %61, float %62, float %63) #4, !srcloc !17
  %92 = extractvalue { float, float, float, float } %91, 0
  %93 = extractvalue { float, float, float, float } %91, 1
  %94 = extractvalue { float, float, float, float } %91, 2
  %95 = extractvalue { float, float, float, float } %91, 3
  %96 = add nuw nsw i32 %59, 16
  %97 = icmp slt i32 %96, %5
  br i1 %97, label %58, label %38, !llvm.loop !18
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_tc_ldmatrix(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias nofree noundef writeonly captures(none) %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %9 = shl nuw nsw i32 %8, 4
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %11 = shl i32 %10, 3
  %12 = icmp sgt i32 %5, 0
  br i1 %12, label %15, label %13

13:                                               ; preds = %6
  %14 = sext i32 %11 to i64
  br label %55

15:                                               ; preds = %6
  %16 = shl nuw nsw i32 %7, 3
  %17 = lshr i32 %7, 1
  %18 = and i32 %16, 8
  %19 = zext nneg i32 %17 to i64
  %20 = getelementptr inbounds nuw [32 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 %19
  %21 = zext nneg i32 %18 to i64
  %22 = getelementptr inbounds nuw [2 x i8], ptr %20, i64 %21
  %23 = add nuw nsw i32 %9, %17
  %24 = mul nuw nsw i32 %5, %23
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %25
  %27 = getelementptr inbounds nuw [2 x i8], ptr %26, i64 %21
  %28 = addrspacecast ptr %22 to ptr addrspace(3)
  %29 = ptrtoint ptr addrspace(3) %28 to i64
  %30 = trunc i64 %29 to i32
  %31 = shl nuw nsw i32 %7, 2
  %32 = and i32 %31, 4
  %33 = getelementptr inbounds nuw [16 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 %19
  %34 = zext nneg i32 %32 to i64
  %35 = getelementptr inbounds nuw [2 x i8], ptr %33, i64 %34
  %36 = sext i32 %11 to i64
  %37 = getelementptr [2 x i8], ptr %1, i64 %36
  %38 = getelementptr [2 x i8], ptr %37, i64 %34
  %39 = addrspacecast ptr %35 to ptr addrspace(3)
  %40 = ptrtoint ptr addrspace(3) %39 to i64
  %41 = trunc i64 %40 to i32
  %42 = and i32 %7, 15
  %43 = icmp samesign ugt i32 %7, 15
  %44 = select i1 %43, i64 8, i64 0
  %45 = zext nneg i32 %42 to i64
  %46 = getelementptr inbounds nuw [32 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 %45
  %47 = getelementptr inbounds nuw [2 x i8], ptr %46, i64 %44
  %48 = addrspacecast ptr %47 to ptr addrspace(3)
  %49 = ptrtoint ptr addrspace(3) %48 to i64
  %50 = trunc i64 %49 to i32
  %51 = getelementptr inbounds nuw [16 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 %45
  %52 = addrspacecast ptr %51 to ptr addrspace(3)
  %53 = ptrtoint ptr addrspace(3) %52 to i64
  %54 = trunc i64 %53 to i32
  br label %80

55:                                               ; preds = %80, %13
  %56 = phi i64 [ %14, %13 ], [ %36, %80 ]
  %57 = phi float [ 0.000000e+00, %13 ], [ %104, %80 ]
  %58 = phi float [ 0.000000e+00, %13 ], [ %103, %80 ]
  %59 = phi float [ 0.000000e+00, %13 ], [ %102, %80 ]
  %60 = phi float [ 0.000000e+00, %13 ], [ %101, %80 ]
  %61 = shl nuw nsw i32 %7, 1
  %62 = and i32 %61, 6
  %63 = lshr i32 %7, 2
  %64 = add nuw nsw i32 %63, 8
  %65 = add nuw nsw i32 %9, %63
  %66 = mul nsw i32 %4, %65
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds [4 x i8], ptr %2, i64 %67
  %69 = getelementptr inbounds [4 x i8], ptr %68, i64 %56
  %70 = zext nneg i32 %62 to i64
  %71 = getelementptr inbounds nuw [4 x i8], ptr %69, i64 %70
  %72 = add nuw nsw i32 %64, %9
  %73 = mul nsw i32 %4, %72
  %74 = sext i32 %73 to i64
  %75 = getelementptr inbounds [4 x i8], ptr %2, i64 %74
  %76 = getelementptr inbounds [4 x i8], ptr %75, i64 %56
  %77 = getelementptr inbounds nuw [4 x i8], ptr %76, i64 %70
  store float %60, ptr %71, align 4, !tbaa !11
  %78 = getelementptr inbounds nuw i8, ptr %71, i64 4
  store float %59, ptr %78, align 4, !tbaa !11
  store float %58, ptr %77, align 4, !tbaa !11
  %79 = getelementptr inbounds nuw i8, ptr %77, i64 4
  store float %57, ptr %79, align 4, !tbaa !11
  ret void

80:                                               ; preds = %15, %80
  %81 = phi i32 [ 0, %15 ], [ %105, %80 ]
  %82 = phi float [ 0.000000e+00, %15 ], [ %101, %80 ]
  %83 = phi float [ 0.000000e+00, %15 ], [ %102, %80 ]
  %84 = phi float [ 0.000000e+00, %15 ], [ %103, %80 ]
  %85 = phi float [ 0.000000e+00, %15 ], [ %104, %80 ]
  %86 = zext nneg i32 %81 to i64
  %87 = getelementptr inbounds nuw [2 x i8], ptr %27, i64 %86
  tail call void asm sideeffect "cp.async.cg.shared.global [$0], [$1], 16;\0A", "r,l"(i32 %30, ptr %87) #4, !srcloc !20
  %88 = add nuw nsw i32 %81, %17
  %89 = mul nsw i32 %88, %4
  %90 = sext i32 %89 to i64
  %91 = getelementptr [2 x i8], ptr %38, i64 %90
  tail call void asm sideeffect "cp.async.ca.shared.global [$0], [$1], 8;\0A", "r,l"(i32 %41, ptr %91) #4, !srcloc !21
  tail call void asm sideeffect "cp.async.commit_group;\0A", ""() #4, !srcloc !22
  tail call void asm sideeffect "cp.async.wait_group 0;\0A", ""() #4, !srcloc !23
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %92 = tail call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A", "=r,=r,=r,=r,r"(i32 %50) #4, !srcloc !24
  %93 = extractvalue { i32, i32, i32, i32 } %92, 0
  %94 = extractvalue { i32, i32, i32, i32 } %92, 1
  %95 = extractvalue { i32, i32, i32, i32 } %92, 2
  %96 = extractvalue { i32, i32, i32, i32 } %92, 3
  %97 = tail call { i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0A", "=r,=r,r"(i32 %54) #4, !srcloc !25
  %98 = extractvalue { i32, i32 } %97, 0
  %99 = extractvalue { i32, i32 } %97, 1
  %100 = tail call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %93, i32 %94, i32 %95, i32 %96, i32 %98, i32 %99, float %82, float %83, float %84, float %85) #4, !srcloc !17
  %101 = extractvalue { float, float, float, float } %100, 0
  %102 = extractvalue { float, float, float, float } %100, 1
  %103 = extractvalue { float, float, float, float } %100, 2
  %104 = extractvalue { float, float, float, float } %100, 3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %105 = add nuw nsw i32 %81, 16
  %106 = icmp slt i32 %105, %5
  br i1 %106, label %80, label %55, !llvm.loop !26
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx87" "uniform-work-group-size" }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nounwind memory(none) }
attributes #4 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!llvm.errno.tbaa = !{!5}
!nvvmir.version = !{!10}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 8]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 24.0.0git (ssh://git@ssh.github.com:443/llvm/llvm-project.git 677a4c33ba942fe7aec6a1be15ba388f1b74d855)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"__libc_errno", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{i32 2, i32 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"float", !8, i64 0}
!13 = !{!7, !7, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"short", !8, i64 0}
!16 = !{i64 2156921074}
!17 = !{i64 1374}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
!20 = !{i64 4728}
!21 = !{i64 4959}
!22 = !{i64 5095}
!23 = !{i64 5195}
!24 = !{i64 5376}
!25 = !{i64 5740}
!26 = distinct !{!26, !19}
