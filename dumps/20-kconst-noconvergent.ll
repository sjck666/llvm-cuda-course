; ModuleID = 'code/unroll_demo.cu'
source_filename = "code/unroll_demo.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_kconst(ptr noalias nofree noundef readonly captures(none) %0, ptr noalias nofree noundef readonly captures(none) %1, ptr noalias nofree noundef writeonly captures(none) %2) local_unnamed_addr #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %6 = shl nuw nsw i32 %5, 4
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %8 = shl i32 %7, 3
  %9 = lshr i32 %4, 2
  %10 = add nuw nsw i32 %9, 8
  %11 = shl nuw nsw i32 %4, 1
  %12 = and i32 %11, 6
  %13 = or disjoint i32 %12, 8
  %14 = add nuw nsw i32 %6, %9
  %15 = shl nuw nsw i32 %14, 6
  %16 = zext nneg i32 %15 to i64
  %17 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %16
  %18 = add nuw nsw i32 %10, %6
  %19 = shl nuw nsw i32 %18, 6
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %20
  %22 = zext nneg i32 %12 to i64
  %23 = zext nneg i32 %13 to i64
  %24 = sext i32 %8 to i64
  %25 = getelementptr [2 x i8], ptr %1, i64 %24
  %26 = zext nneg i32 %9 to i64
  %27 = getelementptr [2 x i8], ptr %25, i64 %26
  %28 = getelementptr inbounds nuw [2 x i8], ptr %17, i64 %22
  %29 = load i32, ptr %28, align 4, !tbaa !11
  %30 = getelementptr inbounds nuw [2 x i8], ptr %21, i64 %22
  %31 = load i32, ptr %30, align 4, !tbaa !11
  %32 = getelementptr inbounds nuw [2 x i8], ptr %17, i64 %23
  %33 = load i32, ptr %32, align 4, !tbaa !11
  %34 = getelementptr inbounds nuw [2 x i8], ptr %21, i64 %23
  %35 = load i32, ptr %34, align 4, !tbaa !11
  %36 = shl nuw nsw i32 %12, 7
  %37 = zext nneg i32 %36 to i64
  %38 = getelementptr [2 x i8], ptr %27, i64 %37
  %39 = shl nuw nsw i32 %13, 7
  %40 = zext nneg i32 %39 to i64
  %41 = getelementptr [2 x i8], ptr %27, i64 %40
  %42 = load i16, ptr %38, align 2, !tbaa !12
  %43 = getelementptr inbounds nuw i8, ptr %38, i64 256
  %44 = load i16, ptr %43, align 2, !tbaa !12
  %45 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %42, i16 %44) #2, !srcloc !14
  %46 = load i16, ptr %41, align 2, !tbaa !12
  %47 = getelementptr inbounds nuw i8, ptr %41, i64 256
  %48 = load i16, ptr %47, align 2, !tbaa !12
  %49 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %46, i16 %48) #2, !srcloc !14
  %50 = tail call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %29, i32 %31, i32 %33, i32 %35, i32 %45, i32 %49, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00) #3, !srcloc !15
  %51 = extractvalue { float, float, float, float } %50, 0
  %52 = extractvalue { float, float, float, float } %50, 1
  %53 = extractvalue { float, float, float, float } %50, 2
  %54 = extractvalue { float, float, float, float } %50, 3
  %55 = getelementptr inbounds nuw i8, ptr %17, i64 32
  %56 = getelementptr inbounds nuw i8, ptr %21, i64 32
  %57 = getelementptr inbounds nuw [2 x i8], ptr %55, i64 %22
  %58 = load i32, ptr %57, align 4, !tbaa !11
  %59 = getelementptr inbounds nuw [2 x i8], ptr %56, i64 %22
  %60 = load i32, ptr %59, align 4, !tbaa !11
  %61 = getelementptr inbounds nuw [2 x i8], ptr %55, i64 %23
  %62 = load i32, ptr %61, align 4, !tbaa !11
  %63 = getelementptr inbounds nuw [2 x i8], ptr %56, i64 %23
  %64 = load i32, ptr %63, align 4, !tbaa !11
  %65 = shl nuw nsw i32 %12, 7
  %66 = zext nneg i32 %65 to i64
  %67 = getelementptr [2 x i8], ptr %27, i64 %66
  %68 = getelementptr i8, ptr %67, i64 4096
  %69 = shl nuw nsw i32 %12, 7
  %70 = zext nneg i32 %69 to i64
  %71 = getelementptr [2 x i8], ptr %27, i64 %70
  %72 = getelementptr i8, ptr %71, i64 6144
  %73 = load i16, ptr %68, align 2, !tbaa !12
  %74 = getelementptr i8, ptr %67, i64 4352
  %75 = load i16, ptr %74, align 2, !tbaa !12
  %76 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %73, i16 %75) #2, !srcloc !14
  %77 = load i16, ptr %72, align 2, !tbaa !12
  %78 = getelementptr i8, ptr %71, i64 6400
  %79 = load i16, ptr %78, align 2, !tbaa !12
  %80 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %77, i16 %79) #2, !srcloc !14
  %81 = tail call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %58, i32 %60, i32 %62, i32 %64, i32 %76, i32 %80, float %51, float %52, float %53, float %54) #3, !srcloc !15
  %82 = extractvalue { float, float, float, float } %81, 0
  %83 = extractvalue { float, float, float, float } %81, 1
  %84 = extractvalue { float, float, float, float } %81, 2
  %85 = extractvalue { float, float, float, float } %81, 3
  %86 = getelementptr inbounds nuw i8, ptr %17, i64 64
  %87 = getelementptr inbounds nuw i8, ptr %21, i64 64
  %88 = getelementptr inbounds nuw [2 x i8], ptr %86, i64 %22
  %89 = load i32, ptr %88, align 4, !tbaa !11
  %90 = getelementptr inbounds nuw [2 x i8], ptr %87, i64 %22
  %91 = load i32, ptr %90, align 4, !tbaa !11
  %92 = getelementptr inbounds nuw [2 x i8], ptr %86, i64 %23
  %93 = load i32, ptr %92, align 4, !tbaa !11
  %94 = getelementptr inbounds nuw [2 x i8], ptr %87, i64 %23
  %95 = load i32, ptr %94, align 4, !tbaa !11
  %96 = shl nuw nsw i32 %12, 7
  %97 = zext nneg i32 %96 to i64
  %98 = getelementptr [2 x i8], ptr %27, i64 %97
  %99 = getelementptr i8, ptr %98, i64 8192
  %100 = shl nuw nsw i32 %12, 7
  %101 = zext nneg i32 %100 to i64
  %102 = getelementptr [2 x i8], ptr %27, i64 %101
  %103 = getelementptr i8, ptr %102, i64 10240
  %104 = load i16, ptr %99, align 2, !tbaa !12
  %105 = getelementptr i8, ptr %98, i64 8448
  %106 = load i16, ptr %105, align 2, !tbaa !12
  %107 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %104, i16 %106) #2, !srcloc !14
  %108 = load i16, ptr %103, align 2, !tbaa !12
  %109 = getelementptr i8, ptr %102, i64 10496
  %110 = load i16, ptr %109, align 2, !tbaa !12
  %111 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %108, i16 %110) #2, !srcloc !14
  %112 = tail call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %89, i32 %91, i32 %93, i32 %95, i32 %107, i32 %111, float %82, float %83, float %84, float %85) #3, !srcloc !15
  %113 = extractvalue { float, float, float, float } %112, 0
  %114 = extractvalue { float, float, float, float } %112, 1
  %115 = extractvalue { float, float, float, float } %112, 2
  %116 = extractvalue { float, float, float, float } %112, 3
  %117 = getelementptr inbounds nuw i8, ptr %17, i64 96
  %118 = getelementptr inbounds nuw i8, ptr %21, i64 96
  %119 = getelementptr inbounds nuw [2 x i8], ptr %117, i64 %22
  %120 = load i32, ptr %119, align 4, !tbaa !11
  %121 = getelementptr inbounds nuw [2 x i8], ptr %118, i64 %22
  %122 = load i32, ptr %121, align 4, !tbaa !11
  %123 = getelementptr inbounds nuw [2 x i8], ptr %117, i64 %23
  %124 = load i32, ptr %123, align 4, !tbaa !11
  %125 = getelementptr inbounds nuw [2 x i8], ptr %118, i64 %23
  %126 = load i32, ptr %125, align 4, !tbaa !11
  %127 = shl nuw nsw i32 %12, 7
  %128 = zext nneg i32 %127 to i64
  %129 = getelementptr [2 x i8], ptr %27, i64 %128
  %130 = getelementptr i8, ptr %129, i64 12288
  %131 = shl nuw nsw i32 %12, 7
  %132 = zext nneg i32 %131 to i64
  %133 = getelementptr [2 x i8], ptr %27, i64 %132
  %134 = getelementptr i8, ptr %133, i64 14336
  %135 = load i16, ptr %130, align 2, !tbaa !12
  %136 = getelementptr i8, ptr %129, i64 12544
  %137 = load i16, ptr %136, align 2, !tbaa !12
  %138 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %135, i16 %137) #2, !srcloc !14
  %139 = load i16, ptr %134, align 2, !tbaa !12
  %140 = getelementptr i8, ptr %133, i64 14592
  %141 = load i16, ptr %140, align 2, !tbaa !12
  %142 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %139, i16 %141) #2, !srcloc !14
  %143 = tail call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %120, i32 %122, i32 %124, i32 %126, i32 %138, i32 %142, float %113, float %114, float %115, float %116) #3, !srcloc !15
  %144 = extractvalue { float, float, float, float } %143, 0
  %145 = extractvalue { float, float, float, float } %143, 1
  %146 = extractvalue { float, float, float, float } %143, 2
  %147 = extractvalue { float, float, float, float } %143, 3
  %148 = shl nuw nsw i32 %14, 7
  %149 = zext nneg i32 %148 to i64
  %150 = getelementptr inbounds nuw [4 x i8], ptr %2, i64 %149
  %151 = getelementptr inbounds [4 x i8], ptr %150, i64 %24
  %152 = getelementptr inbounds nuw [4 x i8], ptr %151, i64 %22
  %153 = shl nuw nsw i32 %18, 7
  %154 = zext nneg i32 %153 to i64
  %155 = getelementptr inbounds nuw [4 x i8], ptr %2, i64 %154
  %156 = getelementptr inbounds [4 x i8], ptr %155, i64 %24
  %157 = getelementptr inbounds nuw [4 x i8], ptr %156, i64 %22
  store float %144, ptr %152, align 4, !tbaa !16
  %158 = getelementptr inbounds nuw i8, ptr %152, i64 4
  store float %145, ptr %158, align 4, !tbaa !16
  store float %146, ptr %157, align 4, !tbaa !16
  %159 = getelementptr inbounds nuw i8, ptr %157, i64 4
  store float %147, ptr %159, align 4, !tbaa !16
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

attributes #0 = { mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx87" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind memory(none) }
attributes #3 = { nounwind }

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
!11 = !{!7, !7, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"short", !8, i64 0}
!14 = !{i64 2156912432}
!15 = !{i64 388}
!16 = !{!17, !17, i64 0}
!17 = !{!"float", !8, i64 0}
