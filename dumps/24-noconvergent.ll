; ModuleID = 'code/noconvergent_demo.cu'
source_filename = "code/noconvergent_demo.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @with_convergent(ptr nofree noundef writeonly captures(none) %0, ptr nofree noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = zext nneg i32 %3 to i64
  %5 = getelementptr inbounds nuw [4 x i8], ptr %1, i64 %4
  %6 = load float, ptr %5, align 4, !tbaa !11
  %7 = tail call contract float asm sideeffect "add.f32 $0, $1, $1;", "=f,f,0"(float %6, float %6) #3, !srcloc !13
  %8 = getelementptr inbounds nuw [4 x i8], ptr %0, i64 %4
  store float %7, ptr %8, align 4, !tbaa !11
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @with_noconvergent(ptr nofree noundef writeonly captures(none) %0, ptr nofree noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = zext nneg i32 %3 to i64
  %5 = getelementptr inbounds nuw [4 x i8], ptr %1, i64 %4
  %6 = load float, ptr %5, align 4, !tbaa !11
  %7 = tail call contract float asm sideeffect "add.f32 $0, $1, $1;", "=f,f,0"(float %6, float %6) #3, !srcloc !14
  %8 = getelementptr inbounds nuw [4 x i8], ptr %0, i64 %4
  store float %7, ptr %8, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @with_noconvergent_stmt(ptr nofree noundef writeonly captures(none) %0, ptr nofree noundef readonly captures(none) %1) local_unnamed_addr #1 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = zext nneg i32 %3 to i64
  %5 = getelementptr inbounds nuw [4 x i8], ptr %1, i64 %4
  %6 = load float, ptr %5, align 4, !tbaa !11
  %7 = tail call contract float asm sideeffect "add.f32 $0, $1, $1;", "=f,f,0"(float %6, float %6) #4, !srcloc !15
  %8 = getelementptr inbounds nuw [4 x i8], ptr %0, i64 %4
  store float %7, ptr %8, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx87" "uniform-work-group-size" }
attributes #1 = { mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx87" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nounwind }
attributes #4 = { nounwind }

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
!13 = !{i64 549}
!14 = !{i64 759}
!15 = !{i64 1138}
