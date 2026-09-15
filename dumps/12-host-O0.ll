; ModuleID = 'tc_mma.cu'
source_filename = "tc_mma.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }
%struct.__half = type { i16 }
%struct.__half_raw = type { i16 }

$_ZN4dim3C2Ejjj = comdat any

$_ZN6__halfaSERK10__half_raw = comdat any

$_ZNK6__halfcv10__half_rawEv = comdat any

@.str = private unnamed_addr constant [45 x i8] c"GEMM: M=%d N=%d K=%d  (sm_89, mma.m16n8k16)\0A\00", align 1
@.str.1 = private unnamed_addr constant [9 x i8] c"malloc A\00", align 1
@.str.2 = private unnamed_addr constant [9 x i8] c"malloc B\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"malloc D\00", align 1
@.str.4 = private unnamed_addr constant [6 x i8] c"H2D A\00", align 1
@.str.5 = private unnamed_addr constant [6 x i8] c"H2D B\00", align 1
@.str.6 = private unnamed_addr constant [9 x i8] c"memset D\00", align 1
@.str.7 = private unnamed_addr constant [14 x i8] c"kernel launch\00", align 1
@.str.8 = private unnamed_addr constant [12 x i8] c"kernel sync\00", align 1
@.str.9 = private unnamed_addr constant [6 x i8] c"D2H D\00", align 1
@.str.10 = private unnamed_addr constant [35 x i8] c"[%s] max |D - ref| = %.6g   -> %s\0A\00", align 1
@.str.11 = private unnamed_addr constant [12 x i8] c"v1 manual  \00", align 1
@.str.12 = private unnamed_addr constant [12 x i8] c"v2 ldmatrix\00", align 1
@.str.13 = private unnamed_addr constant [5 x i8] c"PASS\00", align 1
@.str.14 = private unnamed_addr constant [5 x i8] c"FAIL\00", align 1
@.str.15 = private unnamed_addr constant [22 x i8] c"CUDA error at %s: %s\0A\00", align 1

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local void @__device_stub__mma_tc_manual(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #0 {
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %struct.dim3, align 8
  %14 = alloca %struct.dim3, align 8
  %15 = alloca i64, align 8
  %16 = alloca ptr, align 8
  %17 = alloca { i64, i32 }, align 8
  %18 = alloca { i64, i32 }, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  store i32 %3, ptr %10, align 4
  store i32 %4, ptr %11, align 4
  store i32 %5, ptr %12, align 4
  %19 = alloca ptr, i64 6, align 16
  %20 = getelementptr ptr, ptr %19, i32 0
  store ptr %7, ptr %20, align 8
  %21 = getelementptr ptr, ptr %19, i32 1
  store ptr %8, ptr %21, align 8
  %22 = getelementptr ptr, ptr %19, i32 2
  store ptr %9, ptr %22, align 8
  %23 = getelementptr ptr, ptr %19, i32 3
  store ptr %10, ptr %23, align 8
  %24 = getelementptr ptr, ptr %19, i32 4
  store ptr %11, ptr %24, align 8
  %25 = getelementptr ptr, ptr %19, i32 5
  store ptr %12, ptr %25, align 8
  %26 = call i32 @__cudaPopCallConfiguration(ptr %13, ptr %14, ptr %15, ptr %16)
  %27 = load i64, ptr %15, align 8
  %28 = load ptr, ptr %16, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %17, ptr align 8 %13, i64 12, i1 false)
  %29 = getelementptr inbounds nuw { i64, i32 }, ptr %17, i32 0, i32 0
  %30 = load i64, ptr %29, align 8
  %31 = getelementptr inbounds nuw { i64, i32 }, ptr %17, i32 0, i32 1
  %32 = load i32, ptr %31, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %18, ptr align 8 %14, i64 12, i1 false)
  %33 = getelementptr inbounds nuw { i64, i32 }, ptr %18, i32 0, i32 0
  %34 = load i64, ptr %33, align 8
  %35 = getelementptr inbounds nuw { i64, i32 }, ptr %18, i32 0, i32 1
  %36 = load i32, ptr %35, align 8
  %37 = call noundef i32 @cudaLaunchKernel(ptr noundef @__device_stub__mma_tc_manual, i64 %30, i32 %32, i64 %34, i32 %36, ptr noundef %19, i64 noundef %27, ptr noundef %28)
  br label %38

38:                                               ; preds = %6
  ret void
}

declare i32 @__cudaPopCallConfiguration(ptr, ptr, ptr, ptr)

declare i32 @cudaLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local void @__device_stub__mma_tc_ldmatrix(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #0 {
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %struct.dim3, align 8
  %14 = alloca %struct.dim3, align 8
  %15 = alloca i64, align 8
  %16 = alloca ptr, align 8
  %17 = alloca { i64, i32 }, align 8
  %18 = alloca { i64, i32 }, align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  store i32 %3, ptr %10, align 4
  store i32 %4, ptr %11, align 4
  store i32 %5, ptr %12, align 4
  %19 = alloca ptr, i64 6, align 16
  %20 = getelementptr ptr, ptr %19, i32 0
  store ptr %7, ptr %20, align 8
  %21 = getelementptr ptr, ptr %19, i32 1
  store ptr %8, ptr %21, align 8
  %22 = getelementptr ptr, ptr %19, i32 2
  store ptr %9, ptr %22, align 8
  %23 = getelementptr ptr, ptr %19, i32 3
  store ptr %10, ptr %23, align 8
  %24 = getelementptr ptr, ptr %19, i32 4
  store ptr %11, ptr %24, align 8
  %25 = getelementptr ptr, ptr %19, i32 5
  store ptr %12, ptr %25, align 8
  %26 = call i32 @__cudaPopCallConfiguration(ptr %13, ptr %14, ptr %15, ptr %16)
  %27 = load i64, ptr %15, align 8
  %28 = load ptr, ptr %16, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %17, ptr align 8 %13, i64 12, i1 false)
  %29 = getelementptr inbounds nuw { i64, i32 }, ptr %17, i32 0, i32 0
  %30 = load i64, ptr %29, align 8
  %31 = getelementptr inbounds nuw { i64, i32 }, ptr %17, i32 0, i32 1
  %32 = load i32, ptr %31, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %18, ptr align 8 %14, i64 12, i1 false)
  %33 = getelementptr inbounds nuw { i64, i32 }, ptr %18, i32 0, i32 0
  %34 = load i64, ptr %33, align 8
  %35 = getelementptr inbounds nuw { i64, i32 }, ptr %18, i32 0, i32 1
  %36 = load i32, ptr %35, align 8
  %37 = call noundef i32 @cudaLaunchKernel(ptr noundef @__device_stub__mma_tc_ldmatrix, i64 %30, i32 %32, i64 %34, i32 %36, ptr noundef %19, i64 noundef %27, ptr noundef %28)
  br label %38

38:                                               ; preds = %6
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca %struct.__half, align 2
  %12 = alloca i32, align 4
  %13 = alloca %struct.__half, align 2
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca float, align 4
  %17 = alloca i32, align 4
  %18 = alloca %struct.__half, align 2
  %19 = alloca %struct.__half, align 2
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca %struct.dim3, align 4
  %24 = alloca %struct.dim3, align 4
  %25 = alloca i32, align 4
  %26 = alloca %struct.dim3, align 4
  %27 = alloca %struct.dim3, align 4
  %28 = alloca { i64, i32 }, align 4
  %29 = alloca { i64, i32 }, align 4
  %30 = alloca %struct.dim3, align 4
  %31 = alloca %struct.dim3, align 4
  %32 = alloca { i64, i32 }, align 4
  %33 = alloca { i64, i32 }, align 4
  %34 = alloca double, align 8
  %35 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 256, ptr %2, align 4
  store i32 128, ptr %3, align 4
  store i32 64, ptr %4, align 4
  %36 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef 256, i32 noundef 128, i32 noundef 64) #9
  %37 = call noalias ptr @malloc(i64 noundef 32768) #10
  store ptr %37, ptr %5, align 8
  %38 = call noalias ptr @malloc(i64 noundef 16384) #10
  store ptr %38, ptr %6, align 8
  %39 = call noalias ptr @malloc(i64 noundef 131072) #10
  store ptr %39, ptr %7, align 8
  %40 = call noalias ptr @malloc(i64 noundef 131072) #10
  store ptr %40, ptr %8, align 8
  store i32 12345, ptr %9, align 4
  store i32 0, ptr %10, align 4
  br label %41

41:                                               ; preds = %52, %0
  %42 = load i32, ptr %10, align 4
  %43 = icmp slt i32 %42, 16384
  br i1 %43, label %44, label %55

44:                                               ; preds = %41
  %45 = call noundef float @_ZL5frandRj(ptr noundef nonnull align 4 dereferenceable(4) %9) #9
  %46 = call i16 @_ZL12__float2halff(float noundef %45) #9
  %47 = getelementptr inbounds nuw %struct.__half, ptr %11, i32 0, i32 0
  store i16 %46, ptr %47, align 2
  %48 = load ptr, ptr %5, align 8
  %49 = load i32, ptr %10, align 4
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds %struct.__half, ptr %48, i64 %50
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %51, ptr align 2 %11, i64 2, i1 false)
  br label %52

52:                                               ; preds = %44
  %53 = load i32, ptr %10, align 4
  %54 = add nsw i32 %53, 1
  store i32 %54, ptr %10, align 4
  br label %41, !llvm.loop !6

55:                                               ; preds = %41
  store i32 0, ptr %12, align 4
  br label %56

56:                                               ; preds = %67, %55
  %57 = load i32, ptr %12, align 4
  %58 = icmp slt i32 %57, 8192
  br i1 %58, label %59, label %70

59:                                               ; preds = %56
  %60 = call noundef float @_ZL5frandRj(ptr noundef nonnull align 4 dereferenceable(4) %9) #9
  %61 = call i16 @_ZL12__float2halff(float noundef %60) #9
  %62 = getelementptr inbounds nuw %struct.__half, ptr %13, i32 0, i32 0
  store i16 %61, ptr %62, align 2
  %63 = load ptr, ptr %6, align 8
  %64 = load i32, ptr %12, align 4
  %65 = sext i32 %64 to i64
  %66 = getelementptr inbounds %struct.__half, ptr %63, i64 %65
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %66, ptr align 2 %13, i64 2, i1 false)
  br label %67

67:                                               ; preds = %59
  %68 = load i32, ptr %12, align 4
  %69 = add nsw i32 %68, 1
  store i32 %69, ptr %12, align 4
  br label %56, !llvm.loop !8

70:                                               ; preds = %56
  store i32 0, ptr %14, align 4
  br label %71

71:                                               ; preds = %121, %70
  %72 = load i32, ptr %14, align 4
  %73 = icmp slt i32 %72, 256
  br i1 %73, label %74, label %124

74:                                               ; preds = %71
  store i32 0, ptr %15, align 4
  br label %75

75:                                               ; preds = %117, %74
  %76 = load i32, ptr %15, align 4
  %77 = icmp slt i32 %76, 128
  br i1 %77, label %78, label %120

78:                                               ; preds = %75
  store float 0.000000e+00, ptr %16, align 4
  store i32 0, ptr %17, align 4
  br label %79

79:                                               ; preds = %105, %78
  %80 = load i32, ptr %17, align 4
  %81 = icmp slt i32 %80, 64
  br i1 %81, label %82, label %108

82:                                               ; preds = %79
  %83 = load ptr, ptr %5, align 8
  %84 = load i32, ptr %14, align 4
  %85 = mul nsw i32 %84, 64
  %86 = load i32, ptr %17, align 4
  %87 = add nsw i32 %85, %86
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds %struct.__half, ptr %83, i64 %88
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %18, ptr align 2 %89, i64 2, i1 false)
  %90 = getelementptr inbounds nuw %struct.__half, ptr %18, i32 0, i32 0
  %91 = load i16, ptr %90, align 2
  %92 = call noundef float @_ZL12__half2float6__half(i16 %91) #9
  %93 = load ptr, ptr %6, align 8
  %94 = load i32, ptr %17, align 4
  %95 = mul nsw i32 %94, 128
  %96 = load i32, ptr %15, align 4
  %97 = add nsw i32 %95, %96
  %98 = sext i32 %97 to i64
  %99 = getelementptr inbounds %struct.__half, ptr %93, i64 %98
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %19, ptr align 2 %99, i64 2, i1 false)
  %100 = getelementptr inbounds nuw %struct.__half, ptr %19, i32 0, i32 0
  %101 = load i16, ptr %100, align 2
  %102 = call noundef float @_ZL12__half2float6__half(i16 %101) #9
  %103 = load float, ptr %16, align 4
  %104 = call float @llvm.fmuladd.f32(float %92, float %102, float %103)
  store float %104, ptr %16, align 4
  br label %105

105:                                              ; preds = %82
  %106 = load i32, ptr %17, align 4
  %107 = add nsw i32 %106, 1
  store i32 %107, ptr %17, align 4
  br label %79, !llvm.loop !9

108:                                              ; preds = %79
  %109 = load float, ptr %16, align 4
  %110 = load ptr, ptr %8, align 8
  %111 = load i32, ptr %14, align 4
  %112 = mul nsw i32 %111, 128
  %113 = load i32, ptr %15, align 4
  %114 = add nsw i32 %112, %113
  %115 = sext i32 %114 to i64
  %116 = getelementptr inbounds float, ptr %110, i64 %115
  store float %109, ptr %116, align 4
  br label %117

117:                                              ; preds = %108
  %118 = load i32, ptr %15, align 4
  %119 = add nsw i32 %118, 1
  store i32 %119, ptr %15, align 4
  br label %75, !llvm.loop !10

120:                                              ; preds = %75
  br label %121

121:                                              ; preds = %120
  %122 = load i32, ptr %14, align 4
  %123 = add nsw i32 %122, 1
  store i32 %123, ptr %14, align 4
  br label %71, !llvm.loop !11

124:                                              ; preds = %71
  %125 = call noundef i32 @_ZL10cudaMallocI6__halfE9cudaErrorPPT_m(ptr noundef %20, i64 noundef 32768) #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %125, ptr noundef @.str.1) #9
  %126 = call noundef i32 @_ZL10cudaMallocI6__halfE9cudaErrorPPT_m(ptr noundef %21, i64 noundef 16384) #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %126, ptr noundef @.str.2) #9
  %127 = call noundef i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(ptr noundef %22, i64 noundef 131072) #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %127, ptr noundef @.str.3) #9
  %128 = load ptr, ptr %20, align 8
  %129 = load ptr, ptr %5, align 8
  %130 = call i32 @cudaMemcpy(ptr noundef %128, ptr noundef %129, i64 noundef 32768, i32 noundef 1) #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %130, ptr noundef @.str.4) #9
  %131 = load ptr, ptr %21, align 8
  %132 = load ptr, ptr %6, align 8
  %133 = call i32 @cudaMemcpy(ptr noundef %131, ptr noundef %132, i64 noundef 16384, i32 noundef 1) #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %133, ptr noundef @.str.5) #9
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %23, i32 noundef 16, i32 noundef 16, i32 noundef 1) #9
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %24, i32 noundef 32, i32 noundef 1, i32 noundef 1) #9
  store i32 0, ptr %25, align 4
  br label %134

134:                                              ; preds = %220, %124
  %135 = load i32, ptr %25, align 4
  %136 = icmp slt i32 %135, 2
  br i1 %136, label %137, label %223

137:                                              ; preds = %134
  %138 = load ptr, ptr %22, align 8
  %139 = call i32 @cudaMemset(ptr noundef %138, i32 noundef 0, i64 noundef 131072) #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %139, ptr noundef @.str.6) #9
  %140 = load i32, ptr %25, align 4
  %141 = icmp eq i32 %140, 0
  br i1 %141, label %142, label %158

142:                                              ; preds = %137
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %26, ptr align 4 %23, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %27, ptr align 4 %24, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %28, ptr align 4 %26, i64 12, i1 false)
  %143 = getelementptr inbounds nuw { i64, i32 }, ptr %28, i32 0, i32 0
  %144 = load i64, ptr %143, align 4
  %145 = getelementptr inbounds nuw { i64, i32 }, ptr %28, i32 0, i32 1
  %146 = load i32, ptr %145, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %29, ptr align 4 %27, i64 12, i1 false)
  %147 = getelementptr inbounds nuw { i64, i32 }, ptr %29, i32 0, i32 0
  %148 = load i64, ptr %147, align 4
  %149 = getelementptr inbounds nuw { i64, i32 }, ptr %29, i32 0, i32 1
  %150 = load i32, ptr %149, align 4
  %151 = call i32 @__cudaPushCallConfiguration(i64 %144, i32 %146, i64 %148, i32 %150, i64 noundef 0, ptr noundef null) #9
  %152 = icmp ne i32 %151, 0
  br i1 %152, label %157, label %153

153:                                              ; preds = %142
  %154 = load ptr, ptr %20, align 8
  %155 = load ptr, ptr %21, align 8
  %156 = load ptr, ptr %22, align 8
  call void @__device_stub__mma_tc_manual(ptr noundef %154, ptr noundef %155, ptr noundef %156, i32 noundef 256, i32 noundef 128, i32 noundef 64) #9
  br label %157

157:                                              ; preds = %153, %142
  br label %174

158:                                              ; preds = %137
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %30, ptr align 4 %23, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %31, ptr align 4 %24, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %32, ptr align 4 %30, i64 12, i1 false)
  %159 = getelementptr inbounds nuw { i64, i32 }, ptr %32, i32 0, i32 0
  %160 = load i64, ptr %159, align 4
  %161 = getelementptr inbounds nuw { i64, i32 }, ptr %32, i32 0, i32 1
  %162 = load i32, ptr %161, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %33, ptr align 4 %31, i64 12, i1 false)
  %163 = getelementptr inbounds nuw { i64, i32 }, ptr %33, i32 0, i32 0
  %164 = load i64, ptr %163, align 4
  %165 = getelementptr inbounds nuw { i64, i32 }, ptr %33, i32 0, i32 1
  %166 = load i32, ptr %165, align 4
  %167 = call i32 @__cudaPushCallConfiguration(i64 %160, i32 %162, i64 %164, i32 %166, i64 noundef 0, ptr noundef null) #9
  %168 = icmp ne i32 %167, 0
  br i1 %168, label %173, label %169

169:                                              ; preds = %158
  %170 = load ptr, ptr %20, align 8
  %171 = load ptr, ptr %21, align 8
  %172 = load ptr, ptr %22, align 8
  call void @__device_stub__mma_tc_ldmatrix(ptr noundef %170, ptr noundef %171, ptr noundef %172, i32 noundef 256, i32 noundef 128, i32 noundef 64) #9
  br label %173

173:                                              ; preds = %169, %158
  br label %174

174:                                              ; preds = %173, %157
  %175 = call i32 @cudaGetLastError() #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %175, ptr noundef @.str.7) #9
  %176 = call i32 @cudaDeviceSynchronize() #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %176, ptr noundef @.str.8) #9
  %177 = load ptr, ptr %7, align 8
  %178 = load ptr, ptr %22, align 8
  %179 = call i32 @cudaMemcpy(ptr noundef %177, ptr noundef %178, i64 noundef 131072, i32 noundef 2) #9
  call void @_ZL5check9cudaErrorPKc(i32 noundef %179, ptr noundef @.str.9) #9
  store double 0.000000e+00, ptr %34, align 8
  store i32 0, ptr %35, align 4
  br label %180

180:                                              ; preds = %200, %174
  %181 = load i32, ptr %35, align 4
  %182 = icmp slt i32 %181, 32768
  br i1 %182, label %183, label %203

183:                                              ; preds = %180
  %184 = load double, ptr %34, align 8
  %185 = load ptr, ptr %7, align 8
  %186 = load i32, ptr %35, align 4
  %187 = sext i32 %186 to i64
  %188 = getelementptr inbounds float, ptr %185, i64 %187
  %189 = load float, ptr %188, align 4
  %190 = fpext nsz float %189 to double
  %191 = load ptr, ptr %8, align 8
  %192 = load i32, ptr %35, align 4
  %193 = sext i32 %192 to i64
  %194 = getelementptr inbounds float, ptr %191, i64 %193
  %195 = load float, ptr %194, align 4
  %196 = fpext nsz float %195 to double
  %197 = fsub nsz double %190, %196
  %198 = call nsz double @llvm.fabs.f64(double %197)
  %199 = call nsz double @llvm.maxnum.f64(double %184, double %198)
  store double %199, ptr %34, align 8
  br label %200

200:                                              ; preds = %183
  %201 = load i32, ptr %35, align 4
  %202 = add nsw i32 %201, 1
  store i32 %202, ptr %35, align 4
  br label %180, !llvm.loop !12

203:                                              ; preds = %180
  %204 = load i32, ptr %25, align 4
  %205 = icmp eq i32 %204, 0
  br i1 %205, label %206, label %207

206:                                              ; preds = %203
  br label %208

207:                                              ; preds = %203
  br label %208

208:                                              ; preds = %207, %206
  %209 = phi ptr [ @.str.11, %206 ], [ @.str.12, %207 ]
  %210 = getelementptr inbounds [12 x i8], ptr %209, i64 0, i64 0
  %211 = load double, ptr %34, align 8
  %212 = load double, ptr %34, align 8
  %213 = fcmp olt double %212, 1.000000e-02
  br i1 %213, label %214, label %215

214:                                              ; preds = %208
  br label %216

215:                                              ; preds = %208
  br label %216

216:                                              ; preds = %215, %214
  %217 = phi ptr [ @.str.13, %214 ], [ @.str.14, %215 ]
  %218 = getelementptr inbounds [5 x i8], ptr %217, i64 0, i64 0
  %219 = call i32 (ptr, ...) @printf(ptr noundef @.str.10, ptr noundef %210, double noundef %211, ptr noundef %218) #9
  br label %220

220:                                              ; preds = %216
  %221 = load i32, ptr %25, align 4
  %222 = add nsw i32 %221, 1
  store i32 %222, ptr %25, align 4
  br label %134, !llvm.loop !13

223:                                              ; preds = %134
  %224 = load ptr, ptr %20, align 8
  %225 = call i32 @cudaFree(ptr noundef %224) #9
  %226 = load ptr, ptr %21, align 8
  %227 = call i32 @cudaFree(ptr noundef %226) #9
  %228 = load ptr, ptr %22, align 8
  %229 = call i32 @cudaFree(ptr noundef %228) #9
  %230 = load ptr, ptr %5, align 8
  call void @free(ptr noundef %230) #11
  %231 = load ptr, ptr %6, align 8
  call void @free(ptr noundef %231) #11
  %232 = load ptr, ptr %7, align 8
  call void @free(ptr noundef %232) #11
  %233 = load ptr, ptr %8, align 8
  call void @free(ptr noundef %233) #11
  ret i32 0
}

declare i32 @printf(ptr noundef, ...) #2

; Function Attrs: nounwind allocsize(0)
declare noalias ptr @malloc(i64 noundef) #3

; Function Attrs: mustprogress noinline optnone uwtable
define internal i16 @_ZL12__float2halff(float noundef %0) #4 {
  %2 = alloca %struct.__half, align 2
  %3 = alloca float, align 4
  %4 = alloca %struct.__half_raw, align 2
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store float %0, ptr %3, align 4
  store i32 0, ptr %5, align 4
  store i32 0, ptr %6, align 4
  %7 = load float, ptr %3, align 4
  %8 = call noundef zeroext i16 @_ZL21__internal_float2halffRjS_(float noundef %7, ptr noundef nonnull align 4 dereferenceable(4) %5, ptr noundef nonnull align 4 dereferenceable(4) %6) #9
  %9 = getelementptr inbounds nuw %struct.__half_raw, ptr %4, i32 0, i32 0
  store i16 %8, ptr %9, align 2
  %10 = load i32, ptr %6, align 4
  %11 = icmp ugt i32 %10, -2147483648
  br i1 %11, label %21, label %12

12:                                               ; preds = %1
  %13 = load i32, ptr %6, align 4
  %14 = icmp eq i32 %13, -2147483648
  br i1 %14, label %15, label %25

15:                                               ; preds = %12
  %16 = getelementptr inbounds nuw %struct.__half_raw, ptr %4, i32 0, i32 0
  %17 = load i16, ptr %16, align 2
  %18 = zext i16 %17 to i32
  %19 = and i32 %18, 1
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %21, label %25

21:                                               ; preds = %15, %1
  %22 = getelementptr inbounds nuw %struct.__half_raw, ptr %4, i32 0, i32 0
  %23 = load i16, ptr %22, align 2
  %24 = add i16 %23, 1
  store i16 %24, ptr %22, align 2
  br label %25

25:                                               ; preds = %21, %15, %12
  %26 = call noundef nonnull align 2 dereferenceable(2) ptr @_ZN6__halfaSERK10__half_raw(ptr noundef nonnull align 2 dereferenceable(2) %2, ptr noundef nonnull align 2 dereferenceable(2) %4) #9
  %27 = getelementptr inbounds nuw %struct.__half, ptr %2, i32 0, i32 0
  %28 = load i16, ptr %27, align 2
  ret i16 %28
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal noundef float @_ZL5frandRj(ptr noundef nonnull align 4 dereferenceable(4) %0) #5 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8, !nonnull !14, !align !15
  %4 = load i32, ptr %3, align 4
  %5 = mul i32 %4, 1103515245
  %6 = add i32 %5, 12345
  %7 = load ptr, ptr %2, align 8, !nonnull !14, !align !15
  store i32 %6, ptr %7, align 4
  %8 = load ptr, ptr %2, align 8, !nonnull !14, !align !15
  %9 = load i32, ptr %8, align 4
  %10 = lshr i32 %9, 8
  %11 = and i32 %10, 65535
  %12 = uitofp i32 %11 to float
  %13 = fdiv float %12, 3.276800e+04
  %14 = fsub float %13, 1.000000e+00
  ret float %14
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal noundef float @_ZL12__half2float6__half(i16 %0) #4 {
  %2 = alloca %struct.__half, align 2
  %3 = alloca float, align 4
  %4 = alloca %struct.__half_raw, align 2
  %5 = getelementptr inbounds nuw %struct.__half, ptr %2, i32 0, i32 0
  store i16 %0, ptr %5, align 2
  %6 = call i16 @_ZNK6__halfcv10__half_rawEv(ptr noundef nonnull align 2 dereferenceable(2) %2) #9
  %7 = getelementptr inbounds nuw %struct.__half_raw, ptr %4, i32 0, i32 0
  store i16 %6, ptr %7, align 2
  %8 = getelementptr inbounds nuw %struct.__half_raw, ptr %4, i32 0, i32 0
  %9 = load i16, ptr %8, align 2
  %10 = call noundef float @_ZL21__internal_half2floatt(i16 noundef zeroext %9) #9
  store float %10, ptr %3, align 4
  %11 = load float, ptr %3, align 4
  ret float %11
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #6

; Function Attrs: mustprogress noinline optnone uwtable
define internal void @_ZL5check9cudaErrorPKc(i32 noundef %0, ptr noundef %1) #4 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  %5 = load i32, ptr %3, align 4
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = load ptr, ptr %4, align 8
  %9 = load i32, ptr %3, align 4
  %10 = call ptr @cudaGetErrorString(i32 noundef %9) #9
  %11 = call i32 (ptr, ...) @printf(ptr noundef @.str.15, ptr noundef %8, ptr noundef %10) #9
  call void @exit(i32 noundef 1) #12
  unreachable

12:                                               ; preds = %2
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal noundef i32 @_ZL10cudaMallocI6__halfE9cudaErrorPPT_m(ptr noundef %0, i64 noundef %1) #4 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load i64, ptr %4, align 8
  %7 = call i32 @cudaMalloc(ptr noundef %5, i64 noundef %6) #9
  ret i32 %7
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal noundef i32 @_ZL10cudaMallocIfE9cudaErrorPPT_m(ptr noundef %0, i64 noundef %1) #4 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load i64, ptr %4, align 8
  %7 = call i32 @cudaMalloc(ptr noundef %5, i64 noundef %6) #9
  ret i32 %7
}

declare i32 @cudaMemcpy(ptr noundef, ptr noundef, i64 noundef, i32 noundef) #2

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) unnamed_addr #5 comdat align 2 {
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store i32 %1, ptr %6, align 4
  store i32 %2, ptr %7, align 4
  store i32 %3, ptr %8, align 4
  %9 = load ptr, ptr %5, align 8
  %10 = getelementptr inbounds nuw %struct.dim3, ptr %9, i32 0, i32 0
  %11 = load i32, ptr %6, align 4
  store i32 %11, ptr %10, align 4
  %12 = getelementptr inbounds nuw %struct.dim3, ptr %9, i32 0, i32 1
  %13 = load i32, ptr %7, align 4
  store i32 %13, ptr %12, align 4
  %14 = getelementptr inbounds nuw %struct.dim3, ptr %9, i32 0, i32 2
  %15 = load i32, ptr %8, align 4
  store i32 %15, ptr %14, align 4
  ret void
}

declare i32 @cudaMemset(ptr noundef, i32 noundef, i64 noundef) #2

declare i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64 noundef, ptr noundef) #2

declare i32 @cudaGetLastError() #2

declare i32 @cudaDeviceSynchronize() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maxnum.f64(double, double) #6

declare i32 @cudaFree(ptr noundef) #2

; Function Attrs: nounwind
declare void @free(ptr noundef) #7

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal noundef zeroext i16 @_ZL21__internal_float2halffRjS_(float noundef %0, ptr noundef nonnull align 4 dereferenceable(4) %1, ptr noundef nonnull align 4 dereferenceable(4) %2) #5 {
  %4 = alloca float, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  store float %0, ptr %4, align 4
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %7, ptr align 4 %4, i64 4, i1 false)
  %13 = load i32, ptr %7, align 4
  %14 = and i32 %13, 2147483647
  store i32 %14, ptr %8, align 4
  %15 = load i32, ptr %7, align 4
  %16 = lshr i32 %15, 16
  %17 = and i32 %16, 32768
  %18 = load ptr, ptr %5, align 8, !nonnull !14, !align !15
  store i32 %17, ptr %18, align 4
  %19 = load i32, ptr %8, align 4
  %20 = icmp uge i32 %19, 2139095040
  br i1 %20, label %21, label %32

21:                                               ; preds = %3
  %22 = load ptr, ptr %6, align 8, !nonnull !14, !align !15
  store i32 0, ptr %22, align 4
  %23 = load i32, ptr %8, align 4
  %24 = icmp eq i32 %23, 2139095040
  br i1 %24, label %25, label %29

25:                                               ; preds = %21
  %26 = load ptr, ptr %5, align 8, !nonnull !14, !align !15
  %27 = load i32, ptr %26, align 4
  %28 = or i32 %27, 31744
  br label %30

29:                                               ; preds = %21
  br label %30

30:                                               ; preds = %29, %25
  %31 = phi i32 [ %28, %25 ], [ 32767, %29 ]
  store i32 %31, ptr %9, align 4
  br label %87

32:                                               ; preds = %3
  %33 = load i32, ptr %8, align 4
  %34 = icmp ugt i32 %33, 1199566847
  br i1 %34, label %35, label %40

35:                                               ; preds = %32
  %36 = load ptr, ptr %6, align 8, !nonnull !14, !align !15
  store i32 -2147483648, ptr %36, align 4
  %37 = load ptr, ptr %5, align 8, !nonnull !14, !align !15
  %38 = load i32, ptr %37, align 4
  %39 = or i32 %38, 31743
  store i32 %39, ptr %9, align 4
  br label %86

40:                                               ; preds = %32
  %41 = load i32, ptr %8, align 4
  %42 = icmp uge i32 %41, 947912704
  br i1 %42, label %43, label %54

43:                                               ; preds = %40
  %44 = load i32, ptr %8, align 4
  %45 = shl i32 %44, 19
  %46 = load ptr, ptr %6, align 8, !nonnull !14, !align !15
  store i32 %45, ptr %46, align 4
  %47 = load i32, ptr %8, align 4
  %48 = sub i32 %47, 939524096
  store i32 %48, ptr %8, align 4
  %49 = load ptr, ptr %5, align 8, !nonnull !14, !align !15
  %50 = load i32, ptr %49, align 4
  %51 = load i32, ptr %8, align 4
  %52 = lshr i32 %51, 13
  %53 = or i32 %50, %52
  store i32 %53, ptr %9, align 4
  br label %85

54:                                               ; preds = %40
  %55 = load i32, ptr %8, align 4
  %56 = icmp ult i32 %55, 855638017
  br i1 %56, label %57, label %62

57:                                               ; preds = %54
  %58 = load i32, ptr %8, align 4
  %59 = load ptr, ptr %6, align 8, !nonnull !14, !align !15
  store i32 %58, ptr %59, align 4
  %60 = load ptr, ptr %5, align 8, !nonnull !14, !align !15
  %61 = load i32, ptr %60, align 4
  store i32 %61, ptr %9, align 4
  br label %84

62:                                               ; preds = %54
  %63 = load i32, ptr %8, align 4
  %64 = lshr i32 %63, 23
  store i32 %64, ptr %10, align 4
  %65 = load i32, ptr %10, align 4
  %66 = sub i32 126, %65
  store i32 %66, ptr %11, align 4
  %67 = load i32, ptr %8, align 4
  %68 = and i32 %67, 8388607
  store i32 %68, ptr %12, align 4
  %69 = load i32, ptr %12, align 4
  %70 = or i32 %69, 8388608
  store i32 %70, ptr %12, align 4
  %71 = load i32, ptr %12, align 4
  %72 = load i32, ptr %11, align 4
  %73 = sub i32 32, %72
  %74 = shl i32 %71, %73
  %75 = load ptr, ptr %6, align 8, !nonnull !14, !align !15
  store i32 %74, ptr %75, align 4
  %76 = load ptr, ptr %5, align 8, !nonnull !14, !align !15
  %77 = load i32, ptr %76, align 4
  %78 = load i32, ptr %12, align 4
  %79 = load i32, ptr %11, align 4
  %80 = lshr i32 %78, %79
  %81 = or i32 %77, %80
  store i32 %81, ptr %9, align 4
  %82 = load i32, ptr %9, align 4
  %83 = and i32 %82, 65535
  store i32 %83, ptr %9, align 4
  br label %84

84:                                               ; preds = %62, %57
  br label %85

85:                                               ; preds = %84, %43
  br label %86

86:                                               ; preds = %85, %35
  br label %87

87:                                               ; preds = %86, %30
  %88 = load i32, ptr %9, align 4
  %89 = trunc i32 %88 to i16
  ret i16 %89
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local noundef nonnull align 2 dereferenceable(2) ptr @_ZN6__halfaSERK10__half_raw(ptr noundef nonnull align 2 dereferenceable(2) %0, ptr noundef nonnull align 2 dereferenceable(2) %1) #5 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8, !nonnull !14, !align !16
  %7 = getelementptr inbounds nuw %struct.__half_raw, ptr %6, i32 0, i32 0
  %8 = load i16, ptr %7, align 2
  %9 = getelementptr inbounds nuw %struct.__half, ptr %5, i32 0, i32 0
  store i16 %8, ptr %9, align 2
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal noundef float @_ZL21__internal_half2floatt(i16 noundef zeroext %0) #5 {
  %2 = alloca i16, align 2
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca float, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store i16 %0, ptr %2, align 2
  %9 = load i16, ptr %2, align 2
  %10 = zext i16 %9 to i32
  %11 = lshr i32 %10, 15
  %12 = and i32 %11, 1
  store i32 %12, ptr %3, align 4
  %13 = load i16, ptr %2, align 2
  %14 = zext i16 %13 to i32
  %15 = lshr i32 %14, 10
  %16 = and i32 %15, 31
  store i32 %16, ptr %4, align 4
  %17 = load i16, ptr %2, align 2
  %18 = zext i16 %17 to i32
  %19 = and i32 %18, 1023
  %20 = shl i32 %19, 13
  store i32 %20, ptr %5, align 4
  %21 = load i32, ptr %4, align 4
  %22 = icmp eq i32 %21, 31
  br i1 %22, label %23, label %37

23:                                               ; preds = %1
  %24 = load i32, ptr %5, align 4
  %25 = icmp ne i32 %24, 0
  br i1 %25, label %26, label %29

26:                                               ; preds = %23
  %27 = load i32, ptr %3, align 4
  %28 = lshr i32 %27, 1
  br label %31

29:                                               ; preds = %23
  %30 = load i32, ptr %3, align 4
  br label %31

31:                                               ; preds = %29, %26
  %32 = phi i32 [ %28, %26 ], [ %30, %29 ]
  store i32 %32, ptr %3, align 4
  %33 = load i32, ptr %5, align 4
  %34 = icmp ne i32 %33, 0
  %35 = zext i1 %34 to i64
  %36 = select i1 %34, i32 8388607, i32 0
  store i32 %36, ptr %5, align 4
  store i32 255, ptr %4, align 4
  br label %62

37:                                               ; preds = %1
  %38 = load i32, ptr %4, align 4
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %58

40:                                               ; preds = %37
  %41 = load i32, ptr %5, align 4
  %42 = icmp ne i32 %41, 0
  br i1 %42, label %43, label %57

43:                                               ; preds = %40
  store i32 113, ptr %4, align 4
  br label %44

44:                                               ; preds = %51, %43
  %45 = load i32, ptr %5, align 4
  %46 = and i32 %45, 4194304
  store i32 %46, ptr %7, align 4
  %47 = load i32, ptr %5, align 4
  %48 = shl i32 %47, 1
  store i32 %48, ptr %5, align 4
  %49 = load i32, ptr %4, align 4
  %50 = add i32 %49, -1
  store i32 %50, ptr %4, align 4
  br label %51

51:                                               ; preds = %44
  %52 = load i32, ptr %7, align 4
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %44, label %54, !llvm.loop !17

54:                                               ; preds = %51
  %55 = load i32, ptr %5, align 4
  %56 = and i32 %55, 8388607
  store i32 %56, ptr %5, align 4
  br label %57

57:                                               ; preds = %54, %40
  br label %61

58:                                               ; preds = %37
  %59 = load i32, ptr %4, align 4
  %60 = add i32 %59, 112
  store i32 %60, ptr %4, align 4
  br label %61

61:                                               ; preds = %58, %57
  br label %62

62:                                               ; preds = %61, %31
  %63 = load i32, ptr %3, align 4
  %64 = shl i32 %63, 31
  %65 = load i32, ptr %4, align 4
  %66 = shl i32 %65, 23
  %67 = or i32 %64, %66
  %68 = load i32, ptr %5, align 4
  %69 = or i32 %67, %68
  store i32 %69, ptr %8, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %6, ptr align 4 %8, i64 4, i1 false)
  %70 = load float, ptr %6, align 4
  ret float %70
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local i16 @_ZNK6__halfcv10__half_rawEv(ptr noundef nonnull align 2 dereferenceable(2) %0) #5 comdat align 2 {
  %2 = alloca %struct.__half_raw, align 2
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds nuw %struct.__half, ptr %4, i32 0, i32 0
  %6 = load i16, ptr %5, align 2
  %7 = getelementptr inbounds nuw %struct.__half_raw, ptr %2, i32 0, i32 0
  store i16 %6, ptr %7, align 2
  %8 = getelementptr inbounds nuw %struct.__half_raw, ptr %2, i32 0, i32 0
  %9 = load i16, ptr %8, align 2
  ret i16 %9
}

declare ptr @cudaGetErrorString(i32 noundef) #2

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef) #8

declare i32 @cudaMalloc(ptr noundef, i64 noundef) #2

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #3 = { nounwind allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #4 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #5 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #8 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #9 = { "uniform-work-group-size" }
attributes #10 = { nounwind allocsize(0) "uniform-work-group-size" }
attributes #11 = { nounwind "uniform-work-group-size" }
attributes #12 = { noreturn nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 8]}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 24.0.0git (ssh://git@ssh.github.com:443/llvm/llvm-project.git 677a4c33ba942fe7aec6a1be15ba388f1b74d855)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = !{}
!15 = !{i64 4}
!16 = !{i64 2}
!17 = distinct !{!17, !7}
