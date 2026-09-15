; ModuleID = 'code/tc_mma.cu'
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
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca %struct.__half2, align 4
  %12 = alloca %struct.__half, align 2
  %13 = alloca %struct.__half, align 2
  %14 = alloca %struct.__half, align 8
  %15 = alloca %struct.__half, align 8
  %16 = alloca %struct.__half2, align 4
  %17 = alloca %struct.__half, align 2
  %18 = alloca %struct.__half, align 2
  %19 = alloca %struct.__half, align 8
  %20 = alloca %struct.__half, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca ptr, align 8
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  %27 = alloca ptr, align 8
  %28 = alloca i32, align 4
  %29 = alloca i32, align 4
  %30 = alloca i32, align 4
  %31 = alloca i32, align 4
  %32 = alloca i32, align 4
  %33 = alloca i32, align 4
  %34 = alloca i32, align 4
  %35 = alloca i32, align 4
  %36 = alloca i32, align 4
  %37 = alloca i32, align 4
  %38 = alloca i32, align 4
  %39 = alloca i32, align 4
  %40 = alloca i32, align 4
  %41 = alloca i32, align 4
  %42 = alloca i32, align 4
  %43 = alloca i32, align 4
  %44 = alloca i32, align 4
  %45 = alloca i32, align 4
  %46 = alloca [4 x float], align 4
  %47 = alloca i32, align 4
  %48 = alloca [4 x i32], align 4
  %49 = alloca [2 x i32], align 4
  %50 = alloca ptr, align 8
  %51 = alloca ptr, align 8
  %52 = alloca ptr, align 8
  %53 = alloca ptr, align 8
  %54 = alloca %struct.__half, align 2
  %55 = alloca %struct.__half, align 2
  %56 = alloca %struct.__half, align 2
  %57 = alloca %struct.__half, align 2
  %58 = alloca ptr, align 8
  %59 = alloca ptr, align 8
  store ptr %0, ptr %25, align 8
  store ptr %1, ptr %26, align 8
  store ptr %2, ptr %27, align 8
  store i32 %3, ptr %28, align 4
  store i32 %4, ptr %29, align 4
  store i32 %5, ptr %30, align 4
  %60 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  store i32 %60, ptr %31, align 4
  %61 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %62 = mul i32 %61, 16
  store i32 %62, ptr %32, align 4
  %63 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %64 = mul i32 %63, 8
  store i32 %64, ptr %33, align 4
  %65 = load i32, ptr %31, align 4
  %66 = ashr i32 %65, 2
  store i32 %66, ptr %34, align 4
  %67 = load i32, ptr %31, align 4
  %68 = and i32 %67, 3
  store i32 %68, ptr %35, align 4
  %69 = load i32, ptr %34, align 4
  store i32 %69, ptr %36, align 4
  %70 = load i32, ptr %34, align 4
  %71 = add nsw i32 %70, 8
  store i32 %71, ptr %37, align 4
  %72 = load i32, ptr %35, align 4
  %73 = mul nsw i32 %72, 2
  store i32 %73, ptr %38, align 4
  %74 = load i32, ptr %35, align 4
  %75 = mul nsw i32 %74, 2
  %76 = add nsw i32 %75, 8
  store i32 %76, ptr %39, align 4
  %77 = load i32, ptr %35, align 4
  %78 = mul nsw i32 %77, 2
  store i32 %78, ptr %40, align 4
  %79 = load i32, ptr %35, align 4
  %80 = mul nsw i32 %79, 2
  %81 = add nsw i32 %80, 8
  store i32 %81, ptr %41, align 4
  %82 = load i32, ptr %34, align 4
  store i32 %82, ptr %42, align 4
  %83 = load i32, ptr %34, align 4
  store i32 %83, ptr %43, align 4
  %84 = load i32, ptr %34, align 4
  %85 = add nsw i32 %84, 8
  store i32 %85, ptr %44, align 4
  %86 = load i32, ptr %35, align 4
  %87 = mul nsw i32 %86, 2
  store i32 %87, ptr %45, align 4
  call void @llvm.memset.p0.i64(ptr align 4 %46, i8 0, i64 16, i1 false)
  store i32 0, ptr %47, align 4
  br label %88

88:                                               ; preds = %230, %6
  %89 = load i32, ptr %47, align 4
  %90 = load i32, ptr %30, align 4
  %91 = icmp slt i32 %89, %90
  br i1 %91, label %92, label %233

92:                                               ; preds = %88
  %93 = load ptr, ptr %25, align 8
  %94 = load i32, ptr %32, align 4
  %95 = load i32, ptr %36, align 4
  %96 = add nsw i32 %94, %95
  %97 = load i32, ptr %30, align 4
  %98 = mul nsw i32 %96, %97
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds %struct.__half, ptr %93, i64 %99
  %101 = load i32, ptr %47, align 4
  %102 = sext i32 %101 to i64
  %103 = getelementptr inbounds %struct.__half, ptr %100, i64 %102
  store ptr %103, ptr %50, align 8
  %104 = load ptr, ptr %25, align 8
  %105 = load i32, ptr %32, align 4
  %106 = load i32, ptr %37, align 4
  %107 = add nsw i32 %105, %106
  %108 = load i32, ptr %30, align 4
  %109 = mul nsw i32 %107, %108
  %110 = sext i32 %109 to i64
  %111 = getelementptr inbounds %struct.__half, ptr %104, i64 %110
  %112 = load i32, ptr %47, align 4
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds %struct.__half, ptr %111, i64 %113
  store ptr %114, ptr %51, align 8
  %115 = load ptr, ptr %50, align 8
  %116 = load i32, ptr %38, align 4
  %117 = sext i32 %116 to i64
  %118 = getelementptr inbounds %struct.__half, ptr %115, i64 %117
  store ptr %118, ptr %21, align 8
  %119 = load ptr, ptr %21, align 8
  %120 = load i32, ptr %119, align 4
  %121 = getelementptr inbounds [4 x i32], ptr %48, i64 0, i64 0
  store i32 %120, ptr %121, align 4
  %122 = load ptr, ptr %51, align 8
  %123 = load i32, ptr %38, align 4
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds %struct.__half, ptr %122, i64 %124
  store ptr %125, ptr %22, align 8
  %126 = load ptr, ptr %22, align 8
  %127 = load i32, ptr %126, align 4
  %128 = getelementptr inbounds [4 x i32], ptr %48, i64 0, i64 1
  store i32 %127, ptr %128, align 4
  %129 = load ptr, ptr %50, align 8
  %130 = load i32, ptr %39, align 4
  %131 = sext i32 %130 to i64
  %132 = getelementptr inbounds %struct.__half, ptr %129, i64 %131
  store ptr %132, ptr %23, align 8
  %133 = load ptr, ptr %23, align 8
  %134 = load i32, ptr %133, align 4
  %135 = getelementptr inbounds [4 x i32], ptr %48, i64 0, i64 2
  store i32 %134, ptr %135, align 4
  %136 = load ptr, ptr %51, align 8
  %137 = load i32, ptr %39, align 4
  %138 = sext i32 %137 to i64
  %139 = getelementptr inbounds %struct.__half, ptr %136, i64 %138
  store ptr %139, ptr %24, align 8
  %140 = load ptr, ptr %24, align 8
  %141 = load i32, ptr %140, align 4
  %142 = getelementptr inbounds [4 x i32], ptr %48, i64 0, i64 3
  store i32 %141, ptr %142, align 4
  %143 = load ptr, ptr %26, align 8
  %144 = load i32, ptr %47, align 4
  %145 = load i32, ptr %40, align 4
  %146 = add nsw i32 %144, %145
  %147 = load i32, ptr %29, align 4
  %148 = mul nsw i32 %146, %147
  %149 = sext i32 %148 to i64
  %150 = getelementptr inbounds %struct.__half, ptr %143, i64 %149
  %151 = load i32, ptr %33, align 4
  %152 = sext i32 %151 to i64
  %153 = getelementptr inbounds %struct.__half, ptr %150, i64 %152
  %154 = load i32, ptr %42, align 4
  %155 = sext i32 %154 to i64
  %156 = getelementptr inbounds %struct.__half, ptr %153, i64 %155
  store ptr %156, ptr %52, align 8
  %157 = load ptr, ptr %26, align 8
  %158 = load i32, ptr %47, align 4
  %159 = load i32, ptr %41, align 4
  %160 = add nsw i32 %158, %159
  %161 = load i32, ptr %29, align 4
  %162 = mul nsw i32 %160, %161
  %163 = sext i32 %162 to i64
  %164 = getelementptr inbounds %struct.__half, ptr %157, i64 %163
  %165 = load i32, ptr %33, align 4
  %166 = sext i32 %165 to i64
  %167 = getelementptr inbounds %struct.__half, ptr %164, i64 %166
  %168 = load i32, ptr %42, align 4
  %169 = sext i32 %168 to i64
  %170 = getelementptr inbounds %struct.__half, ptr %167, i64 %169
  store ptr %170, ptr %53, align 8
  %171 = load ptr, ptr %52, align 8
  %172 = getelementptr inbounds %struct.__half, ptr %171, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %54, ptr align 2 %172, i64 2, i1 false)
  %173 = load ptr, ptr %52, align 8
  %174 = load i32, ptr %29, align 4
  %175 = sext i32 %174 to i64
  %176 = getelementptr inbounds %struct.__half, ptr %173, i64 %175
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %55, ptr align 2 %176, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %14, ptr align 2 %55, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %15, ptr align 2 %54, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %12, ptr align 2 %15, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %13, ptr align 2 %14, i64 2, i1 false)
  call void @_ZL14__halves2half26__halfS_(ptr dead_on_unwind writable sret(%struct.__half2) align 4 %11, ptr noundef byval(%struct.__half) align 2 %12, ptr noundef byval(%struct.__half) align 2 %13) #6
  %177 = load i32, ptr %11, align 4
  %178 = getelementptr inbounds [2 x i32], ptr %49, i64 0, i64 0
  store i32 %177, ptr %178, align 4
  %179 = load ptr, ptr %53, align 8
  %180 = getelementptr inbounds %struct.__half, ptr %179, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %56, ptr align 2 %180, i64 2, i1 false)
  %181 = load ptr, ptr %53, align 8
  %182 = load i32, ptr %29, align 4
  %183 = sext i32 %182 to i64
  %184 = getelementptr inbounds %struct.__half, ptr %181, i64 %183
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %57, ptr align 2 %184, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %19, ptr align 2 %57, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %20, ptr align 2 %56, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %17, ptr align 2 %20, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %18, ptr align 2 %19, i64 2, i1 false)
  call void @_ZL14__halves2half26__halfS_(ptr dead_on_unwind writable sret(%struct.__half2) align 4 %16, ptr noundef byval(%struct.__half) align 2 %17, ptr noundef byval(%struct.__half) align 2 %18) #6
  %185 = load i32, ptr %16, align 4
  %186 = getelementptr inbounds [2 x i32], ptr %49, i64 0, i64 1
  store i32 %185, ptr %186, align 4
  %187 = getelementptr inbounds [4 x float], ptr %46, i64 0, i64 0
  %188 = getelementptr inbounds [4 x i32], ptr %48, i64 0, i64 0
  %189 = getelementptr inbounds [2 x i32], ptr %49, i64 0, i64 0
  %190 = getelementptr inbounds [4 x float], ptr %46, i64 0, i64 0
  store ptr %187, ptr %7, align 8
  store ptr %188, ptr %8, align 8
  store ptr %189, ptr %9, align 8
  store ptr %190, ptr %10, align 8
  %191 = load ptr, ptr %7, align 8
  %192 = load ptr, ptr %7, align 8
  %193 = getelementptr inbounds float, ptr %192, i64 1
  %194 = load ptr, ptr %7, align 8
  %195 = getelementptr inbounds float, ptr %194, i64 2
  %196 = load ptr, ptr %7, align 8
  %197 = getelementptr inbounds float, ptr %196, i64 3
  %198 = load ptr, ptr %8, align 8
  %199 = load i32, ptr %198, align 4
  %200 = load ptr, ptr %8, align 8
  %201 = getelementptr inbounds i32, ptr %200, i64 1
  %202 = load i32, ptr %201, align 4
  %203 = load ptr, ptr %8, align 8
  %204 = getelementptr inbounds i32, ptr %203, i64 2
  %205 = load i32, ptr %204, align 4
  %206 = load ptr, ptr %8, align 8
  %207 = getelementptr inbounds i32, ptr %206, i64 3
  %208 = load i32, ptr %207, align 4
  %209 = load ptr, ptr %9, align 8
  %210 = load i32, ptr %209, align 4
  %211 = load ptr, ptr %9, align 8
  %212 = getelementptr inbounds i32, ptr %211, i64 1
  %213 = load i32, ptr %212, align 4
  %214 = load ptr, ptr %10, align 8
  %215 = load float, ptr %214, align 4
  %216 = load ptr, ptr %10, align 8
  %217 = getelementptr inbounds float, ptr %216, i64 1
  %218 = load float, ptr %217, align 4
  %219 = load ptr, ptr %10, align 8
  %220 = getelementptr inbounds float, ptr %219, i64 2
  %221 = load float, ptr %220, align 4
  %222 = load ptr, ptr %10, align 8
  %223 = getelementptr inbounds float, ptr %222, i64 3
  %224 = load float, ptr %223, align 4
  %225 = call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %199, i32 %202, i32 %205, i32 %208, i32 %210, i32 %213, float %215, float %218, float %221, float %224) #7, !srcloc !6
  %226 = extractvalue { float, float, float, float } %225, 0
  %227 = extractvalue { float, float, float, float } %225, 1
  %228 = extractvalue { float, float, float, float } %225, 2
  %229 = extractvalue { float, float, float, float } %225, 3
  store float %226, ptr %191, align 4
  store float %227, ptr %193, align 4
  store float %228, ptr %195, align 4
  store float %229, ptr %197, align 4
  br label %230

230:                                              ; preds = %92
  %231 = load i32, ptr %47, align 4
  %232 = add nsw i32 %231, 16
  store i32 %232, ptr %47, align 4
  br label %88, !llvm.loop !7

233:                                              ; preds = %88
  %234 = load ptr, ptr %27, align 8
  %235 = load i32, ptr %32, align 4
  %236 = load i32, ptr %43, align 4
  %237 = add nsw i32 %235, %236
  %238 = load i32, ptr %29, align 4
  %239 = mul nsw i32 %237, %238
  %240 = sext i32 %239 to i64
  %241 = getelementptr inbounds float, ptr %234, i64 %240
  %242 = load i32, ptr %33, align 4
  %243 = sext i32 %242 to i64
  %244 = getelementptr inbounds float, ptr %241, i64 %243
  %245 = load i32, ptr %45, align 4
  %246 = sext i32 %245 to i64
  %247 = getelementptr inbounds float, ptr %244, i64 %246
  store ptr %247, ptr %58, align 8
  %248 = load ptr, ptr %27, align 8
  %249 = load i32, ptr %32, align 4
  %250 = load i32, ptr %44, align 4
  %251 = add nsw i32 %249, %250
  %252 = load i32, ptr %29, align 4
  %253 = mul nsw i32 %251, %252
  %254 = sext i32 %253 to i64
  %255 = getelementptr inbounds float, ptr %248, i64 %254
  %256 = load i32, ptr %33, align 4
  %257 = sext i32 %256 to i64
  %258 = getelementptr inbounds float, ptr %255, i64 %257
  %259 = load i32, ptr %45, align 4
  %260 = sext i32 %259 to i64
  %261 = getelementptr inbounds float, ptr %258, i64 %260
  store ptr %261, ptr %59, align 8
  %262 = getelementptr inbounds [4 x float], ptr %46, i64 0, i64 0
  %263 = load float, ptr %262, align 4
  %264 = load ptr, ptr %58, align 8
  %265 = getelementptr inbounds float, ptr %264, i64 0
  store float %263, ptr %265, align 4
  %266 = getelementptr inbounds [4 x float], ptr %46, i64 0, i64 1
  %267 = load float, ptr %266, align 4
  %268 = load ptr, ptr %58, align 8
  %269 = getelementptr inbounds float, ptr %268, i64 1
  store float %267, ptr %269, align 4
  %270 = getelementptr inbounds [4 x float], ptr %46, i64 0, i64 2
  %271 = load float, ptr %270, align 4
  %272 = load ptr, ptr %59, align 8
  %273 = getelementptr inbounds float, ptr %272, i64 0
  store float %271, ptr %273, align 4
  %274 = getelementptr inbounds [4 x float], ptr %46, i64 0, i64 3
  %275 = load float, ptr %274, align 4
  %276 = load ptr, ptr %59, align 8
  %277 = getelementptr inbounds float, ptr %276, i64 1
  store float %275, ptr %277, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @mma_tc_ldmatrix(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #0 {
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i32, align 4
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca i32, align 4
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca i32, align 4
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca ptr, align 8
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca ptr, align 8
  %25 = alloca ptr, align 8
  %26 = alloca i32, align 4
  %27 = alloca i32, align 4
  %28 = alloca i32, align 4
  %29 = alloca i32, align 4
  %30 = alloca i32, align 4
  %31 = alloca i32, align 4
  %32 = alloca i32, align 4
  %33 = alloca i32, align 4
  %34 = alloca i32, align 4
  %35 = alloca i32, align 4
  %36 = alloca i32, align 4
  %37 = alloca [4 x float], align 4
  %38 = alloca i32, align 4
  %39 = alloca i32, align 4
  %40 = alloca i32, align 4
  %41 = alloca i32, align 4
  %42 = alloca i32, align 4
  %43 = alloca i32, align 4
  %44 = alloca i32, align 4
  %45 = alloca i32, align 4
  %46 = alloca i32, align 4
  %47 = alloca i32, align 4
  %48 = alloca [4 x i32], align 4
  %49 = alloca i32, align 4
  %50 = alloca [2 x i32], align 4
  %51 = alloca ptr, align 8
  %52 = alloca ptr, align 8
  store ptr %0, ptr %23, align 8
  store ptr %1, ptr %24, align 8
  store ptr %2, ptr %25, align 8
  store i32 %3, ptr %26, align 4
  store i32 %4, ptr %27, align 4
  store i32 %5, ptr %28, align 4
  %53 = call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  store i32 %53, ptr %29, align 4
  %54 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %55 = mul i32 %54, 16
  store i32 %55, ptr %30, align 4
  %56 = call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %57 = mul i32 %56, 8
  store i32 %57, ptr %31, align 4
  %58 = load i32, ptr %29, align 4
  %59 = ashr i32 %58, 2
  store i32 %59, ptr %32, align 4
  %60 = load i32, ptr %29, align 4
  %61 = and i32 %60, 3
  store i32 %61, ptr %33, align 4
  %62 = load i32, ptr %32, align 4
  store i32 %62, ptr %34, align 4
  %63 = load i32, ptr %32, align 4
  %64 = add nsw i32 %63, 8
  store i32 %64, ptr %35, align 4
  %65 = load i32, ptr %33, align 4
  %66 = mul nsw i32 %65, 2
  store i32 %66, ptr %36, align 4
  call void @llvm.memset.p0.i64(ptr align 4 %37, i8 0, i64 16, i1 false)
  store i32 0, ptr %38, align 4
  br label %67

67:                                               ; preds = %231, %6
  %68 = load i32, ptr %38, align 4
  %69 = load i32, ptr %28, align 4
  %70 = icmp slt i32 %68, %69
  br i1 %70, label %71, label %234

71:                                               ; preds = %67
  %72 = load i32, ptr %29, align 4
  %73 = mul nsw i32 %72, 8
  store i32 %73, ptr %39, align 4
  %74 = load i32, ptr %39, align 4
  %75 = ashr i32 %74, 4
  store i32 %75, ptr %40, align 4
  %76 = load i32, ptr %39, align 4
  %77 = and i32 %76, 15
  store i32 %77, ptr %41, align 4
  %78 = load i32, ptr %40, align 4
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds [16 x [16 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 0, i64 %79
  %81 = load i32, ptr %41, align 4
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds [16 x %struct.__half], ptr %80, i64 0, i64 %82
  %84 = load ptr, ptr %23, align 8
  %85 = load i32, ptr %30, align 4
  %86 = load i32, ptr %40, align 4
  %87 = add nsw i32 %85, %86
  %88 = load i32, ptr %28, align 4
  %89 = mul nsw i32 %87, %88
  %90 = sext i32 %89 to i64
  %91 = getelementptr inbounds %struct.__half, ptr %84, i64 %90
  %92 = load i32, ptr %38, align 4
  %93 = sext i32 %92 to i64
  %94 = getelementptr inbounds %struct.__half, ptr %91, i64 %93
  %95 = load i32, ptr %41, align 4
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds %struct.__half, ptr %94, i64 %96
  store ptr %83, ptr %16, align 8
  store ptr %97, ptr %17, align 8
  %98 = load ptr, ptr %16, align 8
  %99 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %98) #6
  %100 = trunc i64 %99 to i32
  store i32 %100, ptr %18, align 4
  %101 = load i32, ptr %18, align 4
  %102 = load ptr, ptr %17, align 8
  call void asm sideeffect "cp.async.cg.shared.global [$0], [$1], 16;\0A", "r,l"(i32 %101, ptr %102) #7, !srcloc !9
  %103 = load i32, ptr %29, align 4
  %104 = mul nsw i32 %103, 4
  store i32 %104, ptr %42, align 4
  %105 = load i32, ptr %42, align 4
  %106 = ashr i32 %105, 3
  store i32 %106, ptr %43, align 4
  %107 = load i32, ptr %42, align 4
  %108 = and i32 %107, 7
  store i32 %108, ptr %44, align 4
  %109 = load i32, ptr %43, align 4
  %110 = sext i32 %109 to i64
  %111 = getelementptr inbounds [16 x [8 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 0, i64 %110
  %112 = load i32, ptr %44, align 4
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [8 x %struct.__half], ptr %111, i64 0, i64 %113
  %115 = load ptr, ptr %24, align 8
  %116 = load i32, ptr %38, align 4
  %117 = load i32, ptr %43, align 4
  %118 = add nsw i32 %116, %117
  %119 = load i32, ptr %27, align 4
  %120 = mul nsw i32 %118, %119
  %121 = sext i32 %120 to i64
  %122 = getelementptr inbounds %struct.__half, ptr %115, i64 %121
  %123 = load i32, ptr %31, align 4
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds %struct.__half, ptr %122, i64 %124
  %126 = load i32, ptr %44, align 4
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds %struct.__half, ptr %125, i64 %127
  store ptr %114, ptr %13, align 8
  store ptr %128, ptr %14, align 8
  %129 = load ptr, ptr %13, align 8
  %130 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %129) #6
  %131 = trunc i64 %130 to i32
  store i32 %131, ptr %15, align 4
  %132 = load i32, ptr %15, align 4
  %133 = load ptr, ptr %14, align 8
  call void asm sideeffect "cp.async.ca.shared.global [$0], [$1], 8;\0A", "r,l"(i32 %132, ptr %133) #7, !srcloc !10
  call void asm sideeffect "cp.async.commit_group;\0A", ""() #7, !srcloc !11
  call void asm sideeffect "cp.async.wait_group 0;\0A", ""() #7, !srcloc !12
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %134 = load i32, ptr %29, align 4
  %135 = ashr i32 %134, 3
  store i32 %135, ptr %45, align 4
  %136 = load i32, ptr %29, align 4
  %137 = and i32 %136, 7
  %138 = load i32, ptr %45, align 4
  %139 = and i32 %138, 1
  %140 = icmp ne i32 %139, 0
  %141 = zext i1 %140 to i64
  %142 = select i1 %140, i32 8, i32 0
  %143 = add nsw i32 %137, %142
  store i32 %143, ptr %46, align 4
  %144 = load i32, ptr %45, align 4
  %145 = icmp sge i32 %144, 2
  %146 = zext i1 %145 to i64
  %147 = select i1 %145, i32 8, i32 0
  store i32 %147, ptr %47, align 4
  %148 = getelementptr inbounds [4 x i32], ptr %48, i64 0, i64 0
  %149 = load i32, ptr %46, align 4
  %150 = sext i32 %149 to i64
  %151 = getelementptr inbounds [16 x [16 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr), i64 0, i64 %150
  %152 = load i32, ptr %47, align 4
  %153 = sext i32 %152 to i64
  %154 = getelementptr inbounds [16 x %struct.__half], ptr %151, i64 0, i64 %153
  store ptr %148, ptr %10, align 8
  store ptr %154, ptr %11, align 8
  %155 = load ptr, ptr %11, align 8
  %156 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %155) #6
  %157 = trunc i64 %156 to i32
  store i32 %157, ptr %12, align 4
  %158 = load ptr, ptr %10, align 8
  %159 = load ptr, ptr %10, align 8
  %160 = getelementptr inbounds i32, ptr %159, i64 1
  %161 = load ptr, ptr %10, align 8
  %162 = getelementptr inbounds i32, ptr %161, i64 2
  %163 = load ptr, ptr %10, align 8
  %164 = getelementptr inbounds i32, ptr %163, i64 3
  %165 = load i32, ptr %12, align 4
  %166 = call { i32, i32, i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A", "=r,=r,=r,=r,r"(i32 %165) #7, !srcloc !13
  %167 = extractvalue { i32, i32, i32, i32 } %166, 0
  %168 = extractvalue { i32, i32, i32, i32 } %166, 1
  %169 = extractvalue { i32, i32, i32, i32 } %166, 2
  %170 = extractvalue { i32, i32, i32, i32 } %166, 3
  store i32 %167, ptr %158, align 4
  store i32 %168, ptr %160, align 4
  store i32 %169, ptr %162, align 4
  store i32 %170, ptr %164, align 4
  %171 = load i32, ptr %29, align 4
  %172 = and i32 %171, 15
  store i32 %172, ptr %49, align 4
  %173 = getelementptr inbounds [2 x i32], ptr %50, i64 0, i64 0
  %174 = load i32, ptr %49, align 4
  %175 = sext i32 %174 to i64
  %176 = getelementptr inbounds [16 x [8 x %struct.__half]], ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2Bs to ptr), i64 0, i64 %175
  %177 = getelementptr inbounds [8 x %struct.__half], ptr %176, i64 0, i64 0
  store ptr %173, ptr %7, align 8
  store ptr %177, ptr %8, align 8
  %178 = load ptr, ptr %8, align 8
  %179 = call noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %178) #6
  %180 = trunc i64 %179 to i32
  store i32 %180, ptr %9, align 4
  %181 = load ptr, ptr %7, align 8
  %182 = load ptr, ptr %7, align 8
  %183 = getelementptr inbounds i32, ptr %182, i64 1
  %184 = load i32, ptr %9, align 4
  %185 = call { i32, i32 } asm sideeffect "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0A", "=r,=r,r"(i32 %184) #7, !srcloc !14
  %186 = extractvalue { i32, i32 } %185, 0
  %187 = extractvalue { i32, i32 } %185, 1
  store i32 %186, ptr %181, align 4
  store i32 %187, ptr %183, align 4
  %188 = getelementptr inbounds [4 x float], ptr %37, i64 0, i64 0
  %189 = getelementptr inbounds [4 x i32], ptr %48, i64 0, i64 0
  %190 = getelementptr inbounds [2 x i32], ptr %50, i64 0, i64 0
  %191 = getelementptr inbounds [4 x float], ptr %37, i64 0, i64 0
  store ptr %188, ptr %19, align 8
  store ptr %189, ptr %20, align 8
  store ptr %190, ptr %21, align 8
  store ptr %191, ptr %22, align 8
  %192 = load ptr, ptr %19, align 8
  %193 = load ptr, ptr %19, align 8
  %194 = getelementptr inbounds float, ptr %193, i64 1
  %195 = load ptr, ptr %19, align 8
  %196 = getelementptr inbounds float, ptr %195, i64 2
  %197 = load ptr, ptr %19, align 8
  %198 = getelementptr inbounds float, ptr %197, i64 3
  %199 = load ptr, ptr %20, align 8
  %200 = load i32, ptr %199, align 4
  %201 = load ptr, ptr %20, align 8
  %202 = getelementptr inbounds i32, ptr %201, i64 1
  %203 = load i32, ptr %202, align 4
  %204 = load ptr, ptr %20, align 8
  %205 = getelementptr inbounds i32, ptr %204, i64 2
  %206 = load i32, ptr %205, align 4
  %207 = load ptr, ptr %20, align 8
  %208 = getelementptr inbounds i32, ptr %207, i64 3
  %209 = load i32, ptr %208, align 4
  %210 = load ptr, ptr %21, align 8
  %211 = load i32, ptr %210, align 4
  %212 = load ptr, ptr %21, align 8
  %213 = getelementptr inbounds i32, ptr %212, i64 1
  %214 = load i32, ptr %213, align 4
  %215 = load ptr, ptr %22, align 8
  %216 = load float, ptr %215, align 4
  %217 = load ptr, ptr %22, align 8
  %218 = getelementptr inbounds float, ptr %217, i64 1
  %219 = load float, ptr %218, align 4
  %220 = load ptr, ptr %22, align 8
  %221 = getelementptr inbounds float, ptr %220, i64 2
  %222 = load float, ptr %221, align 4
  %223 = load ptr, ptr %22, align 8
  %224 = getelementptr inbounds float, ptr %223, i64 3
  %225 = load float, ptr %224, align 4
  %226 = call contract { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %200, i32 %203, i32 %206, i32 %209, i32 %211, i32 %214, float %216, float %219, float %222, float %225) #7, !srcloc !6
  %227 = extractvalue { float, float, float, float } %226, 0
  %228 = extractvalue { float, float, float, float } %226, 1
  %229 = extractvalue { float, float, float, float } %226, 2
  %230 = extractvalue { float, float, float, float } %226, 3
  store float %227, ptr %192, align 4
  store float %228, ptr %194, align 4
  store float %229, ptr %196, align 4
  store float %230, ptr %198, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %231

231:                                              ; preds = %71
  %232 = load i32, ptr %38, align 4
  %233 = add nsw i32 %232, 16
  store i32 %233, ptr %38, align 4
  br label %67, !llvm.loop !15

234:                                              ; preds = %67
  %235 = load ptr, ptr %25, align 8
  %236 = load i32, ptr %30, align 4
  %237 = load i32, ptr %34, align 4
  %238 = add nsw i32 %236, %237
  %239 = load i32, ptr %27, align 4
  %240 = mul nsw i32 %238, %239
  %241 = sext i32 %240 to i64
  %242 = getelementptr inbounds float, ptr %235, i64 %241
  %243 = load i32, ptr %31, align 4
  %244 = sext i32 %243 to i64
  %245 = getelementptr inbounds float, ptr %242, i64 %244
  %246 = load i32, ptr %36, align 4
  %247 = sext i32 %246 to i64
  %248 = getelementptr inbounds float, ptr %245, i64 %247
  store ptr %248, ptr %51, align 8
  %249 = load ptr, ptr %25, align 8
  %250 = load i32, ptr %30, align 4
  %251 = load i32, ptr %35, align 4
  %252 = add nsw i32 %250, %251
  %253 = load i32, ptr %27, align 4
  %254 = mul nsw i32 %252, %253
  %255 = sext i32 %254 to i64
  %256 = getelementptr inbounds float, ptr %249, i64 %255
  %257 = load i32, ptr %31, align 4
  %258 = sext i32 %257 to i64
  %259 = getelementptr inbounds float, ptr %256, i64 %258
  %260 = load i32, ptr %36, align 4
  %261 = sext i32 %260 to i64
  %262 = getelementptr inbounds float, ptr %259, i64 %261
  store ptr %262, ptr %52, align 8
  %263 = getelementptr inbounds [4 x float], ptr %37, i64 0, i64 0
  %264 = load float, ptr %263, align 4
  %265 = load ptr, ptr %51, align 8
  %266 = getelementptr inbounds float, ptr %265, i64 0
  store float %264, ptr %266, align 4
  %267 = getelementptr inbounds [4 x float], ptr %37, i64 0, i64 1
  %268 = load float, ptr %267, align 4
  %269 = load ptr, ptr %51, align 8
  %270 = getelementptr inbounds float, ptr %269, i64 1
  store float %268, ptr %270, align 4
  %271 = getelementptr inbounds [4 x float], ptr %37, i64 0, i64 2
  %272 = load float, ptr %271, align 4
  %273 = load ptr, ptr %52, align 8
  %274 = getelementptr inbounds float, ptr %273, i64 0
  store float %272, ptr %274, align 4
  %275 = getelementptr inbounds [4 x float], ptr %37, i64 0, i64 3
  %276 = load float, ptr %275, align 4
  %277 = load ptr, ptr %52, align 8
  %278 = getelementptr inbounds float, ptr %277, i64 1
  store float %276, ptr %278, align 4
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
  %4 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  %5 = load i16, ptr %1, align 2
  %6 = load i16, ptr %2, align 2
  %7 = call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %5, i16 %6) #8, !srcloc !16
  store i32 %7, ptr %0, align 4
  ret void
}

; Function Attrs: convergent mustprogress noinline nounwind
define internal noundef i64 @_ZL24__cvta_generic_to_sharedPKv(ptr noundef %0) #5 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i64 @__nv_cvta_generic_to_shared_impl(ptr noundef %3) #6
  ret i64 %4
}

; Function Attrs: convergent mustprogress noinline nounwind
define linkonce_odr dso_local i64 @__nv_cvta_generic_to_shared_impl(ptr noundef %0) #5 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = addrspacecast ptr %3 to ptr addrspace(3)
  %5 = ptrtoint ptr addrspace(3) %4 to i64
  ret i64 %5
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
