; 第 10 章实验: 走 intrinsic 路径(而非 inline asm)生成 ldmatrix / mma
;
;   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 \
;       -o - code/nvvm_intrinsic_demo.ll
;
; 下面这些名字和签名都是照 /root/llvm-project 里 IntrinsicsNVVM.td 的定义写的
; (本机的 clang/llc 就是从这棵树构建出来的):
;
;   llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16
;       参数: 任意指针(地址空间写在指针类型里)   返回: {i32,i32,i32,i32}
;   llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16
;       返回: {i32,i32}
;   llvm.nvvm.mma.m16n8k16.row.col.f32.f32
;       参数: A 4 个 <2 x half>, B 2 个, C 4 个 float   返回: {float,float,float,float}
;
; 两个容易踩的点:
;   1) intrinsic 名里**没有** .shared —— 那是 PTX 指令名的一部分
;      (NVPTXIntrinsics.td 里 LDMATRIX 的 AsmString 用 Space.Suffix 拼出来)
;   2) ldmatrix 返回的是**打包的 i32 元组**, 不是 <2 x half> 聚合,
;      所以喂给 mma 之前要先 bitcast i32 -> <2 x half>

target triple = "nvptx64-nvidia-cuda"

declare {i32, i32, i32, i32}
  @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16(i8 addrspace(3)*)

declare {i32, i32}
  @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16(i8 addrspace(3)*)

declare {float, float, float, float}
  @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(
      <2 x half>, <2 x half>, <2 x half>, <2 x half>,
      <2 x half>, <2 x half>,
      float, float, float, float)

define {float, float, float, float} @ldmatrix_then_mma(i8 addrspace(3)* %as,
                                                       i8 addrspace(3)* %bs) {
  %a = call {i32, i32, i32, i32}
         @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16(i8 addrspace(3)* %as)
  %ai0 = extractvalue {i32, i32, i32, i32} %a, 0
  %ai1 = extractvalue {i32, i32, i32, i32} %a, 1
  %ai2 = extractvalue {i32, i32, i32, i32} %a, 2
  %ai3 = extractvalue {i32, i32, i32, i32} %a, 3
  %a0 = bitcast i32 %ai0 to <2 x half>
  %a1 = bitcast i32 %ai1 to <2 x half>
  %a2 = bitcast i32 %ai2 to <2 x half>
  %a3 = bitcast i32 %ai3 to <2 x half>

  %b = call {i32, i32}
         @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16(i8 addrspace(3)* %bs)
  %bi0 = extractvalue {i32, i32} %b, 0
  %bi1 = extractvalue {i32, i32} %b, 1
  %b0 = bitcast i32 %bi0 to <2 x half>
  %b1 = bitcast i32 %bi1 to <2 x half>

  %d = call {float, float, float, float}
         @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(
             <2 x half> %a0, <2 x half> %a1, <2 x half> %a2, <2 x half> %a3,
             <2 x half> %b0, <2 x half> %b1,
             float 0.0, float 0.0, float 0.0, float 0.0)
  ret {float, float, float, float} %d
}
