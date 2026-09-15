# 第 10 章 · 反推练习：给一条新 ISA 指令做前端接入

## 10.0 题目

假设 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 是目标 ISA 里**已经有、但 LLVM 还不认识**的一条新指令——比如你在给一家新硬件厂写后端，或者要给 MLIR 里的一个 tensor 方言做 lowering。

问题：**从零到能用，需要动哪些文件、写哪些东西、怎么验证？**

这一章我们不泛泛而谈，每一步都指到真实源码，最后**亲手把 intrinsic 路径跑通**给你看。

## 10.1 先做决策：三条路，选一条

| 路径 | 要不要改编译器 | 谁能用 | 本课程的例子 |
| --- | --- | --- | --- |
| **A. inline asm** | 不改 | CUDA C++ 程序员 | 我们的 `tc_mma.cu`（第 2–9 章全在讲它） |
| **B. 现成的 LLVM IR intrinsic** | 不改（但要写 IR） | 手写 IR、其他前端、MLIR 后端 | 本章 10.5 的实验 |
| **C. 新增 intrinsic + TableGen pattern** | 改 LLVM 源码 | 编译器工程师 | 10.4 节的六步法 |

决策树很简单：

```
你要在 .cu 里手写 kernel 吗？
  ├─ 是 → 走 A。写完直接用，代价是优化器不认识这段汇编（convergent，第 5 章）
  └─ 否 → 你要让"某个上游"自动生成这条指令吗？
           ├─ 是 → 走 C（或者用别人已经加好的 intrinsic 走 B）
           └─ 否 → 走 B，手写 IR 调 intrinsic
```

**一个必须知道的事实**：在我们的环境里，CUDA 的 `mma` / `ldmatrix` **没有 clang 内建函数**（没有 `__nvvm_mma_m16n8k16_*` 这种 builtin，clang 的 NVPTX builtin 表里有的是 `__hmma_*` 那批老 wmma 的和 sreg 读取的）。所以 CUDA C++ 用户只有两条路：inline asm，或者自己写 IR。这就是为什么 CUDA 世界里 hand-written kernel 清一色是 `asm volatile`。

## 10.2 把指令规格化成编译器要的六件事

动手之前，先把手册上的那一行翻译成编译器需要的六项信息。以我们的指令为例：

| # | 项目 | 内容 |
| --- | --- | --- |
| 1 | **语义** | `D = A × B + C`，warp 级 |
| 2 | **形状** | M=16, N=8, K=16 |
| 3 | **操作数类型与数量** | A: 16×16 f16 → 每线程 8 half；B: 16×8 f16 → 4 half；C/D: 16×8 f32 → 4 float |
| 4 | **寄存器类** | A/B 用 32 位整型寄存器（打包 2 个 f16），C/D 用 f32 寄存器 |
| 5 | **内存与副作用** | 不访问内存；但会隐式影响 warp 状态 |
| 6 | **收敛性** | **是**（warp 必须一起执行） |

把第 4 项算出来："每线程寄存器数 = 总元素数 ÷ 32"，得到 `A=4, B=2, C=D=4`（第 8.2 节推导过）。

第 6 项决定了它在 IR 里必须带 `convergent` 属性——这一条如果漏了，你会得到"大部分时候对、偶尔错"的 kernel，是最难查的 bug。

## 10.3 路径 A 的实现清单（本课程已验证）

如果你选 inline asm，需要做四件事，我们全都做过了：

1. **写约束**：`"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"`——4 输出 + 10 输入，`f` 对应浮点寄存器类、`r` 对应 32 位整型（第 3.6 节）。
2. **接受 convergent**：CUDA 设备端的所有 asm 都会被标 convergent，且拿不掉（第 3.6、5.6 节实测三条路都失败）。
3. **管理 fragment 布局**：A/B/C/D 四张表（第 2.2 节），错了就是 `max |D-ref| = 13.9` 那种量级的错。
4. **验证到底层**：PTX 里确认指令出来了（第 8 章），SASS 里确认 `HMMA.16816.F32` 的操作数符合预期（第 9 章）。

**结**：路径 A 不需要动 LLVM 一行代码，但你要自己承担布局、对齐、收敛性三件事。

## 10.4 路径 C：六步接入法

下面是"要让 LLVM 自己生成这条指令"的完整流程。每一步我都给出**真实源码位置**（本机 llc/clang 就是从 `/root/llvm-project` 这棵树构建的，所以行号和你手上的源码能对上），并说明要写什么。

### Step 1 · 定义 intrinsic：`llvm/include/llvm/IR/IntrinsicsNVVM.td`

NVVM 的 mma/ldmatrix intrinsic 不是一个个手写的，而是 TableGen 批量生成的。三个关键类：

```
class WMMA_REGS<string Geom, string Frag, string PtxEltType, bit IsSparse = false>   // 第 208 行
class NVVM_MMA<WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D>                  // 第 2956 行
class NVVM_LDMATRIX<WMMA_REGS Frag, int Transposed>                                  // 第 3108 行
```

`WMMA_REGS` 干的事就是**把"指令规格化的六件事"编码成 TableGen 数据**：

```tablegen
// 摘录：m16n8k16 的四个 fragment 各占几个寄存器、什么类型
!eq(gft,"m16n8k16:a:f16") : !listsplat(llvm_v2f16_ty, 4),   // A = 4 个 <2 x f16>
!eq(gft,"m16n8k16:b:f16") : !listsplat(llvm_v2f16_ty, 2),   // B = 2 个
!eq(gft,"m16n8k16:c:f32") : !listsplat(llvm_float_ty, 4),   // C = 4 个 f32
!eq(gft,"m16n8k16:d:f32") : !listsplat(llvm_float_ty, 4),   // D = 4 个
```

而 `NVVM_LDMATRIX` 顺手把 convergent 也标上了：

```tablegen
class NVVM_LDMATRIX<WMMA_REGS Frag, int Transposed>
  : Intrinsic<Frag.regs, [llvm_anyptr_ty],
              [IntrReadMem, IntrArgMemOnly, IntrNoCallback, IntrConvergent,
               ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>],
              LDMATRIX_NAME<Frag, Transposed>.intr_name>;
```

**注意 `IntrConvergent`**——Step 1 就要把它写对，否则 Step 5 的测试再全也没用。

intrinsic 名字的拼法也有脚手架（`LDMATRIX_NAME` / `MMA_NAME`）：

```
llvm.nvvm.ldmatrix.sync.aligned.<geom>.<frag>[.trans].<type>
llvm.nvvm.mma.<geom>.<alayout>.<blayout>.<signature>
   ↑ f16 运算的 signature 只写 D.C → 我们的指令是 ...row.col.f32.f32
```

### Step 2 · 定义机器指令 + pattern：`llvm/lib/Target/NVPTX/NVPTXIntrinsics.td`

同样是一批 TableGen 类：

```
class MMA<WMMA_REGINFO FragA, ...>        // 第 5974 行
class LDMATRIX<WMMA_REGINFO Frag, bit Transposed, NVPTXAddressSpace Space>   // 第 6220 行
```

看 `LDMATRIX` 的实现，它把"PTX 汇编字符串怎么拼"和"地址空间怎么约束"都写清楚了（`NVPTXIntrinsics.td:6220`）：

```tablegen
class LDMATRIX<WMMA_REGINFO Frag, bit Transposed, NVPTXAddressSpace Space>
  : WMMA_INSTR<LDMATRIX_NAME<Frag, Transposed>.record_name, [(ins ADDR:$src)]>,
    Requires<Frag.Predicates> {
  // ldmatrix is overloaded on pointer's address space, so only match the
  // intrinsic when it accesses this instruction's address space.
  let IntrinsicPattern = BuildPattern<IntrinsicInAS<Intr, Space>, Args>.ret;

  let OutOperandList = Frag.Outs;
  let InOperandList = !con(Args, (ins MmaCode:$ptx));
  let AsmString = "ldmatrix.sync.aligned."
                  # Frag.geom # "." # Frag.frag
                  # !if(Transposed, ".trans", "")
                  # Space.Suffix
                  # "." # Frag.ptx_elt_type
                  # " " # Frag.regstring # ", [$src];";
}
```

**这段代码把第 10.5 节那个坑解释得明明白白**：intrinsic 的名字由 `LDMATRIX_NAME` 拼（**不带空间**），空间信息放在 `NVPTXAddressSpace Space` 这个模板参数和 `IntrinsicPattern`（按指针的地址空间匹配）里；而 **PTX 汇编串里的 `.shared` 来自 `Space.Suffix`**。也就是说，`.shared` 只出现在 PTX 那一侧——IR 里写它当然是错的。

`MMA` 类同理，把 `mma.sync.aligned.<geom>.<alayout>.<blayout>[.kind][.satfinite]<TypeList>` 拼出来，`TypeList` 就是 `.f32.f16.f16.f32` 那一段。

**换句话说：Step 2 写的不是"代码"，是"指令名字的拼装规则 + 操作数类型约束"。** 这也是 NVPTX 后端最舒服的地方——PTX 和 IR 的抽象层次太接近了。

### Step 3 · 登记内存行为：`llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp`

一个很多人会漏的步骤。ldmatrix 是"从共享内存读"，如果不告诉 DAG 这件事，调度器和别名分析就是瞎的。看 `getTgtMemIntrinsic`（第 4299 行）里那一长串 case：

```cpp
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_b16:          // 第 4418 行
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8:
  ...
```

这一段的作用是：告诉 SelectionDAG"这个 intrinsic 会读 addrspace(3) 的内存，读的字节数是 X"。**新加一条访存类 intrinsic，这一步不能省。**

### Step 4 · 前端识别（如果上游是 C/C++）

如果你的目标是"让 C/C++ 程序员也能用"，还需要一个 builtin。clang 的 NVPTX builtin 表在这里：

```
clang/include/clang/Basic/BuiltinsNVPTX.td      (这棵树里只剩 .td 这一份了)
```

里面已经有旧一代的 `__hmma_*` 系列：

```tablegen
def __hmma_m16n16k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX60>;   // 第 1177 行
def __hmma_m16n16k16_ld_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_70, PTX60>;
```

clang 把 builtin 名字映射到 intrinsic 的机制在第 3.6 节讲过（`CGBuiltin.cpp` 的 `getIntrinsicForClangBuiltin`，靠 `ClangBuiltin<"...">` 关联的名字）。

**但对于 `mma.sync`，上游一直没做这个 builtin**——所以现实里 CUDA C++ 用户用的是 inline asm。你要自己做的话，路径就是：在 builtin 表里声明 → 在 `IntrinsicsNVVM.td` 里用 `ClangBuiltin<"__nvvm_mma_...">` 关联 → 打补丁重编 clang。

### Step 5 · 测试

LLVM 对这类指令的测试是**生成式**的，这招很值得学：`llvm/test/CodeGen/NVPTX/wmma.py` 用 Python 遍历所有 (geom × layout × kind × satfinite) 组合，生成 IR 并配上 FileCheck 断言。mma 那部分是 `gen_mma_tests()`（第 1109 行起）：

```python
mma_intrinsic_template = "llvm.nvvm.mma${b1op}.${geom}.${alayout}.${blayout}${kind}${satf}.${intrinsic_signature}"
mma_instruction_template = "mma.sync${aligned}.${geom}.${alayout}.${blayout}${kind}${satf}.${ptx_signature}${b1op}"
```

生成出来的测试形如：

```
; CHECK: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
; CHECK-NEXT: { ...
```

另外还有一个**专门测收敛语义**的用例，你要加新 mma 时必须照着它写一份：

`llvm/test/CodeGen/NVPTX/mma-no-sink-after-laneid-check.ll`：

```llvm
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx81 | FileCheck %s

declare { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32, i32, i32, float, float, float, float) #1

; COM: llvm.nvvm.mma should not sink to the next block and gets reordered to be after laneid check.
define dso_local void @no_reorder_mma_and_laneid_check(ptr %arg, ptr %arg1) {
bb:
  ; CHECK: mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
  ; CHECK: laneid
  %i = tail call { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32 10, i32 10, i32 8, float 0.0, ...)
  %i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  ...
}
```

**这份官方测试的存在，正好给第 5 章的结论盖了章**：`llvm.nvvm.mma` 不会被下沉到 laneid 检查之后——因为它是 convergent 的。你加新指令时，这条测试就是模板。

### Step 6 · 验证三段

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 -o - demo.ll   # 看 PTX
ptxas -arch=sm_89 -v demo.ptx -o demo.cubin                                # 看 SASS 与资源
cuobjdump -sass demo.cubin | grep HMMA                                     # 确认落到硬件指令
```

## 10.5 亲手验证：intrinsic 路径真的能选出指令

光讲不算数。我们写了一份纯 IR 的 demo（`code/nvvm_intrinsic_demo.ll`），里面**没有一行内联汇编**，全靠 intrinsic：

```llvm
declare {i32, i32, i32, i32}
  @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16(i8 addrspace(3)*)
declare {i32, i32}
  @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16(i8 addrspace(3)*)
declare {float, float, float, float}
  @llvm.nvvm.mma.m16n8k16.row.col.f32.f32(
      <2 x half>, <2 x half>, <2 x half>, <2 x half>,
      <2 x half>, <2 x half>,
      float, float, float, float)
```

跑 `llc`：

```bash
$ llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 \
      -o - code/nvvm_intrinsic_demo.ll
```

```ptx
	ld.param::func.b64 	%rd1, [ldmatrix_then_mma_param_0];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd1];
	ld.param::func.b64 	%rd2, [ldmatrix_then_mma_param_1];
	ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r5, %r6}, [%rd2];
	mov.b32 	%r7, 0f00000000;
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
		{%r8, %r9, %r10, %r11},
		{%r1, %r2, %r3, %r4},
		{%r5, %r6},
		{%r7, %r7, %r7, %r7};
```

**成了。** 三条 intrinsic 一一对应三条 PTX 指令：

| IR intrinsic | PTX |
| --- | --- |
| `llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16` | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` |
| `llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16` | `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` |
| `llvm.nvvm.mma.m16n8k16.row.col.f32.f32` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |

而且 **ldmatrix 输出的 `{%r1,%r2,%r3,%r4}` 直接成了 mma 的 A 操作数**——中间那几条 `bitcast`（i32 → `<2 x half>`）在 ISel 里是免费的。这和我们在第 8 章看到的 inline asm 版本的结果**完全一致**：

```
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r31,%r32,%r33,%r34}, [%r35];
	ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r36,%r37}, [%r38];
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f27,%f28,%f29,%f30}, {%r31,%r32,%r33,%r34}, {%r36,%r37}, {%f27,%f28,%f29,%f30};
```

### 名字和签名都要对：三组实测对照

上面那份 demo 能跑通，是因为它的**名字和签名都跟这套 LLVM 的 intrinsic 定义完全一致**。只要有一边对不上，就会出问题。三个实测的对照组：

**对照组 A：名字正确、签名正确 → 出指令。**

```llvm
declare {i32, i32, i32, i32}
  @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16(i8 addrspace(3)*)
```

```
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd1];
```

**对照组 B：名字多了 `.shared`（照直觉猜的写法）→ 后端报错。**

```llvm
declare {i32, i32, i32, i32}
  @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.shared.b16(i8 addrspace(3)*)
```

```
error: call to unknown intrinsic 'llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.shared.b16'
       cannot be lowered by the NVPTX backend
   call.uni (retval0), llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.shared.b16, (param0);
```

**对照组 C：名字正确、但返回类型写成 `<2 x half>` 聚合 → IR 校验直接不过。**

```llvm
declare {<2 x half>, <2 x half>, <2 x half>, <2 x half>}
  @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16(i8 addrspace(3)*)
```

```
llc: error: 'q.ll': input module cannot be verified
```

这三组对照在 `tools/ch10-experiments.sh` 里是可复现的，实测输出：

```
===== ldmatrix.x4.b16: 三组对照(名字/签名 对与不对) =====
OK    llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16
        	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd1];
FAIL  llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.shared.b16
        error: call to unknown intrinsic '...x4.shared.b16' cannot be lowered by the NVPTX backend
FAIL  llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16
        llc: error: input module cannot be verified
```

**这三条错误信息就是你的排查手册**：

- 报 `unknown intrinsic ... cannot be lowered` → **名字**不对（多半是把 PTX 的名字抄进了 IR）；
- 报 `input module cannot be verified` → **签名**不对（参数或返回类型和 `.td` 里的定义不一致）；
- 什么都不报、PTX 里出现 `.extern .func` + 一条 `call` → 这是**老版本 LLVM 的行为**（名字对不上时静默降级成外部函数调用）。新版本改成了报错，但这个坑依然值得记：**IR"能编译"不等于"真的选出了指令"**，永远要去 PTX 里确认那行 `ldmatrix` / `mma` 存在。

### 这套 LLVM 里的 ldmatrix 长什么样

把 `IntrinsicsNVVM.td` 里那条规则抄下来（第 611 行），事实就都有答案了：

```tablegen
class LDMATRIX_NAME<WMMA_REGS Frag, int Trans> {
  defvar name = "llvm.nvvm.ldmatrix.sync.aligned"
                # "." # Frag.geom
                # "." # Frag.frag
                # !if(Trans, ".trans", "")
                # "." # Frag.ptx_elt_type
                ;
  ...
}
```

拼出来就是 `llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16`——**名字里没有 `.shared`**。那"是共享内存版本还是 generic 版本"靠什么区分？靠**指针类型**：`NVVM_LDMATRIX` 的参数写的是 `llvm_anyptr_ty`（任意指针），地址空间在指针类型里（`i8 addrspace(3)*`）。返回类型是 `Frag.regs`，也就是 `<2 x half>` 数量的**打包 i32 元组** `{i32,i32,i32,i32}`，所以想喂给 `mma` 需要先 `bitcast i32 -> <2 x half>`（我们的 demo 就是这么写的）。

而 PTX 那边写 `.shared`，是因为 **PTX 指令名**里要带空间（`.shared.global` / `.shared`），这跟 intrinsic 名是两套命名——**别把 PTX 的名字套到 IR 上**，这就是对照组 B 踩的坑。

**教训**：写 IR 之前，先确认你手上这套 LLVM 的 intrinsic 名字。方法有两个：

```bash
# 方法一: 直接问编译器(把 PTX 里有没有真指令当判据)
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -o - demo.ll | grep -E '^\s+ldmatrix\.sync'

# 方法二: 从二进制里翻 intrinsic 名字表
strings /usr/local/bin/llc | grep 'nvvm.ldmatrix' | sort -u

# 方法三(最可靠): 直接查源码树里的 .td —— 本机的 llc 就是从它构建出来的
grep -n 'LDMATRIX_NAME' -A 8 /root/llvm-project/llvm/include/llvm/IR/IntrinsicsNVVM.td | head -12
```

## 10.6 新指令接入 checklist

把上面所有内容压缩成一张可以贴在显示器上的表：

```
□ 1. 规格化: 语义 / 形状 / 操作数类型与个数 / 寄存器类 / 内存行为 / 是否 convergent
□ 2. intrinsic: IntrinsicsNVVM.td  —— 签名 + IntrConvergent + 名字拼法
□ 3. 指令:     NVPTXInstrInfo.td / NVPTXIntrinsics.td —— AsmString + Pat + 谓词(arch/PTX 版本)
□ 4. 内存:     NVPTXISelLowering.cpp getTgtMemIntrinsic —— 访存类必须登记
□ 5. 边界:     不支持的类型/地址空间要有明确报错, 别静默生成错误指令
□ 6. 前端:     clang builtin(可选) 或让上游直发 intrinsic
□ 7. 测试:     wmma.py 式生成测试 + convergent 专项测试(照抄 mma-no-sink-after-laneid-check.ll)
□ 8. 三段验证: llc 出 PTX → ptxas 出 SASS → 上机跑数值对拍
□ 9. 版本核对: 确认 intrinsic 名字/类型在你支持的所有 LLVM 版本里是同一个
```

## 10.7 全书收尾

十章的链路，用一句话概括：

> **`.cu` 里的每一行 CUDA 语义，都会在 IR 里找到一个基础构件；IR 里的每一个值，都会在机器层找到一组寄存器；机器层最终把一切摊成交错执行的 SASS 指令——而 ptxas 才是那个决定"哪条指令住在哪个寄存器、怎么排"的角色。**

把这条链再走一遍，你应该能自己回答这些问题了：

| 问题 | 章节 |
| --- | --- |
| `__global__` 凭什么变成 kernel？ | 第 3 章（签名上的 `ptx_kernel` calling convention） |
| 为什么我的 `float c[4]` 在 IR 里没了？ | 第 4 章（SSA + phi）、第 5 章（SROA/mem2reg） |
| 为什么循环没展开？ | 第 5 章（trip count 未知时是 convergent 挡的；已知时是代价模型说了算）、第 9 章（ptxas 在 SASS 里替你做了 4x/2x/1x） |
| 为什么 `mma` 挪不动？ | 第 5 章（convergent → MachineSink/LICM/CSE 全部收手） |
| `ldmatrix` 的地址为什么要那样算？ | 第 2.2、8.3 章（fragment 布局 + `.trans` 只转一次） |
| SASS 里 `HMMA.16816.F32` 的操作数是什么？ | 第 9 章（D/A/B/C = 4/4/2/4） |
| LLVM 会做寄存器分配吗？ | 第 7 章（NVPTX 不分配，ptxas 分配） |
| 怎么给新指令做前端？ | 第 10 章（六步法 + 实测） |

如果你想继续往下挖，三个方向最值钱：

1. **多 warp 的分工**：把 block 从 1 warp 扩到 4/8 warp，看 `cp.async` 的流水怎么组织（`ldmatrix` 地址会变复杂，但本章的 fragment 表不变）。
2. **`wgmma` / `tcgen05`**：sm_90 之后的 warpgroup MMA 和 tcgen05 是完全不同的编程模型（从"每线程持 fragment"变成"操作数放 shared/寄存器 + 描述符"），仓库里 `llvm/test/CodeGen/NVPTX/tcgen05-mma.ll` 是很好的起点。
3. **调优闭环**：用 `ncu` 看 HMMA 的占用率，回来对照第 5 章的 unroll 决策和第 7 章的寄存器数量——你会发现"编译器选择"和"硬件表现"之间那条线，现在你能自己画出来了。
