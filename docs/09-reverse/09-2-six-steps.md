# 09-2 · 六步接入法：逐文件走一遍

这一讲走路线 C。**每一步都给出真实源码位置**（本机的 llc/clang 就是从
`/root/llvm-project` 这棵树构建的，所以行号和你手上的源码能对上）。

## Step 1 · 定义 intrinsic：`IntrinsicsNVVM.td`

NVVM 的 mma/ldmatrix intrinsic **不是一个个手写的**，而是 TableGen 批量生成的
（第 05-3、05-4 讲的内容）。三个关键类：

```
class WMMA_REGS<string Geom, string Frag, string PtxEltType, bit IsSparse = false>   // 第 208 行
class NVVM_MMA<WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D>                  // 第 2956 行
class NVVM_LDMATRIX<WMMA_REGS Frag, int Transposed>                                  // 第 3108 行
```

`WMMA_REGS` 干的事就是**把"规格化六件事"编码成 TableGen 数据**：

```tablegen
// 摘录：m16n8k16 的四个 fragment 各占几个寄存器、什么类型
!eq(gft,"m16n8k16:a:f16") : !listsplat(llvm_v2f16_ty, 4),   // A = 4 个 <2 x f16>
!eq(gft,"m16n8k16:b:f16") : !listsplat(llvm_v2f16_ty, 2),   // B = 2 个
!eq(gft,"m16n8k16:c:f32") : !listsplat(llvm_float_ty, 4),   // C = 4 个
!eq(gft,"m16n8k16:d:f32") : !listsplat(llvm_float_ty, 4),   // D = 4 个
```

而 `NVVM_LDMATRIX` 顺手把 convergent 也标上了（这就是"第 6 件事"的落地）：

```tablegen
class NVVM_LDMATRIX<WMMA_REGS Frag, int Transposed>
  : Intrinsic<Frag.regs, [llvm_anyptr_ty],
              [IntrReadMem, IntrArgMemOnly, IntrNoCallback, IntrConvergent,
               ReadOnly<ArgIndex<0>>, NoCapture<ArgIndex<0>>],
              LDMATRIX_NAME<Frag, Transposed>.intr_name>;
```

**注意 `IntrConvergent`**——Step 1 就要写对，否则后面测试再全也没用。

### intrinsic 名字的拼法也有脚手架

`LDMATRIX_NAME`（第 611 行）：

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

拼出来是 **`llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16`——名字里没有 `.shared`**。

**这一点非常重要**：第 09-3 节会看到，很多人照着 PTX 的名字往 IR 里写
`...x4.shared.b16`，结果 LLVM 直接不认。原因就是——

> **IR 的名字和 PTX 的名字是两套命名，空间信息在 IR 里是由指针类型承载的。**

## Step 2 · 定义机器指令 + pattern：`NVPTXIntrinsics.td`

同样是一批 TableGen 类：

```
class MMA<WMMA_REGINFO FragA, ...>        // 第 5974 行
class LDMATRIX<WMMA_REGINFO Frag, bit Transposed, NVPTXAddressSpace Space>   // 第 6220 行
```

看 `LDMATRIX` 的实现（第 6220 行）：

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
                  # Frag.geom
                  # "." # Frag.frag
                  # !if(Transposed, ".trans", "")
                  # Space.Suffix
                  # "." # Frag.ptx_elt_type
                  # " " # Frag.regstring # ", [$src];";
}
```

**这段代码把第 09-3 节那个坑解释得明明白白：**

```
intrinsic 的名字  →  由 LDMATRIX_NAME 拼（不带空间）
空间信息          →  放在 NVPTXAddressSpace Space 参数 + IntrinsicPattern（按指针地址空间匹配）里
PTX 里的 .shared   →  来自 Space.Suffix
```

**也就是说：`.shared` 只出现在 PTX 那一侧，IR 里写它当然是错的。**

`MMA` 类同理，把 `mma.sync.aligned.<geom>.<alayout>.<blayout>[.kind][.satfinite]<TypeList>`
拼出来。

## Step 3 · 登记内存行为：`NVPTXISelLowering.cpp`

这一步很多人会漏。`ldmatrix` 是"从共享内存读"，如果不告诉 DAG 这件事，
调度器和别名分析就是瞎的。看 `getTgtMemIntrinsic`（第 4299 行）里的那一长串 case：

```cpp
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_b16:          // 第 4418 行
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8:
  ...
```

这一段的作用是：告诉 SelectionDAG"这个 intrinsic 会读 addrspace(3) 的内存，
读多少字节"。

**新加一条访存类 intrinsic，这一步不能省**——否则别名分析会把它的读写顺序搞错，
你会在某些边界情况下看到错误结果。

## Step 4 · 前端 builtin（可选）：`BuiltinsNVPTX.td`

如果你想让 C/C++ 程序员也能用这条指令，还需要一个 builtin。
clang 的 NVPTX builtin 表在这里：

```
clang/include/clang/Basic/BuiltinsNVPTX.td      （这棵树里只剩 .td 这一份了）
```

里面已经有旧一代的 `__hmma_*` 系列（第 1177 行）：

```tablegen
def __hmma_m16n16k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX60>;
```

clang 把 builtin 名字映射到 intrinsic 的机制在第 02-6 讲提过
（`CGBuiltin.cpp` 的 `getIntrinsicForClangBuiltin`，靠 `ClangBuiltin<"...">` 关联名字）。

**但对 `mma.sync`，上游一直没做这个 builtin**——所以现实里 CUDA C++ 用户
用的是 inline asm。你要自己做的话，路径就是：

```
在 builtin 表里声明 → 在 IntrinsicsNVVM.td 里用 ClangBuiltin<"__nvvm_mma_..."> 关联
                  → 打补丁重编 clang
```

## Step 5 · 测试

LLVM 对这类指令的测试是**生成式**的，这招很值得学：
`llvm/test/CodeGen/NVPTX/wmma.py` 用 Python 遍历所有组合，生成 IR 并配上
FileCheck 断言。mma 那部分是 `gen_mma_tests()`（第 1109 行起）：

```python
mma_intrinsic_template = "llvm.nvvm.mma${b1op}.${geom}.${alayout}.${blayout}${kind}${satf}.${intrinsic_signature}"
mma_instruction_template = "mma.sync${aligned}.${geom}.${alayout}.${blayout}${kind}${satf}.${ptx_signature}${b1op}"
```

**"模板 + 遍历参数组合"** 是这类指令测试的标准做法：
一个形状 × 各种布局 × 各种类型，全都生成一遍。

第 09-4 节会展开讲测试。

## Step 6 · 三段验证

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 -o - demo.ll   # 看 PTX
ptxas -arch=sm_89 -v demo.ptx -o demo.cubin                                # 看 SASS 与资源
cuobjdump -sass demo.cubin | grep HMMA                                     # 确认落到硬件指令
```

## 七、把六步压缩成一张 checklist

```
□ 1. 规格化: 语义 / 形状 / 操作数类型与个数 / 寄存器类 / 内存行为 / 是否 convergent
□ 2. intrinsic: IntrinsicsNVVM.td —— 签名 + IntrConvergent + 名字拼法
□ 3. 指令:     NVPTXInstrInfo.td / NVPTXIntrinsics.td —— AsmString + Pat + 谓词
□ 4. 内存:     NVPTXISelLowering.cpp getTgtMemIntrinsic —— 访存类必须登记
□ 5. 边界:     不支持的类型/地址空间要有明确报错, 别静默生成错误指令
□ 6. 前端:     clang builtin（可选）或让上游直发 intrinsic
□ 7. 测试:     生成式测试 + convergent 专项测试
□ 8. 三段验证: llc 出 PTX → ptxas 出 SASS → 上机跑数值对拍
□ 9. 版本核对: 确认 intrinsic 名字/类型在你支持的所有 LLVM 版本里一致
```

## 八、小结

1. 六步法：**intrinsic 定义 → 指令 + pattern → 内存行为登记 → 前端 builtin →
   测试 → 三段验证**；每一步都有真实源码位置。
2. 最关键的三个点：**intrinsic 的 `IntrConvergent`**、
   **pattern 里的地址空间匹配**、**访存 intrinsic 的内存行为登记**。
3. **IR 的 intrinsic 名和 PTX 指令名是两套命名**——这是下一讲要实测的坑。

## 九、动手题

1. 找到 `NVVM_MMA` 类（第 2956 行）的完整定义，说说它是怎么把
   "四个 fragment"组织成一个 intrinsic 签名的。
2. 在 `NVPTXISelLowering.cpp` 的 `getTgtMemIntrinsic` 里找出所有和 ldmatrix
   相关的 case，数一数有几条。

下一讲我们做三组实测对照，把"名字和签名"这个坑彻底钉死。
