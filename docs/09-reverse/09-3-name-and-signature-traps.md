# 09-3 · 名字与签名陷阱（三组实测对照）

这一讲做一件很"硬"的事：**用三组对照实验，把"名字和签名"这个坑彻底钉死。**

先说结论，免得你踩坑：

> **写 IR 调 intrinsic 时，intrinsic 的"名字"和"签名"都必须和这套 LLVM 的
> 定义完全一致。名字错了会报一种错，签名错了会报另一种错。**

## 一、先看一个能跑通的例子

我们准备了一份纯 IR 的 demo（`code/nvvm_intrinsic_demo.ll`），
里面**没有一行内联汇编**，全靠 intrinsic：

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
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 \
    -o - code/nvvm_intrinsic_demo.ll
```

输出（`dumps/29-nvvm-intrinsic-demo.ptx`）：

```
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

而且 **`ldmatrix` 输出的 `{%r1..%r4}` 直接成了 `mma` 的 A 操作数**——
中间那几条 `bitcast`（i32 → `<2 x half>`）在 ISel 里是免费的。

**注意两件事**：

1. IR 里的名字 `...x4.b16` **没有 `.shared`**，而 PTX 里**有**。
2. IR 里的返回类型是 `{i32,i32,i32,i32}`（打包的 32 位），**不是** `<2 x half>` 聚合。

## 二、三组对照实验

这两个"注意"就是坑所在。我们用脚本把它们跑成对照
（`tools/ch10-experiments.sh`）：

```bash
bash tools/ch10-experiments.sh
```

实测输出：

```
===== ldmatrix.x4.b16: 三组对照(名字/签名 对与不对) =====
OK    llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16
        	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd1];
FAIL  llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.shared.b16
        error: call to unknown intrinsic '...x4.shared.b16' cannot be lowered by the NVPTX backend
FAIL  llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16
        llc: error: input module cannot be verified
```

三种情况，两种不同的报错：

| 情况 | 结果 | 报错 |
| --- | --- | --- |
| **A. 名字对 + 签名对** | 出指令 | — |
| **B. 名字多写了 `.shared`** | 失败 | `call to unknown intrinsic ... cannot be lowered by the NVPTX backend` |
| **C. 名字对但返回类型写成 `<2 x half>` 聚合** | 失败 | `llc: error: input module cannot be verified` |

### 报错的意思

**B 的报错**：LLVM 找不到这个名字的 intrinsic，于是它把这条调用当成
**普通的外部函数调用**——但 NVPTX 后端没法lower 一个"未知的外部函数"，
所以报 `cannot be lowered`。

**C 的报错**：名字是对的，所以 LLVM 认为这是那个 intrinsic；
但 **intrinsic 的签名是"重载"的**（在 `.td` 里定义死了），
你写的返回类型和定义不一致 → **IR 校验器直接拒绝**。

**这两条报错合起来就是你的排查手册：**

```
报 unknown intrinsic / cannot be lowered   →  名字错了
报 input module cannot be verified         →  签名错了
```

## 三、为什么名字里没有 `.shared`

回到 `IntrinsicsNVVM.td` 里那条规则（第 611 行）：

```tablegen
class LDMATRIX_NAME<WMMA_REGS Frag, int Trans> {
  defvar name = "llvm.nvvm.ldmatrix.sync.aligned"
                # "." # Frag.geom
                # "." # Frag.frag
                # !if(Trans, ".trans", "")
                # "." # Frag.ptx_elt_type
                ;
```

**名字里只有：家族 + 几何 + fragment + 可选 `.trans` + 元素类型。没有空间。**

那"是共享内存还是 generic"靠什么区分？靠**指针类型**：

```tablegen
class NVVM_LDMATRIX<WMMA_REGS Frag, int Transposed>
  : Intrinsic<Frag.regs, [llvm_anyptr_ty], ...>    // ← 任意指针，地址空间在指针类型里
```

而 PTX 那一侧的空间后缀，来自指令定义的 `Space.Suffix`（第 09-2 节 Step 2）。

**所以：`.shared` 只属于 PTX 的名字，不属于 IR 的名字。**

### 一个能亲手验证的推论

既然空间由**指针类型**承载，那么同一条 intrinsic、换一种指针类型，
应该生成**不同的 PTX 指令**。我们试了一下：

```bash
for T in "i8 addrspace(3)*" "ptr"; do
  # 生成一个只调用 ldmatrix 的 .ll，指针类型用 $T，然后 llc
  ...
done
```

实测结果：

```
i8 addrspace(3)*  →  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%rd1];
ptr               →  ldmatrix.sync.aligned.m8n8.x4.b16        {%r1, %r2, %r3, %r4}, [%rd1];
```

**同一条 intrinsic，指针在共享内存空间就出 `.shared` 版本，在 generic 空间就不带后缀。**

（`.td` 里那两个变体就是这么定义的：`def AddrSpaceGeneric : ... suffix = ""`、
`def AddrSpaceShared : ... suffix = ".shared"`。）

所以"IR 里不要写 `.shared`"不是风格问题——**写了就找不到对应的 intrinsic 了**。

## 四、一个历史教训：老版本是"静默降级"

第 09-3 节的 B 情况在新版本会**报错**，这其实是好事。因为**老版本 LLVM 不报错**：

```
.extern .func (.param .align 4 .b8 func_retval0[16]) llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

它会**把这条调用当成外部函数**，PTX 里出现一条 `.extern .func` 声明
和一条 `call`——**你的 IR 能编译通过，但那条 `ldmatrix` 指令根本没生成**。

这是一个非常隐蔽的坑：程序可能还能跑（因为函数调用会在运行时失败或者被驱动忽略），
也可能诡异地对——但**性能全没了**。

所以那条教训永远有效：

> **判断"真的选出了指令"的标准只有一个：去 PTX 里找那一行指令本身。**

## 五、怎么确认你手上这套 LLVM 的 intrinsic 名字

三个方法，从快到慢：

**方法一：直接问编译器**（把"PTX 里有没有真指令"当判据）：

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -o - demo.ll | grep -E '^\s+ldmatrix\.sync'
```

**方法二：从二进制里翻字符串**：

```bash
strings /usr/local/bin/llc | grep 'nvvm.ldmatrix' | sort -u
```

**方法三（最可靠）：查源码树里的 `.td`** ——本机的 llc 就是从它构建的：

```bash
grep -n 'LDMATRIX_NAME' -A 8 /root/llvm-project/llvm/include/llvm/IR/IntrinsicsNVVM.td | head -12
```

**方法三最可靠**，因为名字的拼法是 TableGen 规则算出来的，
而那个规则就在源码里；而且既然工具链是从这棵树构建的，两者必然一致。

## 六、小结

1. intrinsic 的**名字和签名都必须与这套 LLVM 的定义一致**：
   名字错了报 `unknown intrinsic / cannot be lowered`，签名错了报
   `input module cannot be verified`。
2. **IR 的名字里没有空间后缀**（`.shared` 只属于 PTX）；空间由**指针类型**承载，
   PTX 的后缀来自指令定义的 `Space.Suffix`。
3. 老版本 LLVM 对"名字对不上"是**静默降级成外部函数调用**——
   所以**永远要去 PTX 里确认指令真的生成了**。

## 七、动手题

1. 自己跑一遍三组对照（`bash tools/ch10-experiments.sh`），
   然后把 demo 里的 `i8 addrspace(3)*` 改成 `ptr`，看看 PTX 里的指令有什么变化。
   想想为什么它**不报错**，但生成的指令不一样。
2. 用方法二和方法三各查一次 `mma` 的 intrinsic 名字，看看结果是否一致。

下一讲我们讲最后一块：**测试怎么写、三段验证怎么做**。
