# 02-7 · `__half` 的胶水代码

这一讲讲两个"看起来是细枝末节、但会影响性能"的东西：
**`__half` 在 IR 里是什么**，以及**用整数视角看 half 能省掉什么**。

## 一、`__half` 不是一个类型，是一个结构体

你写 `__half`，以为它是"半精度浮点数"。但在 IR 里它是这个：

```llvm
%struct.__half  = type { i16 }
%struct.__half2 = type { %struct.__half, %struct.__half }
```

**`__half` 就是一个 16 位的结构体**，`__half2` 就是两个这样的东西。

为什么这么设计？因为 `__half` 在 CUDA C++ 里主要是**存储类型**——
你要把它塞进内存、塞进寄存器、打包传输。真正参与计算的"半精度浮点"，
LLVM 里有专门的 `half` 类型（你在 `dumps/04-nvcc-O3.ptx` 那条路径里会看到
`mov.f16` 之类），但我们的代码里从头到尾都是"把 half 当位模式搬来搬去"。

## 二、由此带来的"胶水代码"

因为 `__half` 是结构体，**两个 half 拼成一个 32 位寄存器这件事就没有硬件指令直接对应**，
于是 CUDA 的头文件用一段内联汇编实现了它。

源码：`__halves2half2(lo, hi)`。它的 IR 形态（`dumps/01-device-O0.ll:742`）：

```llvm
%7 = call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %5, i16 %6) #8, !srcloc !16
```

逐段读：

| 部分 | 含义 |
| --- | --- |
| `i32` 返回 | 拼出来的 32 位值 |
| `"{ mov.b32 $0, {$1,$2}; }"` | PTX 里的写法：把两个 16 位寄存器装进一个 32 位寄存器 |
| `"=r,h,h"` | 输出 `=r`（32 位），两个输入 `h`（**16 位整型寄存器**） |
| `#8` | 属性组。在 O0 这份文件里 `#8 = { convergent nounwind memory(none) }` |

注意最后那个 `h` 约束——它表示 **16 位整型寄存器**。这是"约束字符"里比较少见的几个之一
（常见的是 `r` 32 位、`l` 64 位、`f` 浮点）。你会见到它的地方基本都和 half/bf16 有关。

到了机器层（`dumps/26-isel.mir`）：

```
INLINEASM &"{  mov.b32 $0, {$1,$2};}\0A", isconvergent attdialect,
          regdef:B32, def %72, reguse:B16, %73, reguse:B16, %74, !17
```

**注意它带着 `isconvergent`**——虽然是纯数据打包。
这就是上一讲那条"全标 convergent"规则的直接后果：**连一条 `mov.b32` 都不能幸免**。

## 三、于是有了那个"整数视角"的技巧

既然 `__half` 是结构体、打包要走汇编，那我们干脆**别用 `__half` 视角**。

回头看你很熟的那段 A 装载代码：

```cpp
__device__ __forceinline__ uint32_t pack_half2(const __half *p) {
  return *reinterpret_cast<const uint32_t *>(p);
}
```

我们把两个连续的 half **当成一个 `uint32_t` 直接读出来**。因为 A fragment 的
同一寄存器里就是"同一行相邻的两个 K"（第 01-2 讲），它们在内存里本来就是连续的。

这一下省掉了什么？

```
按 __half 读：ld.global.nc.b16 ×2  +  { mov.b32 } 打包   → 3 条指令
按 uint32_t 读：ld.global.nc.b32 ×1                        → 1 条指令
```

在 IR 层面，这个差别就是"两条 `load i16` + 一条 asm 调用" 对比 "一条 `load i32`"。
在 `dumps/02-device-O2.ll` 的循环里，你能看到 A 的四条 `load i32`：

```llvm
%68 = load i32, ptr %67, align 4, !tbaa !13     ; a[0]
%70 = load i32, ptr %69, align 4, !tbaa !13     ; a[1]
%72 = load i32, ptr %71, align 4, !tbaa !13     ; a[2]
%74 = load i32, ptr %73, align 4, !tbaa !13     ; a[3]
```

而 **B 就没这么幸运**：B 的两个 half 在内存里隔了 N 个元素（第 01-3 讲），
所以只能老老实实两条 `load i16` + 一次打包：

```llvm
%83 = load i16, ptr %78, align 2, !tbaa !14     ; B[k][n]
%85 = load i16, ptr %84, align 2, !tbaa !14     ; B[k+1][n]
%86 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %83, i16 %85) #3, !srcloc !16
```

**这一段对比是整门课里"fragment 布局如何影响指令数"最直观的例子。**
你在第 8 部分读 SASS 时，会看到这四条 `load i32` 变成四条 `LDG.E.CONSTANT`，
而那两条 `load i16` + 打包变成两条 `LDG.E.U16.CONSTANT` + 一条 `PRMT`。

## 四、`mov.b32` 到 SASS 里是什么

PTX 里的 `mov.b32 %r20, {%rs1,%rs2}` 在 SASS 里没有直接对应——
硬件没有"把两个 16 位拼进一个 32 位"的 mov。ptxas 的做法是用**字节置换指令**：

```
PRMT R24, R27, 0x5410, R29 ;
```

`0x5410` 这个立即数说的是"取第 4、5 个字节放低半、第 0、1 个字节放高半"之类
的选择模式。第 8 部分会把这条指令拆开讲。

**一句话：PTX 是"虚拟 ISA"，它允许存在语义清楚但硬件没有直接对应的指令，
ptxas 的职责之一就是把这些"理想指令"落地成真实指令。**

## 五、顺带说：`__cvta_generic_to_shared` 也是"胶水"，但它会被优化掉

第 02-3 讲我们提过：`__cvta_generic_to_shared` 在 IR 里是一个**真实的函数**
（`linkonce_odr` + `comdat`），核心就两条 `addrspacecast` + `ptrtoint`。

到了 `-O2` 之后，这个函数被内联、被折叠，最后在 PTX 里只剩一次截断：

```
cvt.u32.u64 	%r19, %rd12;      ; 64 位地址 → 32 位 shared 偏移
```

**它和 `__halves2half2` 的区别很有意思**：

| 胶水 | 原型 | 最终变成 |
| --- | --- | --- |
| `__halves2half2` | 内联汇编 `mov.b32 {a,b}` | `PRMT`（真实指令，去不掉） |
| `__cvta_generic_to_shared` | 普通函数（地址转换） | 一次 `cvt` 截断（几乎免费） |

**为什么前者去不掉？** 因为它做的事硬件真的没有"一条指令"能干（两个 16 位拼装
需要字节置换），而后者（指针转换）在 sm_89 上本来就只是"截断"。

## 六、小结

1. `__half` 在 IR 里是 `%struct.__half = type { i16 }`，
   **它是个结构体，不是浮点类型**；`__half2` 是两个这样的东西。
2. 由此产生的 `__halves2half2` 是一条内联汇编 `mov.b32 {a,b}`，
   约束里用 `h`（16 位整数寄存器）；它**也带 `convergent`**。
3. 我们的 A fragment 用 `uint32_t` 视角直接读，**一条 `load i32` 代替
   "两条 `load i16` + 一次打包"**；B 因为布局原因做不到这一点。
   这是"用整数视角看 half"这个技巧的全部意义。

## 七、动手题

1. 把 A 的 `pack_half2` 改成"老老实实两个 `__half` 相加再打包"的写法，重新生成
   `dumps/03-clang-O2.ptx`，看看 A 那部分指令变成了几条。
   （做完记得改回来。）
2. 数一数 `dumps/02-device-O2.ll` 里 `mov.b32` 那条 asm 出现了几次，
   和 `dumps/03-clang-O2.ptx` 里的 `mov.b32 %rXX, {%rsX,%rsY}` 对一下。

下一讲我们把镜头转向 host 侧：`<<<grid, block>>>` 到底变成了什么。
