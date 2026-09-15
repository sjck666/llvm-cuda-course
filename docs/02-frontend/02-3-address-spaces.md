# 02-3 · 地址空间：CUDA 的内存模型怎么落到 IR

这一讲我想讲 CUDA 前端对 LLVM 做的**最有实质影响**的一件事：把 CUDA 的那套内存模型
翻译成 LLVM IR 的"地址空间"。

为什么说"最有实质影响"？因为地址空间不只是个编号。**它决定了后端用哪条访存指令、
走哪条缓存路径**。我们这门课的案例里，`__shared__` 和 global memory 的差异，
最后在 SASS 里变成了 `LDSM` / `LDGSTS` 和 `LDG` 的区别——根就在这一讲。

## 一、先看结论：CUDA 的几种内存，各有各的编号

在 CUDA C++ 里你知道有这些内存：

```cpp
__shared__   __half As[16][16];    // 共享内存
__constant__ float table[256];     // 常量内存
__device__   int counter;          // 全局内存里的设备变量
float x;                           // 线程私有的局部变量（可能进寄存器，也可能溢出到 local）
```

在 LLVM IR 里，这些差异用**指针的地址空间编号**来表达。NVPTX 用的编号是：

| 编号 | 名字 | 什么时候出现 |
| --- | --- | --- |
| `0` | generic | **函数参数**、`malloc` 出来的指针、还没被推断出来的指针 |
| `1` | global | `__device__` / `__constant__` 全局变量、`cudaMalloc` 传进来的指针（推断之后） |
| `3` | shared | `__shared__` 变量、动态共享内存 |
| `4` | constant | `__constant__` |
| `5` | local | 寄存器溢出、局部数组落到的"本地内存" |
| `101` | param | kernel 参数区（这是 NVPTX 私有的编号，第 6 部分会看到） |

这些数字可以在 LLVM 源码里查到（`llvm/include/llvm/Support/NVPTXAddrSpace.h`）：

```cpp
enum AddressSpace : unsigned {
  ADDRESS_SPACE_GENERIC = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_SHARED = 3,
  ADDRESS_SPACE_CONST = 4,
  ADDRESS_SPACE_LOCAL = 5,
  ADDRESS_SPACE_TENSOR = 6,          // sm_90+ 的 tensor memory
  ADDRESS_SPACE_SHARED_CLUSTER = 7,  // sm_90+ 的 cluster 共享内存
  ADDRESS_SPACE_ENTRY_PARAM = 101,
};
```

（注意 2 被跳过了——那是历史遗留。`6` 和 `7` 是新加的，对应 sm_90 之后的特性，
我们 sm_89 用不到，但它们已经躺在 datalayout 里了。）

## 二、`__shared__` 数组：从"函数里的静态变量"升成"模块级全局变量"

我们的 v2 里有这么两行：

```cpp
__shared__ __align__(16) __half As[16][16];
__shared__ __align__(16) __half Bs[16][8];
```

它们在 IR 里**不在函数里，而是跑到模块级**去了。打开 `dumps/01-device-O0.ll` 第 15 行：

```llvm
@_ZZ15mma_tc_ldmatrixE2As = internal addrspace(3) global [16 x [16 x %struct.__half]] undef, align 16
@_ZZ15mma_tc_ldmatrixE2Bs = internal addrspace(3) global [16 x [8  x %struct.__half]] undef, align 16
```

逐段读第一行：

| 部分 | 含义 |
| --- | --- |
| `@_ZZ15mma_tc_ldmatrixE2As` | 名字。这是 Itanium ABI 里"函数内静态变量"的命名法（下面细讲） |
| `internal` | 链接属性：模块私有，不导出 |
| `addrspace(3)` | **住在共享内存空间** |
| `global` | 这是个全局对象（不是常数、不是别名） |
| `[16 x [16 x %struct.__half]]` | 类型：16×16 个 `__half` |
| `undef` | 初值未定义——共享内存不需要初始化，`undef` 是最省事的写法 |
| `align 16` | 16 字节对齐（就是我们写的 `__align__(16)`） |

### 那个又长又丑的名字是怎么来的

`_ZZ15mma_tc_ldmatrixE2As` 拆开看：

```
_ZZ                     ← "这是函数内的静态变量"（Itanium ABI 的约定）
   15                   ← 函数名的长度：`mma_tc_ldmatrix` 有 15 个字符
     mma_tc_ldmatrix    ← 函数名
                    E   ← 函数名结束
                     2  ← "变量名长度是 2 位数字"（这是 ABI 的细节）
                      As← 变量名
```

**编译器把"函数里的静态变量"当成了"每个函数一份的全局对象"来命名**，
尽管它最终就住在模块级。

### 为什么必须搬到模块级

因为**共享内存是"整个 block 共享"的**，不是某个线程私有的。
如果它留在函数里（比如作为栈上对象），语义就错了。
把它提成模块级的 global 之后：

- 所有线程访问的是同一块内存；
- 后端知道它属于 `addrspace(3)`，可以据此生成 `ld.shared` / `st.shared`；
- `.shared` 声明里的字节数（`.b8 As[512]`）就是从这里算出来的。

## 三、访问它的时候：每次都要"跨一次地址空间"

看一个实际的访问点（`dumps/01-device-O0.ll:472`）：

```llvm
%80 = getelementptr inbounds [16 x [16 x %struct.__half]],
      ptr addrspacecast (ptr addrspace(3) @_ZZ15mma_tc_ldmatrixE2As to ptr),
      i64 0, i64 %79
```

注意中间那个 `addrspacecast`：**它把 `addrspace(3)` 的指针转成了 generic 指针**，
然后才做 `getelementptr`。

为什么不直接在 `addrspace(3)` 上算？因为源码里我们写的是"取 `&As[r][c]`"，
而在 C/C++ 的语义里，**取地址得到的是一个普通指针**。前端老老实实做了转换。

这个过程会重复出现很多次（每个访问点一次）。而它们在优化之后会被处理掉——
第 6 部分你会看到后端是怎么把这些 `addrspacecast` 收拾干净的。

## 四、函数参数为什么是 generic（回顾 + 加深）

上一讲我们说：kernel 的指针参数是 `ptr`（地址空间 0）。现在你知道为什么这是**必须**的了：

```cpp
mma_tc_manual<<<grid, block>>>(dA, dB, dD, M, N, K);
```

`dA` 是 host 侧 `cudaMalloc` 出来的。它的"身份"要到设备端才清楚，
而且 CUDA 允许你把它当 generic 指针用（比如你自己 `__cvta_generic_to_shared`）。

所以前端统一当 generic 处理，**把"这是 global 内存"这件事留给后端去推断**。
后端的做法是：

```
1. 标记 kernel 的指针参数"一定是 global"（插 addrspacecast 对）
2. 用地址空间推断 pass 把这条信息传播下去
3. 于是所有从参数来的访存都变成 addrspace(1) 的访存
4. 最终生成 ld.global / st.global（而不是通用的 ld / st）
```

这条链的证据在第 6 部分会一环一环给你看：

```
IR:  addrspacecast ptr %0 to ptr addrspace(1)
MIR: cvta_to_global_64
PTX: cvta.to.global.u64
```

## 五、反过来：从 generic 转进 shared

还有一个常见操作是"把 generic 指针转成 shared 偏移"。我们的 v2 里用了
`__cvta_generic_to_shared`：

```cpp
uint32_t smem = (uint32_t)__cvta_generic_to_shared(&As[r][c]);
```

在 IR 里它**不是一个 intrinsic，而是一个真实的函数**（`dumps/01-device-O0.ll:757`）：

```llvm
define linkonce_odr dso_local i64 @__nv_cvta_generic_to_shared_impl(ptr noundef %0) #5 comdat {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = addrspacecast ptr %3 to ptr addrspace(3)     ; generic -> shared
  %5 = ptrtoint ptr addrspace(3) %4 to i64          ; 变成整数
  ret i64 %5
}
```

两个细节：

1. **`linkonce_odr` + `comdat`**：意思是"每个用到它的编译单元各带一份定义，链接器去重"。
   所以你在 `.ll` 里既能看到定义，也能看到调用点。
2. 它的核心就是 **`addrspacecast` + `ptrtoint`**——把指针先转到 shared 空间，
   再变成整数。

这个函数在被优化之后会**完全消失**，变成纯粹的地址算术 + 一次截断：

```
mov.b64   %rd9, _ZZ15mma_tc_ldmatrixE2As;   ; 符号基址
add.s64   %rd10, %rd9, %rd8;                ; + 偏移
mul.wide.u32 %rd11, %r11, 2;
add.s64   %rd12, %rd10, %rd11;
cvt.u32.u64 %r19, %rd12;                    ; 64 位地址 → 32 位 shared 偏移
cp.async.cg.shared.global [%r19], [%rd43], 16;
```

**最后那一步 `cvt.u32.u64` 就是 `__cvta_generic_to_shared` 的真身。**
它不需要任何专用指令——因为 sm_89 上共享内存本来就是"32 位偏移"寻址，
只要把 64 位地址截断就行了。（这一点在 datalayout 里也有声明，
第 3 部分读 IR 头部时会看到那个 `p6:32:32` 的段落。）

## 六、小结

1. CUDA 的内存模型在 IR 里靠**地址空间编号**表达：`0` generic、`1` global、
   `3` shared、`4` constant、`5` local，加上 NVPTX 私有的 `101` param。
2. `__shared__` 数组会被**提升成模块级的 `addrspace(3)` global**，
   名字是 ABI 生成的（`_ZZ15<函数名>E2<变量名>`），初值 `undef`，对齐来自 `__align__`。
3. 函数参数是 generic（`ptr`），"这是 global"的定性发生在**后端**；
   而 `__cvta_generic_to_shared` 最终被优化成一次 `cvt.u32.u64` 截断。

## 七、动手题

1. 数一数 `addrspacecast` 的次数：

   ```bash
   grep -c addrspacecast dumps/01-device-O0.ll    # 5
   grep -c addrspacecast dumps/02-device-O2.ll    # 8
   ```

   **优化之后居然变多了。为什么？**（提示：`__cvta_generic_to_shared_impl`
   在 O0 里是一个独立函数，只有一处 `addrspacecast`；到了 O2 它被内联进调用者，
   于是那些 `addrspacecast` 出现在函数体里了。**"次数变多"不等于"代码变差"**——
   要看它们最终有没有变成真指令。）
2. 对照 PTX 里的共享内存声明：

   ```
   .shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2As[512];   ← 16×16×2 字节
   .shared .align 16 .b8 _ZZ15mma_tc_ldmatrixE2Bs[256];   ← 16×8×2 字节
   ```

   自己算一遍这两个数字，确认它们和 IR 里的类型对得上。
2. 试着把 `__shared__` 数组的 `__align__(16)` 去掉，重新生成 PTX，
   看 `.shared` 声明里的对齐变成了什么。
   （提示：`cp.async` 的 16 字节版本要求目的地址 16 字节对齐——
   对齐掉了会怎样？）

下一讲我们看 `threadIdx.x` 这种"读硬件寄存器"的写法，在 IR 里变成了什么。
