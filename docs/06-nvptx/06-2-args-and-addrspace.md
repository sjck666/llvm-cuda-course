# 06-2 · 参数与地址空间 lowering：generic → global 的证据链

这一讲讲 NVPTX 后端**最关键的一个改写**，也是第 02-3、02-2 讲埋下的伏笔：

> **kernel 的指针参数在 IR 里是 generic（地址空间 0），
> 后端怎么把它"定性"成 global（地址空间 1）？**

为什么这件事重要？因为**地址空间决定了用哪条访存指令**：

```
generic  → 通用的 ld / st（要运行时判断走哪条路径，慢）
global   → ld.global / st.global（直接、快）
shared   → ld.shared / st.shared
```

## 一、整体是三个 pass 的接力

先说结论，再给证据：

| 顺序 | pass | 干什么 |
| --- | --- | --- |
| 1 | `NVPTX Mark Kernel Pointers Global` | 在 kernel 指针参数上**插一对 `addrspacecast`**，声明"它指向 global" |
| 2 | `Lower pointer arguments of CUDA kernels`（`nvptx-lower-args`） | 处理 **byval 参数**，把它落到 `addrspace(101)`（param 空间） |
| 3 | `Infer address spaces`（通用 pass） | 把上面那对 `addrspacecast` **折叠成一条**，并把后续访存都染成 global |

第 1、2 步是 NVPTX 专属的，第 3 步是通用 pass。**这就是"目标插 pass + 通用 pass 干活"
的典型配合。**

## 二、第 1 步：为什么是"插一对 addrspacecast"

看源码（`llvm/lib/Target/NVPTX/NVPTXMarkKernelPtrsGlobal.cpp`）。开头注释说：

```
// For CUDA kernels, pointers loaded from byval parameters are known to be in
// global address space. This pass inserts addrspacecast pairs to make that
// explicit, enabling later address-space inference to propagate the global AS.
// It also handles the pattern where a pointer is loaded as an integer and then
// converted via inttoptr.
```

然后是这个函数：

```cpp
static void markPointerAsAS(Value *Ptr, unsigned AS) {
  ...
  Instruction *PtrInGlobal = new AddrSpaceCastInst(
      Ptr, PointerType::get(Ptr->getContext(), AS), Ptr->getName(), InsertPt);
  Value *PtrInGeneric = new AddrSpaceCastInst(PtrInGlobal, Ptr->getType(),
                                              Ptr->getName(), InsertPt);
  Ptr->replaceAllUsesWith(PtrInGeneric);
  PtrInGlobal->setOperand(0, Ptr);
}
```

**读这段代码的关键是最后两行**：

```
1. 造一个"转成 global"的 cast（PtrInGlobal）
2. 再造一个"转回 generic"的 cast（PtrInGeneric）
3. 把所有使用者接到 PtrInGeneric 上
```

**为什么要转一圈回来？** 因为它**不想改变语义**——它只想在图上留下一条
"这条链来自 global"的痕迹，让后面的地址空间推断能看见。

**这是一个非常漂亮的技巧**：

> **"插一个提示，但马上绕回原样"——语义不变，信息增加。**

## 三、第 2 步：`nvptx-lower-args` 在做什么

注意它处理的是 **byval 参数**（按值传进来的结构体）。
它的注释（`NVPTXLowerArgs.cpp` 开头）解释得很清楚：

```
// Arguments to kernel functions are passed via param space, which imposes
// certain restrictions:
// ...
// Kernel parameters are read-only and accessible only via ld.param
// instruction, directly or via a pointer.
```

**"kernel 参数在 param 空间里，只读，只能通过 ld.param 访问"**——
这就是它要处理的核心约束。

而"指针参数指向的数据"（比如我们的 A/B/D）不在 param 空间，
它们指向 global 内存——这正是第 1 步要标注的东西。

## 四、第 3 步：`Infer address spaces` 把提示"消化"掉

这个通用 pass 会做**前向传播**：

```
如果 %p 是 addrspace(1)，那么所有由 %p 派生的指针（GEP、bitcast…）也是 addrspace(1)
```

于是第 1 步插的那对 cast 会被折叠：**回程的 cast 消失了，指针保持 global**。

## 五、完整的证据链（这是本讲的重点）

我们从三个层次各取一份证据，形成一条链：

### IR 层：喂给 ISel 的那份 IR

注意：**不是** `dumps/02-device-O2.ll`（那是 clang 的输出，还没跑后端 pass）。
后端 pass 跑完之后的 IR，嵌在 MIR 文件的头部（`dumps/26-isel.mir:13`）：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(ptr noalias nofree noundef readonly captures(none) %0, ...) #0 {
  %7 = addrspacecast ptr %0 to ptr addrspace(1)     ; ← Mark Kernel Pointers Global 插的
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
  ...
```

**只剩一条了**——原来是"转过去再转回来"的一对，`Infer address spaces`
把回程那条连它的使用者一起折掉了。

所以后面所有对 A/B/D 的访问都直接在 `addrspace(1)` 上做。

### MIR 层：地址转换变成了机器指令

```bash
grep -n "cvta_to_global" dumps/26-isel.mir | head -6
```

```
%0:b64 = cvta_to_global_64 killed %46
%1:b64 = cvta_to_global_64 killed %47
%2:b64 = cvta_to_global_64 killed %48
%0:b64 = cvta_to_global_64 killed %37
%1:b64 = cvta_to_global_64 killed %38
%2:b64 = cvta_to_global_64 killed %39
```

**一共 6 处**（两个 kernel × 3 个指针参数）。源操作数就是前面 `LD_i64`
从 param 空间读出来的参数值。

**这就是前端那个 generic 参数在后端第一次被"定性"的地方。**

### PTX 层：变成了 `cvta.to.global.u64`

```bash
grep -n "cvta.to.global" dumps/03-clang-O2.ptx | head -6
```

```
	cvta.to.global.u64 	%rd3, %rd8;
	cvta.to.global.u64 	%rd1, %rd6;
	cvta.to.global.u64 	%rd2, %rd7;
```

**一条链，三个层次，名字环环相扣**：

```
IR:   addrspacecast ptr %0 to ptr addrspace(1)
MIR:  %0:b64 = cvta_to_global_64 killed %46
PTX:  cvta.to.global.u64 %rd3, %rd8;
```

## 六、顺带说一个细节：param 空间的 load

参数本身是从哪里读出来的？看 MIR（`dumps/26-isel.mir:482`）：

```
%45:b32 = LD_i32 0, 0, 101, 3, 32, -1, <mcsymbol mma_tc_manual_param_5>, 0, 0, $noreg
```

关键在中间那个 **`101`**——它是 `ADDRESS_SPACE_ENTRY_PARAM`（第 02-3 讲那张表）。
`<mcsymbol mma_tc_manual_param_5>` 就是第 6 个参数（`K`）的符号。

对应的 PTX：

```
	ld.param::entry.b32 	%r10, [mma_tc_manual_param_5];
```

**`.param::entry` 是新版 PTX 的写法**（老版本写 `ld.param.u32`），
意思很直白："从入口函数的参数区读"。

## 七、为什么不一开始就用 global 指针

你可能会想：既然 kernel 的指针参数一定是 global，为什么前端不直接声明成
`ptr addrspace(1)`，省得后面再转换？

因为**前端的视角和设备的视角不一样**：

```
host 侧：cudaMalloc 返回的指针，在 host 看来就是"一个普通指针"
device 侧：CUDA 允许你把它当 generic 用（自己 __cvta_* 转换）
```

而且 `__restrict__`、`__device__` 变量、动态共享内存这些情况混在一起时，
"这个指针到底指向哪个空间"**只能靠分析**（比如从它派生出来的访问方式）。

所以 LLVM 的策略是：**前端保守地用 generic，后端用"标记 + 推断"来定性**。
这和我们第 02-3 讲的 `__cvta_generic_to_shared` 是同一个思路的另一面。

## 八、小结

1. kernel 指针参数的"定性"是**三棒接力**：`NVPTXMarkKernelPtrsGlobal` 插
   `addrspacecast` 对 → `Infer address spaces` 折叠传播 → 后续访存全部变成 global。
2. 完整证据链：**IR 的 `addrspacecast` → MIR 的 `cvta_to_global_64` →
   PTX 的 `cvta.to.global.u64`**（各 6 处）。
3. 参数本身住在 **param 空间（101）**，靠 `LD_i32`/`LD_i64` 读出，
   对应 PTX 的 `ld.param::entry.b32`。

## 九、动手题

1. 亲手把这条链数一遍：

   ```bash
   grep -c "cvta_to_global_64" dumps/26-isel.mir      # MIR 层
   grep -c "cvta.to.global"    dumps/03-clang-O2.ptx  # PTX 层
   grep -c "addrspacecast ptr %0 to ptr addrspace(1)" dumps/26-isel.mir
   ```

   三个数字之间是什么关系？（提示：想想 6 处 cvta 是怎么来的。）
2. 找一个反例：在 `dumps/02-device-O2.ll`（clang 的输出）里搜 `addrspacecast`，
   看看有几处、分别在哪。**为什么这里没有参数那三处？**

下一讲我们看指令选择之后的结果：那些 `LD_GLOBAL_NC_i32`、`SHL32_ri`、
`SETP_i32ri` 到底在说什么。
