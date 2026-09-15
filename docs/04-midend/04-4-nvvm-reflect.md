# 04-4 · NVVMReflect：CUDA 特有的那个 pass

这一讲讲一个"只有 CUDA 世界才有"的东西：`__nvvm_reflect`。

它很小，但很能说明问题：**为什么一个目标专属的 pass 必须塞在流水线的最前面**，
以及**"编译期常量"是怎么决定你走哪条代码路径的**。

## 一、`__nvvm_reflect` 是什么

CUDA 的工具链提供了一组"编译期查询"，写法像函数调用：

```c
if (__nvvm_reflect("__CUDA_FTZ")) { /* FTZ 版本 */ }
else                              { /* 非 FTZ 版本 */ }

int arch = __nvvm_reflect("__CUDA_ARCH");   // sm_89 → 890
```

**注意它不是运行时函数。** 它的返回值在编译期就确定了：

- `__CUDA_FTZ`：当前编译是否启用了 flush-to-zero；
- `__CUDA_ARCH`：目标架构编号（sm_89 → `89 * 10 = 890`）。

libdevice（第 02-1 讲那份 NVIDIA 提供的数学库 bitcode）里到处在用这个东西：
同一个数学函数，它会写好"FTZ 版"和"非 FTZ 版"两条路径，
然后用 `__nvvm_reflect` 在编译期选一条。

而 **NVVMReflectPass 的活就是：把这些调用折成常数，然后把死掉的支路删掉。**

## 二、自己造一个最小的例子

我们准备了一份只有 40 行的 IR（`code/reflect_demo.ll`），里面有两个函数：

```llvm
; 用 __CUDA_FTZ 选择 FTZ / 非 FTZ 两条路径
define float @fmul_ftz(float %a, float %b, float %c) {
entry:
  %r = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @ftz to ptr))
  %is_ftz = icmp ne i32 %r, 0
  br i1 %is_ftz, label %ftz, label %nftz
...
; 直接返回 __CUDA_ARCH
define i32 @arch_id() {
entry:
  %a = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @arch to ptr))
  ret i32 %a
}

; CUDA 工具链用这个 module flag 传 FTZ 设置
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"nvvm-reflect-ftz", i32 1}
```

## 三、先单独用 `opt` 跑：注意那个 `0`

```bash
opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -passes=nvvm-reflect \
    -S code/reflect_demo.ll
```

输出（`dumps/18-reflect-opt.ll`）：

```llvm
define float @fmul_ftz(float %a, float %b, float %c) #0 {
entry:
  br label %ftz                        ; <- 调用被折成常数，条件跳转随即被折掉

ftz:
  %m1 = fmul float %a, %b
  %s1 = fadd float %m1, %c
  ret float %s1

nftz:                                  ; No predecessors!
  ...
}

define i32 @arch_id() #0 {
entry:
  ret i32 0                            ; <- __CUDA_ARCH 折成了 0 !?
}
```

两个现象：

**现象一：`nftz` 那个块被标上了 `No predecessors!`。**

因为 `__CUDA_FTZ` 被折成了常数 `1`，那条 `br i1` 立刻判死。
**非 FTZ 那条路径虽然还打印在文本里，但已经没有任何前驱了**——
它要等后面某个 CFG 清理 pass 才被真正删掉。

**现象二：`arch_id` 返回了 `0`。**

那个 `0` 不是笔误：**单独跑 `opt` 时，pass 拿不到 SM 版本**。
它是由 `NVVMReflectPass(Subtarget.getSmVersion())` 构造进来的，
而 `opt -passes=nvvm-reflect` 用的是默认值 `0`。

## 四、再让真正的后端跑一遍：`890`

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -o - code/reflect_demo.ll
```

输出（`dumps/19-reflect-llc.ptx`，只摘 `arch_id`）：

```
.visible .func  (.param .b32 func_retval0) arch_id() // @arch_id
{
// %bb.0:                               // %entry
	st.param.b32 	[func_retval0], 890;
	ret;
}
```

**`890 = 89 * 10`**——sm_89 的架构编号。而且注意：那个常数被直接折进了 `st.param`
的立即数里，连一条 `mov` 都省了。

**为什么这次拿到了？** 因为 `llc -mcpu=sm_89` 让后端知道自己在为哪个 SM 编译，
于是 `NVVMReflectPass(890)` 把它折成了正确的值。

### 顺手看一个细节：`.version 7.8`

你可能会注意到这份 PTX 的头是 `.version 7.8`，而我们主案例的 PTX 是 `.version 8.7`。
原因不是版本变了，而是**这条命令没带 `-mattr=+ptx87`**：

```
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 ...              → ptx 默认版本（7.8）
clang++ ... --cuda-gpu-arch=sm_89 ...                          → clang 会带上 +ptx87
```

**"PTX 版本"是 target feature 决定的，不是 `-mcpu` 决定的**——这是个很容易混的点。

## 五、为什么这个 pass 必须排在流水线最前面

回到第 04-1 讲的流水线列表，`nvvm-reflect` 排在第 4 位。

为什么这么靠前？设想它排在后面会怎样：

```
__nvvm_reflect 的结果还没折成常数
   ↓
后面的 pass 看到的是一个"函数调用"，返回值未知
   ↓
分支不能判定、死代码不能删、常量传播做不了
   ↓
等你最后折完，前面的优化机会已经错过了
```

**折常数这件事，越早做收益越大。** 这是编译器里一条通用的经验：
"能把未知变已知的 pass 要尽早跑"。

同样的道理也解释了 `nvvm-intr-range`（第 5 项）为什么那么靠前——
它给 NVVM intrinsic 补上取值范围信息，后面的优化都靠这个信息。

## 六、FTZ 从哪来：module flag 和命令行开关

细心的话你会发现，FTZ 的值有两个来源：

1. **module flag**（我们 demo 文件里那行）：

   ```llvm
   !0 = !{i32 1, !"nvvm-reflect-ftz", i32 1}
   ```

   在真实编译里，clang 会按 `-ftz=true/false` 生成这个 flag。

2. **命令行开关**：`NVVMReflect::populateReflectMap` 会读上面那个 flag，
   同时也留了口子让你手工指定：

   ```bash
   opt -passes=nvvm-reflect -nvvm-reflect-add='__CUDA_FTZ=1' ...
   ```

（源码在 `llvm/lib/Target/NVPTX/NVVMReflect.cpp`，函数 `populateReflectMap`。）

**这一节的教学价值在于**：它让你看到"中端流水线"和"目标机扩展点"是**混在一起**的。
你看到的那 121 项里，有一部分是 target 自己插进来的。
读 pipeline 的时候别以为全是通用的 O2 流程。

## 七、小结

1. `__nvvm_reflect` 是**编译期查询**（FTZ 状态、目标架构），libdevice 大量使用它；
   NVVMReflectPass 负责把它折成常数并清理死支路。
2. 单独用 `opt` 跑时拿不到 SM 版本，`__CUDA_ARCH` 折出来是 `0`；
   走 `llc -mcpu=sm_89` 才会折成 `890`。
3. 它必须排在流水线最前面：**"把未知变已知"的 pass 越早越好**，
   否则后面的优化全都建立在"不知道"的基础上。

## 八、动手题

1. 跑这两条命令，对比 `arch_id` 的返回值：

   ```bash
   opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -passes=nvvm-reflect -S code/reflect_demo.ll | grep -A2 "arch_id"
   llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -o - code/reflect_demo.ll | grep -A3 "arch_id"
   ```
2. 把 demo 里的 `nvvm-reflect-ftz` 从 `i32 1` 改成 `i32 0`，重跑 `opt`，
   看看这次是哪条分支活着。

下一讲是这一部分的重头戏：**循环展开**。
