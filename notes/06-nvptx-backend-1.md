# 第 6 章 · NVPTX 后端 lowering（上）：IR → SelectionDAG → Machine IR

## 6.0 本章目标

前五章我们把 IR 读透了。从这一章开始，我们跟着一份 `.ll` 走进后端。

要回答的问题：

1. 后端流水线里到底有哪些阶段？（165 个 dump 项，我们把它们分组）
2. 进入 ISel 之前，NVPTX 还做了哪些**目标相关**的改写？（地址空间、kernel 参数、alloca）
3. SelectionDAG 把我们的 IR 变成了哪些机器指令？为什么有的 opcode 带 `INT_PTX_` 前缀，有的就是普通的 `SHLi32ri`？

## 6.1 后端流水线全景

前面我们一直用 `-stop-after=` 抓某一站，这次换个办法：`-print-after-all` 让它把每一站都打印出来，我们再从标题里把 pass 名字抽出来。

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -O2 -print-after-all \
    -o /dev/null dumps/02-device-O2.ll 2>&1 \
  | awk '/IR Dump After/ { sub(/^.*IR Dump After /, ""); sub(/ \(.*$/, ""); print }' \
  | uniq > dumps/25-codegen-passes.txt
```

结果 **165 项**（`dumps/25-codegen-passes.txt`）。这 165 项里真正有意思的是下面这些（序号按这一版的实测输出，不同小版本会漂移几个位置）：

| 序号 | pass | 属于哪一层 |
| --- | --- | --- |
| 2 | Pre-ISel Intrinsic Lowering | IR 层（codegen 前） |
| 4 | Replace occurrences of `__nvvm_reflect()` calls with 0/1 | IR 层，NVPTX 专属 |
| 5 | **Assign valid PTX names to globals** | IR 层，NVPTX 专属 |
| 6 | Ensure that the global variables are in the global address space | IR 层，NVPTX 专属 |
| 8 | **NVPTX Mark Kernel Pointers Global** | IR 层，NVPTX 专属 |
| 10 | **Lower pointer arguments of CUDA kernels** | IR 层，NVPTX 专属 |
| 12 | convert address space of alloca'ed memory to local | IR 层，NVPTX 专属 |
| 13 | Infer address spaces | IR 层 |
| 14 | NVPTX lower atomics of local memory | IR 层，NVPTX 专属 |
| 50 | **NVPTX Tag Invariant Loads** | IR 层，NVPTX 专属 |
| 51 | **NVPTX IR Peephole** | IR 层，NVPTX 专属 |
| 53/119 | CodeGen Prepare | IR 层（最后的 IR 整理） |
| 60 | **NVPTX specific alloca hoisting** | IR 层，NVPTX 专属 |
| 61 | **NVPTX DAG->DAG Pattern Instruction Selection** | **指令选择** |
| 62 | Finalize ISel and expand pseudo-instructions | 指令选择收尾 |
| 63–73 | Early Tail Dup / Machine LICM / Machine CSE / Machine code sinking / Peephole | 机器层优化 |
| 74 | **NVPTX Forward Params** | 机器层，NVPTX 专属 |
| 75 | **NVPTX Address Folder** | 机器层，NVPTX 专属 |
| 76 | **NVPTX Proxy Register Instruction Erasure** | 机器层，NVPTX 专属 |
| 79–84 | PHI 消除 / Two-Address / 寄存器合并 / **Machine Instruction Scheduler** | 编码前准备 |
| 86 | **NVPTX Prolog Epilog Pass** | 机器层，NVPTX 专属 |
| 90 | Control Flow Optimizer | 机器层 |
| 91 | Post-RA pseudo instruction expansion | 机器层 |
| 93 | Branch Probability Basic Block Placement | 机器层 |
| 99 | **NVPTX Assembly Printer** | 发射 PTX |

三条结论先摆出来：

1. **后端流水线 = "IR 尾巴 + ISel + 机器层优化"**，不是只有指令选择。第 56 项 `CodeGen Prepare` 还在改 IR，第 63 项还在做 NVPTX 专属的 IR 变换。
2. **NVPTX 的专属 pass 大量落在 IR 层**（assign PTX names / mark kernel ptrs global / lower args / tag invariant loads / IR peephole / alloca hoisting）。这是因为 PTX 和 LLVM IR 的抽象层次很接近，很多"后端活"可以在 IR 层就干完。注意 `NVPTX Forward Params`、`NVPTX Address Folder`、`NVPTX Prolog Epilog Pass` 这几个名字——它们说明**机器层也有 NVPTX 自己的活**，尤其"前导/后记代码"是它自己实现的。
3. **真的没有"寄存器分配"这个 pass**（第 7 章详述，这是本章最重要的伏笔）。

## 6.2 kernel 指针参数：generic → global

这是 NVPTX 后端里最关键的一个改写，而且它在 LLVM 24 里是**两个 pass 配合完成的**：

| pass | 文件 | 干什么 |
| --- | --- | --- |
| `NVPTX Mark Kernel Pointers Global`（第 8 项） | `llvm/lib/Target/NVPTX/NVPTXMarkKernelPtrsGlobal.cpp` | 在 kernel 的指针参数上**插一对 `addrspacecast`**，声明"这个指针指向 global" |
| `Lower pointer arguments of CUDA kernels`（第 10 项） | `llvm/lib/Target/NVPTX/NVPTXLowerArgs.cpp` | 处理 **byval 参数**（按值传进来的结构体），把它落到 `addrspace(101)`（param 空间） |
| `Infer address spaces`（第 13 项） | 通用 pass | 把上面那对 `addrspacecast` 折叠成一条，并把后续的 `load`/`store` 都染成 global |

先看第一个 pass 的源码注释（`NVPTXMarkKernelPtrsGlobal.cpp:8`）：

```
// For CUDA kernels, pointers loaded from byval parameters are known to be in
// global address space. This pass inserts addrspacecast pairs to make that
// explicit, enabling later address-space inference to propagate the global AS.
// It also handles the pattern where a pointer is loaded as an integer and then
// converted via inttoptr.
```

它干的事就是这个函数（同一个文件，第 30 行起）：

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

**先往 global 空间转一次，再转回 generic 并把所有使用者接过去**——这是一个"只加提示、不改语义"的写法，唯一目的是让后面的 `Infer address spaces` 看见"这条链来自 global"。

为什么这一步重要？因为**地址空间决定了 PTX 用哪种访存指令**：

| 地址空间 | PTX 访存 |
| --- | --- |
| generic | `ld` / `st`（需要运行时判断走哪条路径，慢） |
| global (1) | `ld.global` / `st.global` |
| shared (3) | `ld.shared` / `st.shared` |
| param (101) | `ld.param` / `st.param` |

我们 kernel 的参数 `const __half* A` 在 IR 里是 generic 指针（第 3 章讲过），如果不转换，后面的 load 就只能用通用 `ld`，性能差一截。这个 pass 给它前后各插一个 `addrspacecast`，随后 `NVPTXInferAddressSpaces` 把转发链吃掉，load 就变成了 `ld.global`。

### 证据链：IR → MIR → PTX

**IR 层**（喂给 ISel 的那份 IR，也就是 `dumps/26-isel.mir` 头部嵌的那一段）会看到 `addrspacecast`：

```llvm
define dso_local ptx_kernel void @mma_tc_manual(ptr noalias nofree noundef readonly captures(none) %0, ...) #0 {
  %7 = addrspacecast ptr %0 to ptr addrspace(1)     ; <- Mark Kernel Pointers Global 插的
  %8 = addrspacecast ptr %1 to ptr addrspace(1)
  %9 = addrspacecast ptr %2 to ptr addrspace(1)
```

**只剩一条了**——原来是"转过去再转回来"的一对，`Infer address spaces` 把回程那条连它的使用者一起折掉了，所以后面所有对 A/B/D 的访问都直接在 `addrspace(1)` 上做。

**MIR 层**（`dumps/26-isel.mir`，搜索 `cvta`）：

```
%0:b64 = cvta_to_global_64 killed %46
%1:b64 = cvta_to_global_64 killed %47
%2:b64 = cvta_to_global_64 killed %48
```

一共 6 处 `cvta_to_global_64`（两个 kernel × 3 个指针参数），源操作数就是上面 `LD_i64` 读出来的参数值。**这就是前端那个 generic 参数在后端第一次被"定性"的地方。**

**PTX 层**（`dumps/03-clang-O2.ptx`）：

```
ld.param::entry.b64 	%rd8, [mma_tc_manual_param_2];
cvta.to.global.u64 	%rd3, %rd8;      <- 这里
...
ld.param::entry.b64 	%rd6, [mma_tc_manual_param_0];
cvta.to.global.u64 	%rd1, %rd6;
ld.param::entry.b64 	%rd7, [mma_tc_manual_param_1];
cvta.to.global.u64 	%rd2, %rd7;
```

`ld.param::entry.b64` 读的是 kernel 参数区（`.param::entry` 是新版 PTX 里"入口函数参数"的写法，等价于老的 `ld.param.u64`），`cvta.to.global.u64` 把它变成 global 空间的地址。后面所有对 A/B/D 的访存都能用 `ld.global` / `st.global` 了。

顺带一提 `ld.param` 在 MIR 里长这样：

```
%37:b64 = LD_i64 0, 0, 101, 3, 64, -1, <mcsymbol mma_tc_manual_param_0>, 0, 0, $noreg ::
          (dereferenceable invariant load (s64), addrspace 101)
```

**`addrspace(101)` 就是 PTX 的参数空间**（NVPTX 私有的地址空间编号，`NVPTXAddrSpace.h` 里的 `ADDRESS_SPACE_ENTRY_PARAM`），`LD_i64` 是加载参数用的机器指令，`<mcsymbol ...>` 就是那个 `$param` 符号。

## 6.3 SelectionDAG 怎么看：工具限制与替代方案

需求清单里提到用 `-debug-only=isel` 抓 SelectionDAG。这条命令**只在开启了断言的 LLVM 构建里存在**，本机这套是 Release 构建：

```
$ llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -debug-only=isel ...
llc: Unknown command line argument '-debug-only=isel'.  Try: 'llc --help'
llc: Did you mean '--debug-pass=isel'?
```

所以本章用这套替代方案，全部在 Release 构建里可用：

| 想看什么 | 命令 |
| --- | --- |
| ISel **之前**的 IR | `llc -print-before=nvptx-isel` |
| ISel **之后**的 MIR | `llc -stop-after=finalize-isel` |
| 每个 pass 之后的 IR/MIR | `llc -print-after-all` |
| 自己构建 LLVM 的话 | `-debug-only=isel`（需要 `-DLLVM_ENABLE_ASSERTIONS=ON`） |

ISel 本身分三步（这一步是 LLVM 通用流程，不是 NVPTX 特有）：

```
SelectionDAG 构建        每个 IR 指令 → 一个或多个 DAG 节点（ISD::ADD、ISD::LOAD、ISD::CALLSEQ_START...）
  ↓ DAG Combine / Legalize
   目标合法的 DAG        类型合法化（把不支持的向量拆开）、操作合法化（把不支持的运算展开成库调用）
  ↓ Pattern Match（TableGen 生成）
   机器指令              DAG 节点匹配到 .td 里的 pattern，产出 MachineInstr
  ↓
   MachineFunction       ScheduleDAG 排序 → 基本块 + 指令序列
```

我们这份代码在 Legalize 阶段**几乎没有工作要做**——因为 `<4 x i8>`、`<2 x i16>` 这些东西在 C++ 层就被我们拆成了 `uint32_t`（第 3.7 节讲的"用整数视角看 half"），IR 里根本没有向量运算。**这是手写 fragment 代码的一个隐性好处：把类型合法化的活儿在源码层就干完了。**

## 6.4 ISel 结果：NVPTX 专有 opcode vs 通用 opcode

ISel 之后的 MIR 在 `dumps/26-isel.mir`。我们把里面的机器 opcode 统计出来：

```bash
grep -oE 'INT_PTX_[A-Za-z_0-9]+|BARRIER_[A-Za-z_0-9]+|LD_GLOBAL_NC_[A-Za-z0-9_]+|LD_i(32|64)|ST_i32|cvta_to_global_64|MUL_WIDE[A-Za-z0-9_]+|SHL32_ri|SRL32_ri|AND_b32ri|CVT_[a-z0-9_]+|ADD(32|64)(rr|ri)|MULT32rr|SETP_i32[a-z]+' \
     dumps/26-isel.mir | sort | uniq -c | sort -rn
```

```
     34 ADD64rr
     13 SHL32_ri
     10 MUL_WIDEu32_ri
     10 MULT32rr
      9 MUL_WIDEs32_ri
      8 ST_i32
      7 ADD32rr
      6 cvta_to_global_64
      6 LD_i64
      6 AND_b32ri
      5 CVT_s64_s32
      5 ADD64ri
      5 ADD32ri
      4 LD_i32
      4 LD_GLOBAL_NC_i32
      4 LD_GLOBAL_NC_i16
      4 CVT_u64_u32
      4 CVT_u32_u64
      3 SRL32_ri
      3 SETP_i32ri
      2 SETP_i32rr
      2 INT_PTX_SREG_TID_x
      2 INT_PTX_SREG_CTAID_y
      2 INT_PTX_SREG_CTAID_x
      2 BARRIER_CTA_SYNC_ALIGNED_ALL_i
```

这些 opcode 可以分成三类，每一类都能对上源码里的一个决定：

> **给老读者提个醒**：LLVM 24 把 NVPTX 的机器指令命名改了一遍。老版本里那条 `INT_PTX_LDG_GLOBAL_i32ari64` 现在叫 `LD_GLOBAL_NC_i32`，`SHLi32ri` 现在叫 `SHL32_ri`，`MULWIDES64Imm` 现在叫 `MUL_WIDEs32_ri`。**读机器名的时候别只认记忆，dump 一次最快。**

### 第一类：读特殊寄存器（`INT_PTX_` 前缀）

| opcode | 来自源码 | 说明 |
| --- | --- | --- |
| `INT_PTX_SREG_TID_x` / `_CTAID_x` / `_CTAID_y` | `threadIdx.x` / `blockIdx.x/y` | 读特殊寄存器（sreg） |
| `BARRIER_CTA_SYNC_ALIGNED_ALL_i` | `__syncthreads()` | 对应 `llvm.nvvm.barrier.cta.sync.aligned.all`，`_i` 表示带一个立即数（barrier 编号） |

**这一版只有 sreg 读取和 barrier 还顶着 `INT_PTX_`/专用名字。** 老的 `INT_PTX_LDG_GLOBAL_i32ari64` 那种"把寻址模式编进名字"的写法被废弃了，访存指令统一成下面第二类的短名字。

### 第二类：NVPTX 专有，但不带前缀

| opcode | 说明 |
| --- | --- |
| `LD_i32` / `LD_i64` | 从**参数空间**（addrspace 101）加载，操作数里那个 `<mcsymbol ..._param_N>` 就是参数名 |
| `ST_i32` | 往参数/全局空间存（byval 参数会被拷一份，就靠它） |
| `LD_GLOBAL_NC_i32` / `LD_GLOBAL_NC_i16` | 从 **global** 空间读，**`NC` = non-coherent**（只读缓存路径，PTX 里的 `ld.global.nc`）。它对应源码里 `const __half* __restrict__` 推出来的 `readonly` 属性——**属性直接写进了 opcode 名字** |
| `cvta_to_global_64` | generic → global 的地址转换（第 6.2 节那条链的中间站） |
| `MUL_WIDEu32_ri` / `MUL_WIDEs32_ri` | 32×32→64 的乘法（wide multiply），`u`/`s` = 无符号/有符号，`ri` = 寄存器×立即数 |
| `SHL32_ri` / `SRL32_ri` / `AND_b32ri` / `ADD32rr` / `ADD64ri` / `MULT32rr` / `CVT_u32_u64` … | 通用整数运算，但**名字是 NVPTX 自己的**（`rr` = reg,reg；`ri` = reg,imm；`32`/`64` 是宽度） |
| `SETP_i32ri` / `SETP_i32rr` | 比较并写谓词（set predicate），立即数版本/寄存器版本 |

**这里有个容易搞混的点**：`ADD32rr`、`SHL32_ri` 这种名字看起来像通用 IR，其实它们是 **NVPTX 目标自己的机器指令**。而 `ISD::ADD`、`ISD::SHL` 是 DAG 节点，不是机器指令。MIR 里出现的都是机器指令。

还有一点值得注意：`LD_GLOBAL_NC_*` 里带数据类型（`i32`/`i16`），而 `LD_i32`/`LD_i64` 不带地址模式后缀——**同一个"load"，在 NVPTX 里按地址空间分成了不同家族**。这就是第 3 章"地址空间决定访存指令"那句话在机器层的落地。

### 第三类：真正的通用机器指令和伪指令

| opcode | 说明 |
| --- | --- |
| `CBranch` / `GOTO` / `Return` | 控制流（TargetOpcode 里的通用伪指令） |
| `PHI` | SSA 合并点，后面会被 PHI 消除 pass 干掉 |
| `COPY` | 寄存器复制 |
| `INLINEASM` | 我们的 `mma` / `ldmatrix` / `cp.async` |

`INLINEASM` 值得单独看一眼（`dumps/26-isel.mir`）：

```
INLINEASM &"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};\0A",
          sideeffect isconvergent attdialect,
          regdef:B32, def %78, regdef:B32, def %79, regdef:B32, def %80, regdef:B32, def %81,
          reguse:B32, %82, reguse:B32, %83, reguse:B32, %84, reguse:B32, %85,
          reguse:B32, %86, reguse:B32, %87, reguse:B32, %88,
          reguse:B32, %89, reguse:B32, %90, reguse:B32, %91, !18
```

每个操作数前面都标着**寄存器类别**：4 个 `regdef:B32`（`=f` 输出）、6 个 `reguse:B32`（A/B 的 `r` 输入）、4 个 `reguse:B32`（C 的 `f` 输入）。

**这些类别就是第 3 章那条约束字符串 `"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"` 在后端的落地形式。** 前端写约束 → ISel 翻译成寄存器类别 → 寄存器分配/合并时按类别处理 → AsmPrinter 按类别选 PTX 寄存器名（第 7 章）。

有趣的是：`=f`（浮点约束）在这里也是 `B32`，和 `r` 用的同一个寄存器类。因为 NVPTX 的 PTX 虚拟寄存器本来就"只有宽度、没有类型"，f32 和 b32 是同一批 32 位寄存器（对照第 7 章 PTX 里那段 `.reg .b32 %r<37>;`——**连一个 `.f32` 声明都没有**）。

## 6.5 MIR 逐段读

我们挑 `mma_tc_ldmatrix` 的函数头看（`dumps/26-isel.mir`）：

```
body:             |
  bb.0 (%ir-block.6):
    successors: %bb.2(0x50000000), %bb.1(0x30000000)

    %36:b32 = LD_i32 0, 0, 101, 3, 32, -1, <mcsymbol mma_tc_ldmatrix_param_5>, 0, 0, $noreg :: (dereferenceable invariant load (s32), addrspace 101)
    %35:b32 = LD_i32 0, 0, 101, 3, 32, -1, <mcsymbol mma_tc_ldmatrix_param_4>, 0, 0, $noreg :: (dereferenceable invariant load (s32), addrspace 101)
    %37:b64 = LD_i64 0, 0, 101, 3, 64, -1, <mcsymbol mma_tc_ldmatrix_param_0>, 0, 0, $noreg :: (dereferenceable invariant load (s64), addrspace 101)
    ...
    %0:b64 = cvta_to_global_64 killed %37
    ...
    %3:b32 = INT_PTX_SREG_TID_x
    %40:b32 = INT_PTX_SREG_CTAID_y
    %4:b32 = nuw nsw SHL32_ri killed %40, 4        ; tile_m = blockIdx.y * 16
    %41:b32 = INT_PTX_SREG_CTAID_x
    %5:b32 = SHL32_ri killed %41, 3                ; tile_n = blockIdx.x * 8
    %42:b1 = SETP_i32ri %36, 0, 4                  ; K > 0 ?
    CBranch killed %42, %bb.2, 0
    GOTO %bb.1
```

逐行对照源码，你会发现**机器层和源码的对应关系比 IR 更直接**：

- `LD_i32 ... <mcsymbol mma_tc_ldmatrix_param_5>` = 读第 6 个参数（`K`），操作数里直接写着参数名；
- `INT_PTX_SREG_TID_x` = `threadIdx.x`，对应 PTX 的 `mov.u32 %r1, %tid.x;`；
- `SHL32_ri %40, 4` = 乘以 16（`blockIdx.y * 16`），前面那个 `nuw nsw` 是从 IR 一路带下来的标记；
- `SETP_i32ri %36, 0, 4` = "set predicate"，把 `K` 和 0 比较，谓词条件码 `4` = `>`（这是 NVPTX 的 ICC/条件码编码）；
- `CBranch` + `GOTO` = 条件跳转 + 无条件跳转。

再看循环体里 `ldmatrix` 那一段（同一个文件）：

```
    INLINEASM &"cp.async.cg.shared.global [$0], [$1], 16;\0A", sideeffect isconvergent attdialect,
              reguse:B32, %79, reguse:B64, %80, !21
    ...
    INLINEASM &"cp.async.commit_group;\0A", sideeffect isconvergent attdialect, !23
    INLINEASM &"cp.async.wait_group 0;\0A", sideeffect isconvergent attdialect, !24
    BARRIER_CTA_SYNC_ALIGNED_ALL_i 0
    %87:b32 = COPY %12
    INLINEASM &"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A",
              sideeffect isconvergent attdialect,
              regdef:B32, def %83, regdef:B32, def %84, regdef:B32, def %85,
              regdef:B32, def %86, reguse:B32, %87, !25
```

**注意这里的寄存器类别又出现了**：`cp.async` 的共享内存地址是 `B32`（32 位），全局地址是 `B64`（64 位）——正是第 3 章说的"两边宽度不一样"。`ldmatrix` 用 4 个 `B32` def + 1 个 `B32` use。

到这里，**前端写的约束、IR 里的类型、MIR 里的寄存器类别**三者完全对齐了。

## 6.6 小结与衔接

- 后端流水线 165 项，**IR 尾巴很长**：`Assign valid PTX names`、`Mark Kernel Pointers Global`、`Lower pointer arguments`、`NVPTX IR Peephole`、`NVPTXAllocaHoisting`、`Infer address spaces` 都在 ISel 之前。
- kernel 的 generic 参数被"定性"成 global，是**两个 pass 加一次推断**的接力：`NVPTXMarkKernelPtrsGlobal` 插 `addrspacecast` 对 → `Infer address spaces` 折成一条。**证据链**：IR 的 `addrspacecast ptr %0 to ptr addrspace(1)` → MIR 的 `cvta_to_global_64` → PTX 的 `cvta.to.global.u64`。
- ISel 之后，**NVPTX 专属 opcode（`LD_GLOBAL_NC_i32`、`SHL32_ri`、`SETP_i32ri` 等）和通用机器指令（`CBranch`、`COPY`、`PHI`）混在一起**；`INLINEASM` 原样保留，只带上寄存器类别约束（现在打印成 `regdef:B32` 这样的可读名字）。
- `-debug-only=isel` 在 Release 构建不可用，替代方案是 `-print-before/-print-after/-stop-after`。

下一章我们继续往右走：**MIR 怎么变成 PTX 文本**。中间你会看到一个反直觉的事实——**NVPTX 后端根本不做寄存器分配**。
