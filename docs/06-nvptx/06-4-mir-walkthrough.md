# 06-4 · MIR 逐段精读

这一讲我们做第 3 部分做过的同一件事，但**换到 MIR 这一层**：
把 `mma_tc_ldmatrix` 的前半段一行一行读完。

读完你会发现一个规律：**MIR 比 IR 更"啰嗦"，但也更"直白"**——
每条指令做的事都很小，组合起来却很明确。

## 一、先看函数头（第 800 行附近）

```bash
sed -n '796,810p' dumps/26-isel.mir      # mma_tc_ldmatrix 的函数头
```

对 `mma_tc_ldmatrix` 来说：

```
body:             |
  bb.0 (%ir-block.6):
    successors: %bb.2(0x50000000), %bb.1(0x30000000)

    %36:b32 = LD_i32 0, 0, 101, 3, 32, -1, <mcsymbol mma_tc_ldmatrix_param_5>, 0, 0, $noreg
              :: (dereferenceable invariant load (s32), addrspace 101)
    %35:b32 = LD_i32 0, 0, 101, 3, 32, -1, <mcsymbol mma_tc_ldmatrix_param_4>, 0, 0, $noreg
              :: (dereferenceable invariant load (s32), addrspace 101)
    %37:b64 = LD_i64 0, 0, 101, 3, 64, -1, <mcsymbol mma_tc_ldmatrix_param_0>, 0, 0, $noreg
              :: (dereferenceable invariant load (s64), addrspace 101)
    %0:b64 = cvta_to_global_64 killed %37
    ...
```

逐行说：

| 行 | 对应什么 |
| --- | --- |
| `bb.0 (%ir-block.6)` | 基本块 0，来自 IR 的 `%6` 这个块 |
| `successors: %bb.2(0x50000000), %bb.1(0x30000000)` | 后继块与概率（`0x50000000` ≈ 31%，`0x30000000` ≈ 19%） |
| `%36:b32 = LD_i32 ... param_5` | 读第 6 个参数（`K`） |
| `%35:b32 = LD_i32 ... param_4` | 读第 5 个参数（`N`） |
| `%37:b64 = LD_i64 ... param_0` | 读第 1 个参数（A 的指针） |
| `%0:b64 = cvta_to_global_64 killed %37` | 把它定性成 global（第 06-2 讲那条链） |

三个可以留意的点：

1. **`killed` 是"这个虚拟寄存器在这里最后一次被使用"**。
   它对寄存器分配器很重要（用完就能复用），在 NVPTX 上则主要影响合并。
2. **参数只读出了用到的三个**（`K`=param_5、`N`=param_4、A/B/D 三个指针），
   `M`（param_3）**没有被读**——印证了第 02-2 讲"签名里有不代表会被用"。
3. **`:: (dereferenceable invariant load (s32), addrspace 101)`**
   是这条指令的"内存操作数信息"：从 addrspace 101 读一个 32 位、可解引用、
   不变（invariant）的值。这些信息来自 IR 的元数据（比如 `!invariant.load`）。

## 二、接着是 sreg 与算术

```
    %3:b32 = INT_PTX_SREG_TID_x
    %40:b32 = INT_PTX_SREG_CTAID_y
    %4:b32 = nuw nsw SHL32_ri killed %40, 4          ; tile_m = blockIdx.y * 16
    %41:b32 = INT_PTX_SREG_CTAID_x
    %5:b32 = SHL32_ri killed %41, 3                  ; tile_n = blockIdx.x * 8
    %42:b1 = SETP_i32ri %36, 0, 4                    ; K > 0 ?
    CBranch killed %42, %bb.2, 0
    GOTO %bb.1
```

对照第 03-4 讲的 IR，你会发现**机器层和源码的对应关系比 IR 更直接**：

| MIR | 源码 |
| --- | --- |
| `INT_PTX_SREG_TID_x` | `threadIdx.x` |
| `SHL32_ri %40, 4` | `blockIdx.y * 16` |
| `SETP_i32ri %36, 0, 4` | `K > 0`（条件码 `4` = `>`） |
| `CBranch` + `GOTO` | 条件跳转 + 无条件跳转 |

几个细节：

1. **`nuw nsw` 跟着 SHL 一起下来了**——IR 里的标记会一路传到机器指令上，
   告诉后端"这个移位不会溢出"。这影响后端能不能做进一步的化简。
2. **`CBranch killed %42, %bb.2, 0`** 末尾那个 `0` 在 IR 里是没有的——
   它是机器层的信息（分支的某种标志/权重）。
   **这就是 MIR 比 IR 多出来的东西：硬件/编码相关的附加操作数。**
3. **`SETP_i32ri %36, 0, 4`** 里的 `4` 是 NVPTX 的条件码编码
   （这个编码来自 `.td` 里的 `PatLeaf`/`SDNodeXForm`，属于"目标约定"）。

## 三、循环体（第 914 行附近）

```bash
sed -n '914,926p' dumps/26-isel.mir
```

```
    INLINEASM &"cp.async.cg.shared.global [$0], [$1], 16;\0A", sideeffect isconvergent attdialect,
              reguse:B32, %79, reguse:B64, %80, !21
    ...
    INLINEASM &"cp.async.ca.shared.global [$0], [$1], 8;\0A", sideeffect isconvergent attdialect,
              reguse:B32, %81, reguse:B64, %82, !22
    INLINEASM &"cp.async.commit_group;\0A", sideeffect isconvergent attdialect, !23
    INLINEASM &"cp.async.wait_group 0;\0A", sideeffect isconvergent attdialect, !24
    BARRIER_CTA_SYNC_ALIGNED_ALL_i 0
    %87:b32 = COPY %12
    INLINEASM &"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A",
              sideeffect isconvergent attdialect,
              regdef:B32, def %83, regdef:B32, def %84, regdef:B32, def %85,
              regdef:B32, def %86, reguse:B32, %87, !25
    %90:b32 = COPY %13
    INLINEASM &"ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0A",
              sideeffect isconvergent attdialect,
              regdef:B32, def %88, regdef:B32, def %89, reguse:B32, %90, !26
    INLINEASM &"mma.sync...", sideeffect isconvergent attdialect,
              regdef:B32, def %91, ..., !18
```

**一眼就能看出这一段的"骨架"**：

```
cp.async ×2  →  commit  →  wait  →  barrier  →  ldmatrix ×2  →  mma
```

和我们在源码里写的一模一样！（中间穿插的那些 `MUL_WIDEs32_ri` / `ADD64rr` / `COPY`
是在算下一次拷贝的地址。）

**这说明这一版 LLVM 基本上没动这个循环的顺序**——真正的重排发生在 ptxas 那边，
第 8 部分会看到它把 `cp.async` 提前了整整一轮（软件流水）。

三个细节值得说：

1. **`cp.async` 的两个操作数类别不同**：
   `reguse:B32`（共享内存地址，32 位）+ `reguse:B64`（全局地址，64 位）。
   这正是第 02-6 讲说的"两边宽度不一样"——也是手写 `cp.async` 最容易错的地方。
2. **`BARRIER_CTA_SYNC_ALIGNED_ALL_i 0`** 就是 `__syncthreads()`。
   注意它**没有** `isconvergent` 标记，因为它本身就是一条机器指令（不是内联汇编），
   收敛性是它固有的性质，不需要额外标注。
3. **`%87:b32 = COPY %12`** 这行很有意思：`ldmatrix` 需要一个 32 位的地址，
   而 IR 里那个地址在 `%12`，所以插一条 COPY 把它搬到新的虚拟寄存器上。
   **这类 COPY 就是寄存器合并（coalescing）要消灭的目标**（第 05-9 讲）。

## 四、怎么读任意一条不认识的指令

你现在应该有一套固定流程了：

```
1. 看 opcode 名字 → 猜家族（前缀/位宽/操作数形态）
2. 在 NVPTXGenInstrInfo.inc 里搜名字 → 得到 .td 的行号
3. 去 .td 看 (outs ...) / (ins ...) → 知道每个位置的语义
4. 回到 MIR 按位置对号入座
```

举个完整的例子（`MUL_WIDEs32_ri`）：

```bash
grep -n "MUL_WIDEs32_ri\s*=" /root/llvm-build/lib/Target/NVPTX/NVPTXGenInstrInfo.inc | head -2
grep -rn "MUL_WIDEInst\|MUL_WIDEs32" /root/llvm-project/llvm/lib/Target/NVPTX/NVPTXInstrInfo.td | head -5
```

你会看到它来自 `multiclass MULWIDEInst<...>`——又是一次"模板批量生成"。

## 五、MIR 里的"多出来的东西"

把 MIR 和 IR 摆在一起看，MIR 多了三类信息：

| 多了什么 | 例子 | 为什么需要 |
| --- | --- | --- |
| **寄存器类别** | `%36:b32` | 寄存器分配/合并要知道"能放进哪一类" |
| **指令附加操作数** | `CBranch killed %42, %bb.2, 0` 末尾的 `0` | 编码/调度需要 |
| **内存操作数信息** | `:: (dereferenceable invariant load (s32), addrspace 101)` | 别名分析、重排、优化 |

而 MIR 少了什么？**少了 SSA 的"单赋值"约束**（虚拟寄存器可以被多次赋值），
也少了类型系统的大部分（只有寄存器类 + 指令自带的类型信息）。

**一句话：IR 是给"优化"看的，MIR 是给"分配和发射"看的。**

## 六、小结

1. MIR 的读法和 IR 一样是"逐行对源码"，但语义一半在 opcode 上、一半在
   **按位置排列的操作数**上——后者要去 `.td` 里查。
2. 我们案例的循环体在 MIR 里几乎**原样保留**了源码的顺序
   （`cp.async ×2 → commit → wait → barrier → ldmatrix ×2 → mma`），
   重排发生在 ptxas 那一侧。
3. MIR 比 IR 多了三类信息：**寄存器类别、指令附加操作数、内存操作数信息**；
   少了 SSA 的单赋值约束。**IR 给优化看，MIR 给分配和发射看。**

## 七、动手题

1. 用上面的四步流程，搞清楚 `%105:b64 = MUL_WIDEs32_ri %22, 2` 这条指令的
   每个操作数是什么意思。
2. 数一数 `mma_tc_ldmatrix` 的循环体里有多少条 `INLINEASM`，
   和源码里的内联汇编数量对一下（提示：源码里 `mma` 一条、`ldmatrix` 两条、
   `cp.async` 四条 = 七条）。

下一讲我们走完最后一站：**从 MIR 到 PTX——AsmPrinter 做了什么。**
