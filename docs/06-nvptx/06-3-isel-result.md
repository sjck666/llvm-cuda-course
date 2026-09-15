# 06-3 · ISel 之后：opcode 家族与命名规律

指令选择跑完之后，我们手里是 `dumps/26-isel.mir`。这一讲做一件很实用的事：
**给里面的机器指令分类，并总结它们的命名规律。**

学会这个，你以后读任何后端的 MIR 都不会两眼一抹黑。

## 一、先统计一遍

```bash
grep -oE 'INT_PTX_[A-Za-z_0-9]+|BARRIER_[A-Za-z_0-9]+|LD_GLOBAL_NC_[A-Za-z0-9_]+|LD_i(32|64)|ST_i32|cvta_to_global_64|MUL_WIDE[A-Za-z0-9_]+|SHL32_ri|SRL32_ri|AND_b32ri|CVT_[a-z0-9_]+|ADD(32|64)(rr|ri)|MULT32rr|SETP_i32[a-z]+' \
     dumps/26-isel.mir | sort | uniq -c | sort -rn
```

实测输出（节选）：

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

**给老读者提个醒**：LLVM 24 把 NVPTX 的机器指令命名改过一遍。
老版本里那条 `INT_PTX_LDG_GLOBAL_i32ari64` 现在叫 `LD_GLOBAL_NC_i32`，
`SHLi32ri` 现在叫 `SHL32_ri`，`MULWIDES64Imm` 现在叫 `MUL_WIDEs32_ri`。

> **读机器指令名时别只凭记忆，dump 一次最快。**

## 二、三个家族

### 家族一：读特殊寄存器（还顶着 `INT_PTX_` 前缀）

| opcode | 来自源码 | 说明 |
| --- | --- | --- |
| `INT_PTX_SREG_TID_x` / `_CTAID_x` / `_CTAID_y` | `threadIdx.x` / `blockIdx.x/y` | 读特殊寄存器（sreg） |
| `BARRIER_CTA_SYNC_ALIGNED_ALL_i` | `__syncthreads()` | 末尾 `_i` 表示"带一个立即数操作数"（barrier 编号） |

**这一版只有 sreg 读取和 barrier 还保留着 `INT_PTX_` 这类老式命名**，
其它访存指令都改成了短名字（见家族二）。

### 家族二：NVPTX 专有，但不带前缀

| opcode | 说明 |
| --- | --- |
| `LD_i32` / `LD_i64` | 从 **param 空间**（addrspace 101）加载；操作数里那个 `<mcsymbol ..._param_N>` 就是参数名 |
| `ST_i32` | 往内存里存（比如写返回值到 `func_retval0`） |
| `LD_GLOBAL_NC_i32` / `LD_GLOBAL_NC_i16` | 从 **global** 空间读；**`NC` = non-coherent**（只读缓存路径） |
| `cvta_to_global_64` | generic → global 的地址转换（第 06-2 讲那条链的中间站） |
| `MUL_WIDEu32_ri` / `MUL_WIDEs32_ri` | 32×32→64 的乘法；`u`/`s` = 无符号/有符号，`ri` = 寄存器×立即数 |
| `SHL32_ri` / `SRL32_ri` / `AND_b32ri` / `ADD32rr` / `ADD64ri` / `MULT32rr` | 通用整数运算，但**名字是 NVPTX 自己的** |
| `CVT_u32_u64` / `CVT_s64_s32` … | 类型转换（`CVT_源_目标`） |
| `SETP_i32ri` / `SETP_i32rr` | 比较并写谓词（set predicate） |

### 家族三：真正的通用机器指令和伪指令

| opcode | 说明 |
| --- | --- |
| `CBranch` / `GOTO` / `RETURN` | 控制流（框架里的通用伪指令） |
| `PHI` | SSA 合并点——**注意它在 MIR 里还活着**，要到 PHI 消除才消失（第 05-9 讲） |
| `COPY` | 寄存器复制 |
| `INLINEASM` | 我们写的 `mma` / `ldmatrix` / `cp.async` |

## 三、命名规律：一套可以推导的语法

把家族二的命名规则总结一下，你以后看到新指令名能自己拆：

```
LD_GLOBAL_NC_i32
│  │      │  └──── 数据类型（i32 / i16 / b32）
│  │      └─────── 缓存性质：NC = non-coherent（只读缓存）
│  └────────────── 落在哪个地址空间：GLOBAL
└───────────────── 操作类别：LD = load

SHL32_ri
│   │ └── 操作数形态：ri = register + immediate（rr = reg+reg）
│   └──── 位宽：32
└──────── 操作：SHL = 左移

MUL_WIDEs32_ri
│   │    │  │ └── ri = 寄存器×立即数
│   │    │  └──── 32 = 输入位宽
│   │    └─────── s = 有符号（u = 无符号）
│   └──────────── WIDE = 结果比输入宽（32×32→64）
└──────────────── 乘法
```

**这套命名不是巧合，它来自 `.td` 里的 `multiclass` 拼接规则**
（第 05-4 讲那个 `defm ADD : I3<"add.s", add, ...>`）：

```
名字 = 前缀 + 位宽 + 操作数形态
```

所以你完全可以从名字反推它是怎么被生成的。

## 四、验证：从 MIR 的 opcode 一路追回 `.td`

我们拿 `LD_GLOBAL_NC_i32` 走一遍（第 05-3 讲做过一次，这里再走一遍加深印象）：

```
MIR:   %82:b32 = LD_GLOBAL_NC_i32 3, 32, -1, %28, -16, 0, $noreg
   ↓ 在生成文件里搜名字
NVPTXGenInstrInfo.inc:  LD_GLOBAL_NC_i32 = 1523, // NVPTXIntrinsics.td:3449
   ↓ 去那个行号
NVPTXIntrinsics.td:3449: def LD_GLOBAL_NC_i32 : LDG_G<B32>;
   ↓ 看模板
NVPTXIntrinsics.td:3439: class LDG_G<NVPTXRegClass regclass>
                          : NVPTXInst<(outs regclass:$result), (ins AtomicCode:$Sign,
                            i32imm:$fromWidth, UsedBytesMask:$usedBytes, ADDR:$src, ...),
                            !strconcat(...)>;
```

于是 MIR 里那串数字就对上了：

```
3        → Sign（符号相关枚举值）
32       → fromWidth（位宽）
-1       → usedBytes（有效字节掩码，-1 = 全用）
%28, -16 → ADDR（基寄存器 + 立即数偏移）
0        → evictionAndPrefetchHint
$noreg   → policy（没有额外缓存策略）
```

**这就是读 MIR 的标准方法：看 opcode → 查 `.td` → 按位置对操作数。**

## 五、`INLINEASM` 的寄存器类别

内联汇编在 MIR 里的长这样（`dumps/26-isel.mir:606`）：

```
INLINEASM &"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};\0A",
          sideeffect isconvergent attdialect,
          regdef:B32, def %78, regdef:B32, def %79, regdef:B32, def %80, regdef:B32, def %81,
          reguse:B32, %82, reguse:B32, %83, ..., !18
```

每个操作数前面都标着**寄存器类别**：

```
4 个 regdef:B32    ← "=f" 约束的 4 个输出
6 个 reguse:B32    ← A/B 的 "r" 输入（4 + 2）
4 个 reguse:B32    ← C 的 "f" 输入
```

**注意：`=f`（浮点）和 `r`（整型）在这里都是 `B32`。**
因为 NVPTX 的虚拟寄存器按宽度分类（第 05-3 讲的 `def B32 :
NVPTXRegClass<[i32, ..., f32], 32, ...>`），
浮点和整型共用同一批 32 位寄存器。

老版本的 MIR 打印的是数字 ID（`262153 /* reguse:Int32Regs */`），
新版直接打名字，读起来省事多了。

## 六、小结

1. ISel 之后的 opcode 分成三大家族：**sreg/barrier**（`INT_PTX_*`）、
   **NVPTX 专有的短名字**（`LD_GLOBAL_NC_i32`、`SHL32_ri`…）、
   **通用伪指令**（`CBranch`、`COPY`、`PHI`、`INLINEASM`）。
2. 命名是有规律的：`前缀 + 位宽 + 操作数形态`（`ri` / `rr`），
   加上 `u`/`s`、`NC`、`WIDE` 之类的修饰——**它们来自 `.td` 里的拼接规则**。
3. 读 MIR 的标准方法：**看 opcode → 在生成文件里查名字（带 `.td` 行号）→
   按 `.td` 的操作数表对位置**。

## 七、动手题

1. 自己挑一个 opcode 走一遍上面的追查流程（建议用 `SETP_i32ri` 或 `MUL_WIDEs32_ri`）：

   ```bash
   grep -n "SETP_i32ri" /root/llvm-build/lib/Target/NVPTX/NVPTXGenInstrInfo.inc | head -2
   ```

2. 数一数我们这份 MIR 里 `INLINEASM` 出现了几次，分别对应源码里的哪几条汇编。

下一讲我们把 MIR 从头读到尾，把每一行都对回源码。
