# 05-3 · TableGen（上）：寄存器与指令是怎么"生成"出来的

从这一讲开始，我们进入 LLVM 后端设计里最独特的一块：**TableGen**。

如果你只从这五讲里学一样东西，我希望是它——因为**这是 LLVM 后端"能用"的根本原因**。

## 一、先看问题：如果没有 TableGen 会怎样

我们要描述一条指令。以 NVPTX 的 `LD_GLOBAL_NC_i32` 为例，至少要说清这些：

```
名字          LD_GLOBAL_NC_i32
汇编写法       ld.global.nc.b32 %r1, [%rd2];
输出操作数     一个 32 位寄存器
输入操作数     符号(Sign)、位宽(fromWidth)、有效字节掩码、地址、驱逐提示、缓存策略
副作用         不访问内存？不，它是 load
选择模式       当 IR 里有一个"从 addrspace(1) 读 i32"的节点时，把它选成这条指令
特性要求       需要哪个 SM / PTX 版本
调度信息       几个周期、占什么资源
```

这还只是**一条**。NVPTX 有多少条指令？看生成的枚举文件：

```bash
grep -c "= [0-9]*, // NVPTX" /root/llvm-build/lib/Target/NVPTX/NVPTXGenInstrInfo.inc
```

```
9743
```

**近一万条。** 而且这还只是 NVPTX——一个相对"薄"的后端。

如果每条指令都手写 C++ 来描述上面那一堆信息，会发生什么？

1. **重复到崩溃**：`add.s16`/`add.s32`/`add.s64` 三条指令的信息几乎一样；
2. **容易不一致**：改了汇编写法忘了改匹配模式，编译能过但行为是错的；
3. **无法批量生成**：指令选择器、汇编输出器、寄存器信息……这些代码都得跟着改。

**TableGen 就是为了解决这三件事**：它是一门**描述性语言**（.td 文件）加一个
**代码生成器**（`llvm-tblgen`），把"指令描述"编译成 C++ 代码。

## 二、TableGen 是什么：一门"给编译器用的编译器"

用一句话概括：

> **TableGen 是一种"以类和继承为核心"的声明式语言，专门用来写"表"，
> 然后由一个生成器把这些表变成 C++ 代码或数据表。**

它的语法看着像这样（我们待会儿看真实的）：

```
class 某类指令<参数...> { ... }        // 定义模板
def 某条指令 : 某个类<参数>;            // 实例化
multiclass 某组指令<...> { ... }        // 定义"一族的生成规则"
defm 前缀 : 某组指令<...>;              // 批量实例化
```

**关键在最后两行**：`multiclass` + `defm` 是"批量生成"的机制。
`add.s16`/`add.s32`/`add.s64` 这种一组指令，就是一次 `defm` 生成出来的。

## 三、先看寄存器：NVPTX 的 `.td` 怎么写

寄存器定义在 `NVPTXRegisterInfo.td`。先是单个寄存器：

```tablegen
// PTX 的寄存器是"虚拟"的，所以名字长这样：%r0..%r4、%rd0..%rd4 ...
foreach i = 0...31 in {
  def ENVREG#i : NVPTXReg<"%envreg"#i>;
}
```

然后是**寄存器类**（register class）：

```tablegen
def B1  : NVPTXRegClass<[i1],  8,  (add (sequence "P%u", 0, 4))>;
def B16 : NVPTXRegClass<[i16, f16, bf16], 16, (add (sequence "RS%u", 0, 4))>;
def B32 : NVPTXRegClass<[i32, v2f16, v2bf16, v2i16, v4i8, f32], 32,
                              (add (sequence "R%u", 0, 4),
                              VRFrame32, VRFrameLocal32)>;
def B64 : NVPTXRegClass<[i64, v2i32, v2f32, f64], 64,
                        (add (sequence "RL%u", 0, 4),
                         VRFrame64, VRFrameLocal64)>;
def B128 : NVPTXRegClass<[i128], 128, (add (sequence "RQ%u", 0, 4))>;
```

逐段读 `B32` 那一行：

| 部分 | 含义 |
| --- | --- |
| `B32` | 类的名字（生成的枚举里会变成 `NVPTX::B32`） |
| `[i32, v2f16, v2bf16, v2i16, v4i8, f32]` | **这个寄存器类能承载哪些类型** |
| `32` | 位宽 |
| `(sequence "R%u", 0, 4)` | 包含 `R0..R4` 这些寄存器 |
| `VRFrame32, VRFrameLocal32` | 还包含帧指针相关寄存器 |

**这一行就解释了第 02-6 讲那个谜题**：为什么 `=f`（浮点输出）和 `r`（整型输入）
在 MIR 里都显示成 `regdef:B32` / `reguse:B32`？

看类型列表：`[i32, ..., f32]`——**`f32` 和 `i32` 在同一个寄存器类里**。
因为 PTX 的虚拟寄存器就是"按宽度分类"的，32 位的整数和浮点数共用一批寄存器。

这就是"**TableGen 里的一个类型列表，决定了 MIR 里看到的寄存器类别**"。

## 四、再看指令：一条指令的定义

指令的基类在 `NVPTXInstrFormats.td`：

```tablegen
class NVPTXInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
  : Instruction {
  field bits<14> Inst;

  let Namespace = "NVPTX";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let UseNamedOperandTable = true;
  ...
```

这四行基本就是一条指令描述的全部要素：

```
OutOperandList   输出操作数
InOperandList    输入操作数
AsmString        汇编怎么写
Pattern          什么样的 IR/DAG 形状可以选成它
```

### 具体例子：`LD_GLOBAL_NC_i32`

`NVPTXIntrinsics.td:3439` 开始：

```tablegen
// Don't annotate ld.global.nc as mayLoad, because these loads go through the
// non-coherent texture cache, and therefore the values read must be read-only
// during the lifetime of the kernel.
class LDG_G<NVPTXRegClass regclass>
  : NVPTXInst<(outs regclass:$result),
              (ins AtomicCode:$Sign, i32imm:$fromWidth,
                   UsedBytesMask:$usedBytes, ADDR:$src,
                   EvictionAndPrefetchHint:$evictionAndPrefetchHint,
                   CachePolicy:$policy),
               !strconcat("${usedBytes}", "ld.global.nc", CacheHintQualifiers,
                          ".${Sign:sign}$fromWidth \t$result, [$src]${policy};")>;

def LD_GLOBAL_NC_i16 : LDG_G<B16>;
def LD_GLOBAL_NC_i32 : LDG_G<B32>;
def LD_GLOBAL_NC_i64 : LDG_G<B64>;
```

三个值得注意的地方：

1. **`(outs regclass:$result)`**：输出操作数的类型是"参数传进来的那个寄存器类"。
   `LDG_G<B32>` 和 `LDG_G<B16>` 就是靠这个参数区分开——**一条模板，三条指令**。
2. **`AsmString` 是拼出来的**：`!strconcat` 把 `usedBytes`（前缀）+
   `"ld.global.nc"` + 缓存修饰 + `".${Sign:sign}$fromWidth"` 拼成最终的写法。
   `${Sign:sign}` 是"取 Sign 操作数的 sign 字段"，最后会变成 `u32` 或 `s32` 这样的后缀。
3. **类名 `LDG_G` 的注释说明了一个重要设计**：它特意**不**标 `mayLoad`，
   因为 `ld.global.nc` 走的是非一致缓存，要求数据在内核生命周期内只读。
   一条指令的"内存属性"会影响优化器的重排决策——**这也是编译器设计的一部分**。

## 五、生成物长什么样（这是最有说服力的部分）

我们真的有编译好的 LLVM，所以可以直接看生成物。它们在这里：

```bash
ls -l /root/llvm-build/lib/Target/NVPTX/*.inc
```

### 生成物 1：指令编号枚举

```bash
grep -n "LD_GLOBAL_NC_i32" /root/llvm-build/lib/Target/NVPTX/NVPTXGenInstrInfo.inc | head -3
```

```
    LD_GLOBAL_NC_i32                       = 1523, // NVPTXIntrinsics.td:3449
    { 1523, 8, 1, 0, 0, 0, 0, NVPTXOpInfoBase + 2368, 0,
      0|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::ExtraSrcRegAllocReq)|...
      , 0x0ULL },  // LD_GLOBAL_NC_i32
```

**注意第一行末尾那个注释：`// NVPTXIntrinsics.td:3449`**——
生成器把"这条指令定义在哪个文件的哪一行"也写进去了！

这条注释有两个好处：

1. 你在调试时看到某个 opcode，能立刻知道去 `.td` 的哪儿看；
2. 它反过来证明了一件事：**MIR 里那些 opcode 名字，全部来自 `.td` 的 `def` 名字**。
   我们第 06-3 讲读 MIR 时看到的 `LD_GLOBAL_NC_i32`，就是这一行的 `def` 名字。

### 生成物 2：指令信息表

第二行那一长串就是这个 opcode 的元信息：

```
{ 1523, 8, 1, 0, 0, 0, 0, NVPTXOpInfoBase + 2368, 0,
  0|(1ULL<<MCID::UnmodeledSideEffects)|(1ULL<<MCID::ExtraSrcRegAllocReq)|... }
```

`MCID::*` 是一堆标志位：有没有副作用、寄存器分配要不要特殊处理……
**这些标志直接影响后端 pass 的行为**（比如"有副作用"的指令不能被随便删掉）。

`ExtraSrcRegAllocReq`/`ExtraDefRegAllocReq` 这两个标志很有意思：它表示
**这条指令对源/目的寄存器有额外约束**（比如必须落在同一个寄存器上）。
这类约束会让寄存器分配器更小心。

### 生成物 3：汇编输出器

```bash
grep -n 'ld.global.nc' /root/llvm-build/lib/Target/NVPTX/NVPTXGenAsmWriter.inc | head -3
```

```
    O << "ld.global.nc";
```

**AsmString 被拆成了"逐段拼接的 C++ 语句"**：常量部分直接 `O << "..."`，
操作数部分变成 `printOperand(MI, N, STI, O)`。

这就是为什么你改 `.td` 里的汇编写法，PTX 输出会跟着变——**它是生成出来的。**

### 生成物 4（预告）：指令选择匹配器

还有个更大的文件：

```bash
ls -l /root/llvm-build/lib/Target/NVPTX/NVPTXGenDAGISel.inc
```

```
3247177 /root/llvm-build/lib/Target/NVPTX/NVPTXGenDAGISel.inc     ← 3.2 MB
```

**3.2 MB 的"指令选择器"**，全部由 `.td` 里的 pattern 生成。
下一讲我们专门讲它。

## 六、自己跑一次 TableGen

你也可以手工调用生成器（我们编译好的 LLVM 里有这个工具）：

```bash
ls /root/llvm-build/bin/ | grep tblgen
# clang-tblgen  llvm-min-tblgen  llvm-tblgen

/root/llvm-build/bin/llvm-tblgen \
    -I /root/llvm-project/llvm/lib/Target/NVPTX \
    -I /root/llvm-project/llvm/include \
    --gen-instr-info /root/llvm-project/llvm/lib/Target/NVPTX/NVPTX.td \
    | head -40
```

你会看到我们前面贴的那些内容被打印出来——**这就是表驱动的代码生成在眼前发生**。

（`--gen-instr-info` 只是其中一种；还有 `--gen-dag-isel`、`--gen-asm-writer`、
`--gen-register-info` 等等，每个对应一个生成物。）

## 七、小结

1. TableGen 是"**描述性语言 + 代码生成器**"，用来解决"指令描述爆炸"的问题：
   写一次模板，批量生成指令、选择器、汇编输出器、寄存器信息。
2. 寄存器类里的**类型列表**决定了 MIR 里的寄存器类别——
   `B32` 同时包含 `i32` 和 `f32`，这就是"`=f` 和 `r` 都是 `B32`"的根源。
3. 生成物里会**标注源码位置**（`// NVPTXIntrinsics.td:3449`），
   这让我们能顺着 MIR 的 opcode 名字一路找回 `.td` 的定义。

## 八、动手题

1. 在生成文件里找出你自己感兴趣的几条指令：

   ```bash
   grep -nE "^\s+(ADD32rr|SHL32_ri|LD_GLOBAL_NC_i32|LOAD_CONST32)\s+=" \
     /root/llvm-build/lib/Target/NVPTX/NVPTXGenInstrInfo.inc | head
   ```

   每一条后面的注释都会指向 `.td` 的行号，去那一行看看它的定义。
2. 想一想：为什么寄存器类要写"能承载哪些类型"？
   （提示：寄存器分配器需要知道"这个虚拟寄存器能放进哪个类"，
   指令选择的合法性检查也要靠它。）

下一讲我们看 TableGen 的另一半：pattern、谓词和 Feature——
也就是"**这条指令什么时候能被选中**"。
