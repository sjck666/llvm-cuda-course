# 02-6 · 内联汇编的三层信息，以及 `convergent` 从哪儿来

这一讲是第 2 部分的核心。我们要把这一行看透：

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
```

它在 IR 里变成一条 asm 调用。我们要回答三个问题：

1. 那 14 个占位符和约束字符串是怎么变成"寄存器类别"的？
2. 为什么它身上带着 `convergent`？这个标记到底是谁加的？
3. 想把 `convergent` 摘掉，有哪几条路，哪条真的有效？

## 一、先看它变成了什么

`dumps/01-device-O0.ll` 第 297 行（我把长行折了一下）：

```llvm
%225 = call contract { float, float, float, float } asm sideeffect
       "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
       "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A",
       "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
       (i32 %199, i32 %202, i32 %205, i32 %208,
        i32 %210, i32 %213,
        float %215, float %218, float %221, float %224) #7, !srcloc !6
```

先别急着往下读，我建议你对照源码看三秒钟：

- 汇编模板里的 `%0..%13` 在 IR 里变成了 `$0..$13`（LLVM 用 `$` 表示操作数占位符）；
- 输出那 4 个 `"=f"(d[0..3])` 对应前 4 个操作数，但**它们没有出现在参数列表里**
  （因为它们不是输入）；
- 输入 10 个（`a[0..3]`、`b[0..1]`、`c[0..3]`）原样出现在参数列表里。

## 二、拆第一层：返回类型是"四个 float 的结构体"

```llvm
call contract { float, float, float, float } asm ...
```

一条 `mma` 有 4 个输出寄存器，IR 用**匿名结构体** `{ float, float, float, float }`
把它们打包成一个返回值。所以后面紧接着四条 `extractvalue`：

```llvm
%226 = extractvalue { float, float, float, float } %225, 0
%227 = extractvalue { float, float, float, float } %225, 1
%228 = extractvalue { float, float, float, float } %225, 2
%229 = extractvalue { float, float, float, float } %225, 3
```

这是"内联汇编多返回值"的通用表示法，和 CUDA 无关。**任何**多条输出的内联汇编
在 LLVM IR 里都是"一个聚合返回值 + 若干 extractvalue"。

小知识：那个 `contract` 标记是浮点收缩（FMA）相关的，和 `convergent` 没关系。
它是新版 IR 里新增的写法（老版本这里没有），读到的时候知道它不影响我们要讲的东西就行。

## 三、拆第二层：`asm sideeffect` 是 `volatile` 换来的

```llvm
asm sideeffect "..."
```

`sideeffect` 对应源码里的 `volatile`。它告诉 LLVM：

> 这段汇编有副作用。**不许因为"结果没人用"就把它删掉，也不许随便复制。**

如果你把源码里的 `asm volatile` 改成 `asm`（去掉 volatile），这里就不会有 `sideeffect`，
编译器就可能在"结果没被使用"时把整条指令优化掉。

对我们的 `mma` 来说，去掉 volatile 是危险的：它的结果当然会被用到，
但更重要的是它**有隐式的 warp 级副作用**（影响 warp 的执行状态）。
所以源码里一直带着 `volatile`。

## 四、拆第三层：约束字符串和 14 个寄存器

```llvm
"=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
```

这个字符串和源码里那 14 个占位符一一对应：

| 约束 | 个数 | 含义 | 对应源码 |
| --- | --- | --- | --- |
| `=f` | 4 | 输出，浮点寄存器 | `d[0..3]`（D 的输出） |
| `r` | 4 | 输入，32 位整型寄存器 | `a[0..3]`（A，装的是 2 个 f16） |
| `r` | 2 | 输入，32 位整型寄存器 | `b[0..1]`（B） |
| `f` | 4 | 输入，浮点寄存器 | `c[0..3]`（累加器输入） |

合计 4 + 4 + 2 + 4 = **14**，和模板里的 `{%0..%13}` 对得上。

不过这里有个"预告"：到了后端的机器层，你会发现这些约束被翻译成了**寄存器类别 ID**，
而且 `=f` 和 `r` 居然落到了**同一个类别**上：

```
regdef:B32, def %78, ... reguse:B32, %82, ...
```

因为 NVPTX 的虚拟寄存器是**按宽度分类**的，32 位的浮点和 32 位的整数用同一批寄存器。
第 6 部分会详细讲这件事。

## 五、拆第四层：`#7 = { convergent nounwind }`——这一节是重点

先看这个编号对应的属性组（`dumps/01-device-O0.ll` 的文件末尾）：

```llvm
attributes #7 = { convergent nounwind }
```

`nounwind` 好理解（CUDA 设备代码不抛异常）。**`convergent` 就是这一讲的"戏肉"。**

### 问题：一条纯计算的 `mma`，凭什么被标成"收敛操作"？

去 clang 源码里找原因。`clang/lib/CodeGen/CGStmt.cpp:3328`：

```cpp
if (!NoConvergent && getLangOpts().assumeFunctionsAreConvergent())
  // Conservatively, mark all inline asm blocks in CUDA or OpenCL as
  // convergent (meaning, they may call an intrinsically convergent op, such
  // as bar.sync, and so can't have certain optimizations applied around
  // them) unless it's explicitly marked 'noconvergent'.
  Result.addFnAttr(llvm::Attribute::Convergent);
```

翻译成人话：**"保守起见，CUDA/OpenCL 里所有内联汇编块都标成 convergent。"**

那 `assumeFunctionsAreConvergent()` 什么时候为真？追到
`clang/lib/Frontend/CompilerInvocation.cpp:4290`：

```cpp
bool HasConvergentOperations = Opts.isTargetDevice() || Opts.OpenCL ||
                               Opts.HLSL || T.isAMDGPU() || T.isNVPTX();
Opts.ConvergentFunctions =
    Args.hasFlag(OPT_fconvergent_functions, OPT_fno_convergent_functions,
                 HasConvergentOperations);
```

编译 CUDA 设备端时 `isTargetDevice()` 为真 → `ConvergentFunctions` 默认打开
→ 你写的**每一条** `asm` 都被标 convergent。

### 为什么 clang 要这么保守

因为 clang **不可能知道**你的汇编里有没有 `bar.sync`、`shfl`、`mma` 这类
"必须整个 warp 一起执行"的指令。它看到的就是一段字符串。

所以它的策略是：**一律按最坏情况处理**。

代价是明确的：

> **被标 convergent 的指令不能被复制、不能被随意移出/移入控制流。**
> 于是某些优化会因此收手——哪怕你写的其实只是 `mov.b32` 这种纯数据搬运。

### 这个标记会一路传下去，影响后端

先剧透一下它造成的后果（第 4、6 部分展开）：

```
IR:      attributes #7 = { convergent nounwind }
MIR:     INLINEASM &"mma.sync...", sideeffect isconvergent attdialect, ...
影响:    MachineSink / MachineLICM / MachineCSE / IfConversion / TailDuplicator
         见到它就收手
```

也就是说：**一条 `mma` 进来，等于给一整族机器级优化按了暂停键。**

## 六、想把 `convergent` 摘掉：四条路的实测结果

如果你很确定自己的汇编没有收敛语义（比如只有 `mov.b32` 打包），
直觉上会想去掉这个标记。我们把四条路都试了一遍
（脚本在 `tools/convergent-experiment.sh` 的 G 段，你可以自己跑）：

| 尝试 | 结果 |
| --- | --- |
| `-fno-convergent-functions` | **有效**：驱动把它传给了设备端 `cc1`，IR 里所有属性组都不再含 `convergent` |
| `__attribute__((noconvergent))` 放在 asm 前 | **语法错误**：`error: an attribute list cannot appear here` |
| `[[clang::noconvergent]]` 直接放在 asm 前 | 能编译，但警告 `'clang::noconvergent' attribute ignored [-Wignored-attributes]`，**没生效** |
| `[[clang::noconvergent]] { asm ...; }` 包一层语句块 | **有效**：属性挂到 `AttributedStmt` 上，asm 调用点的属性组变成 `{ nounwind }` |

四种做法的实测输出（原样抄自脚本运行结果）：

```
  with_convergent          属性组 #3  含 convergent: 是
  with_noconvergent        属性组 #3  含 convergent: 是     <- 直接挂属性，无效
  with_noconvergent_stmt   属性组 #4  含 convergent: 否     <- 挂语句块，有效
```

### 为什么"直接挂"不行，"包一层"就行

规矩其实很简单：**属性必须挂在"语句"上，不能直接挂在 `asm` 语法元素上。**

`CGStmt.cpp` 里处理 `NoConvergent` 的那段代码（第 809 行附近）读的是
`InNoConvergentAttributedStmt`——一个只在"带属性的语句"（`AttributedStmt`）里
才会被设置的状态。你写 `[[clang::noconvergent]]` 直接贴在 `asm` 前面时，
clang 根本没机会把它包成 `AttributedStmt`，于是属性被忽略（这就是那条警告的含义）。

而 `__attribute__((noconvergent))` 那种写法干脆连语法都不成立，
因为这里不是"能挂属性"的位置。

### 那到底该不该摘

我的建议是：**默认不要摘**。理由：

- `mma`、`ldmatrix`、`cp.async`、`bar.sync` 这些指令**本来就是收敛的或半收敛的**，
  标 `convergent` 是正确的；
- 只有当你写的汇编确实是纯数据操作（比如我们 v1 里那条 `mov.b32` 打包），
  才值得考虑用 `[[clang::noconvergent]] { ... }` 单独放行那一条；
- 全局的 `-fno-convergent-functions` 更要小心：它会把 `bar.sync` 之类的保护一起关掉。

**请把 `convergent` 当成"编译器给你的安全气囊"，而不是"性能敌人"。**
第 4 部分我们会看到它究竟挡住了哪些具体优化，到时候你再决定要不要动它。

## 七、拆第五层：`!srcloc !6`

```llvm
#7, !srcloc !6
```

`!srcloc` 指向"这段代码在源文件里的位置"，调试和 `-Rpass` 之类的诊断会用到它。
对代码生成没有影响，知道它在那儿就行。

## 八、其余几条汇编的 IR（对照用）

同一份 kernel 里还有三条内联汇编，它们的形式完全一样：

```llvm
; cp.async 16 字节
call void asm sideeffect "cp.async.cg.shared.global [$0], [$1], 16;\0A",
     "r,l"(i32 %101, ptr %102) #7, !srcloc !9

; ldmatrix.x4
%166 = call { i32, i32, i32, i32 } asm sideeffect
       "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];\0A",
       "=r,=r,=r,=r,r"(i32 %165) #7, !srcloc !13

; ldmatrix.x2.trans
%185 = call { i32, i32 } asm sideeffect
       "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0,$1}, [$2];\0A",
       "=r,=r,r"(i32 %184) #7, !srcloc !14
```

注意两个新约束：

- `l` = 64 位整型（long），用在 `cp.async` 的**全局地址**上；
- `r` = 32 位整型，用在**共享内存那一侧**（因为我们已经把共享地址转成 32 位偏移了）。

**"两边宽度不一样"是手写 `cp.async` 最容易写错的点**，第 7 部分会专门讲。

## 九、小结

1. 一条内联汇编在 IR 里由五部分信息组成：**返回类型**（多输出用聚合）、
   **`sideeffect`**（来自 volatile）、**约束字符串**（14 个约束 ↔ 14 个占位符）、
   **属性组**（含 `convergent`）、**`!srcloc`**。
2. `convergent` 不是 clang 偷懒，而是它**无法判断你的汇编里有没有 warp 级同步语义**，
   于是保守地全标上（`CGStmt.cpp:3328` + `CompilerInvocation.cpp:4290`）。
3. 想摘掉它，**只有挂在语句上的 `[[clang::noconvergent]] { ... }` 和全局的
   `-fno-convergent-functions` 有效**；直接贴在 asm 前会被忽略，`__attribute__` 写法直接报错。

## 十、动手题

1. 跑一遍三段验证，确认 `convergent` 的传递路径：

   ```bash
   grep -n "convergent" dumps/01-device-O0.ll | head -3        # IR 层：属性组
   grep -n "isconvergent" dumps/26-isel.mir | head -3           # 机器层：标志位
   ```

   然后想一想：为什么 `mov.b32` 那条打包汇编**也**带 convergent？
   （提示：它也是 `asm volatile`，而 clang 的规则是"一视同仁"。）
2. 把这行改一改再编一次，观察属性组的变化：

   ```cpp
   [[clang::noconvergent]] {
     asm volatile("mov.b32 %0, {%1,%2};" : "=r"(x) : "h"(lo), "h"(hi));
   }
   ```

   （可以直接改 `code/noconvergent_demo.cu`，它里面已经有三个对照 kernel 了。）

下一讲我们看 `__half` 带来的"胶水代码"，以及 `__cvta_generic_to_shared` 的真身。
