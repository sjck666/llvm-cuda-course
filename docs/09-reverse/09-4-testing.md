# 09-4 · 测试与三段验证

第 9 部分最后一讲。我们讲"怎么证明你新加的指令是对的"。

这件事比看起来重要：**编译器 bug 的特点是"大部分时候对，偶尔错"**。
没有测试，你根本不知道自己有没有写对。

## 一、LLVM 的测试长什么样

LLVM 的测试基本是"**输入文件 + FileCheck 断言**"这种形式：

```llvm
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx81 | FileCheck %s
...
; CHECK: mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
```

读法：

```
; RUN: 这一行告诉测试框架"怎么跑"
; CHECK: 这一行告诉它"输出里必须出现什么"
```

**关键认识：它检查的是"编译器输出的文本"**，而不是"程序的运行结果"。
因为对编译器来说，"生成什么指令"才是它的输出。

### 一个真实的收敛性测试

LLVM 里有一个专门测收敛语义的用例，你加新 mma 指令时应该照着写一份：

`llvm/test/CodeGen/NVPTX/mma-no-sink-after-laneid-check.ll`：

```llvm
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx81 | FileCheck %s

declare { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32, i32, i32, float, float, float, float) #1

; COM: llvm.nvvm.mma should not sink to the next block and gets reordered
; COM: to be after laneid check.
; CHECK-LABEL: no_reorder_mma_and_laneid_check
define dso_local void @no_reorder_mma_and_laneid_check(ptr %arg, ptr %arg1) {
bb:
  ; CHECK: mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
  ; CHECK: laneid
  %i = tail call { float, float, float, float } @llvm.nvvm.mma.m16n8k4.row.col.tf32(i32 10, ...)
  %i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  ...
}
```

**它在测什么？** 测"`mma` 不会被下沉到 `laneid` 检查之后"——
因为 `mma` 是 convergent 的，不能被移到依赖 `laneid` 的控制流里。

**这份测试的存在，正好给第 04-6 讲的结论盖了章。**

### 生成式测试

对于"形状 × 布局 × 类型"这种组合爆炸的指令，LLVM 用**生成器**：

```
llvm/test/CodeGen/NVPTX/wmma.py        用 Python 遍历所有组合，生成 .ll 和断言
llvm/test/CodeGen/NVPTX/wmma-ptx87-sm120a.py 等  按 PTX/SM 版本分组的生成脚本
```

模板（第 1109 行）：

```python
mma_intrinsic_template = "llvm.nvvm.mma${b1op}.${geom}.${alayout}.${blayout}${kind}${satf}.${intrinsic_signature}"
mma_instruction_template = "mma.sync${aligned}.${geom}.${alayout}.${blayout}${kind}${satf}.${ptx_signature}${b1op}"
```

写一条新指令时，**你只需要把它加进参数列表，生成器会帮你把测试补齐**。

## 二、三段验证：光有 PTX 测试还不够

LLVM 的测试只验证到"**PTX 层面的正确性**"。但要证明这条指令真的能用，你要走三段：

```
第一段：llc 出 PTX          —— 确认指令被正确选择、操作数顺序对
第二段：ptxas 出 SASS       —— 确认它真的能汇编成硬件指令（且有资源报告）
第三段：上机跑数值对拍       —— 确认语义真的对
```

### 第一段：确认指令出来了

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx87 -o - demo.ll \
  | grep -E '^\s+(ldmatrix|mma)\.sync'
```

**判据是"指令本身"，不是"有没有报错"**（第 09-3 讲那个静默降级的坑）。

### 第二段：确认能汇编、看资源

```bash
ptxas -arch=sm_89 -v demo.ptx -o demo.cubin
cuobjdump -sass demo.cubin | grep -E "HMMA|LDSM|LDGSTS"
```

这一步能抓到"PTX 写得对但 ptxas 不认"的问题（比如谓词用错、PTX 版本不够）。

### 第三段：上机对拍（最硬的一关）

这就是我们这门课一直在用的那个裁判：

```
[v1 manual  ] max |D - ref| = 4.29153e-06   -> PASS
[v2 ldmatrix] max |D - ref| = 4.29153e-06   -> PASS
```

**布局错了，误差会从 `1e-6` 直接炸到个位数。** 这就是为什么第三段不能省：

```
PTX 里指令出来了（第一段通过）  ≠  语义对
SASS 能汇编（第二段通过）        ≠  语义对
只有上机跑数值，才知道语义对不对
```

## 三、把三段验证脚本化

我们这门课的工具脚本其实就是这三段的实现：

| 脚本 | 对应哪一段 |
| --- | --- |
| `tools/dump-all.sh` | 第一、二段（生成全部中间产物） |
| `tools/analyze-sass.sh` | 第二段（统计指令分布） |
| `tools/build-and-run.sh` | 第三段（上机对拍） |
| `tools/ch10-experiments.sh` | 指令选择的"名字/签名"探测 |

**如果你要给新指令写验证，照这套抄一遍就行。**

## 四、一个补充：怎么验证"边界情况"

真实指令往往有边界情况，测试要专门覆盖：

```
不支持的类型组合 → 应该有明确报错，而不是静默生成错误指令
不支持的地址空间   → 同上
PTX 版本不够      → 应该选择别的实现，或者报错
SM 版本不够      → 同上（Requires<[PTX80, SM90]> 就是干这个的，第 05-4 讲）
```

第 09-2 节 checklist 里第 5 条"边界"说的就是这个：

```
□ 5. 边界: 不支持的类型/地址空间要有明确报错, 别静默生成错误指令
```

**"静默生成错误指令"是编译器里最糟糕的一类 bug**——因为用户看到的是"能编译、结果错"，
很难定位。

## 五、第 9 部分总结

```
[09-1] 三条路线（inline asm / intrinsic / 新增 intrinsic）+ 规格化六件事
[09-2] 六步接入法：intrinsic → 指令+pattern → 内存行为 → builtin → 测试 → 三段验证
[09-3] 名字与签名陷阱：两组报错、IR 名不带空间后缀、老版本的静默降级
[09-4] 测试（FileCheck + 生成式 + 收敛专项）与三段验证
```

## 六、全书收尾

九部分走下来，这条链路的每一站你都停过了：

```
.cu ──clang──► IR ──opt──► 优化后 IR ──llc──► PTX ──ptxas──► SASS
  ▲     ▲        ▲      ▲        ▲          ▲        ▲         ▲
  │     │        │      │        │          │        │         │
  │  第2部分   第3部分 第4部分  第5、6部分  第7部分  第9部分   第8部分
  │  (前端)    (IR)   (中端)   (后端设计)  (PTX)   (实战)   (SASS)
  └─ 第1部分：案例与 fragment（地基）
```

如果你只能记住三句话，我希望是这三句：

1. **每一层都在"降级"**：语言的语义 → 通用的 IR → 目标相关的机器指令 →
   物理的寄存器与机器码。每一层都会丢掉一些信息、增加一些约束。
2. **属性不是注释**：`__restrict__` 决定走哪条缓存路径，
   `asm volatile` 带出 `convergent` 并让一整族优化收手，
   `__global__` 决定函数是 `.entry` 还是 `.func`。
3. **不同的事情在不同层做**：LLVM 决定语义与结构，
   ptxas 决定寄存器与调度；trip count 已知时 IR 层展开、未知时 SASS 层展开。
   **知道"这件事该在哪一层解决"，就是这门课真正想教给你的东西。**

## 七、动手题

1. 用三段验证跑一遍你自己的某个 CUDA kernel（哪怕是"hello world"级别的），
   确认你能在 PTX 里找到对应的指令、能读出 SASS、能跑出正确结果。
2. 挑战题：给 `code/nvvm_intrinsic_demo.ll` 加一条新的 intrinsic 调用
   （比如 `llvm.nvvm.stmatrix`），先查 `.td` 确认名字和签名，再验证 PTX 里
   真的出现了 `stmatrix`。**这就是第 09-3 节那三组对照实验的正向用法。**
