# 04-2 · SROA + mem2reg：数组是怎么消失的

第 03-5 讲我们看到：O0 的 IR 里到处都是 `alloca`，O2 里一条都没有。
这一讲我们**亲手把这个过程跑一遍**。

## 一、先踩一个坑：`opt` 为什么不干活

你可能会想："直接把 O0 的 IR 丢给 `opt` 不就行了？"

```bash
opt -passes='sroa,mem2reg' -S dumps/01-device-O0.ll -o /tmp/x.ll
wc -l /tmp/x.ll
```

你会发现**几乎什么都没变**。原因不是 pass 失效，而是：

> **`-O0` 的 clang 会给每个函数打上 `optnone` 属性。**

```llvm
attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone
                  "frame-pointer"="all" ... }
                                                   ^^^^^^^
```

**凡是带 `optnone` 的函数，绝大多数 pass 会直接跳过。**
这是 LLVM 的硬规矩：`-O0` 的语义就是"不要优化"。

所以做实验时要先去掉它。标准做法：

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda-12.8 --cuda-gpu-arch=sm_89 \
        --cuda-device-only -O0 -Xclang -disable-O0-optnone \
        -S -emit-llvm -o dumps/01b-device-O0-nooptnone.ll code/tc_mma.cu
```

`-Xclang -disable-O0-optnone` 就是关掉它的开关。
你在 LLVM 自己的测试用例里到处能看到这个组合。

## 二、跑实验：796 → 353

```bash
opt -passes='sroa,mem2reg'             -S dumps/01b-device-O0-nooptnone.ll \
    -o dumps/15-after-sroa-mem2reg.ll
opt -passes='sroa,mem2reg,instcombine' -S dumps/01b-device-O0-nooptnone.ll \
    -o dumps/16-after-instcombine.ll
```

四份文件的行数：

```
01-device-O0.ll                  796 行      <- clang -O0 原样（带 optnone）
01b-device-O0-nooptnone.ll       796 行      <- 去掉 optnone，给 opt 用
15-after-sroa-mem2reg.ll         353 行      <- 少了一半多
16-after-instcombine.ll          338 行
02-device-O2.ll                  291 行      <- clang -O2 的最终形态
```

**796 → 353 这一步，就是 SROA 和 mem2reg 的功劳。**

## 三、看具体变化

### 优化前（`01b`）：数组是真的数组

```llvm
  %7  = alloca %struct.__half2, align 4
  %10 = alloca %struct.__half2, align 4
  ...
  %222 = load ptr, ptr %10, align 8
  %223 = getelementptr inbounds float, ptr %222, i64 3
  %224 = load float, ptr %223, align 4
  %225 = call contract { float, float, float, float } asm sideeffect "mma.sync..."(...) #7, !srcloc !6
```

注意那三条：**先 `load` 出指针，再 `getelementptr` 算偏移，再 `load` 出值**。
取一个数组元素要三步。这就是"内存世界"。

### 优化后（`15`）：输入直接是 SSA 值

```llvm
  %80 = call contract { float, float, float, float } asm sideeffect "mma.sync..."(
         i32 %46, i32 %49, i32 %52, i32 %55, i32 %75, i32 %79,
         float %.sroa.097.0, float %.sroa.499.0,
         float %.sroa.7.0,   float %.sroa.10.0) #7, !srcloc !6
```

**`alloca` + `load`/`store` 那一整圈消失了**：fragment 的输入直接以 SSA 值的形式
进 asm 调用。

### 注意那些 `.sroa.` 名字

```llvm
float %.sroa.097.0, float %.sroa.499.0, float %.sroa.7.0, float %.sroa.10.0
```

这是 **SROA 拆数组留下的痕迹**。SROA 把 `float c[4]` 拆成了 4 个独立标量，
但一时还没法给它们起好名字，就先叫 `.sroa.<编号>.<下标>`。
再过几个 pass，这些名字会被丢掉。

**读 IR 时看到 `.sroa.` / `.mem2reg` 这类名字，你就能判断"这里刚刚发生过什么变换"。**

## 四、两个 pass 各自在干什么

这两个 pass 经常一起出现，但分工不同：

| pass | 全名 | 干什么 |
| --- | --- | --- |
| **SROA** | Scalar Replacement of Aggregates | 把**聚合类型**（数组、结构体）拆成独立的标量 |
| **mem2reg** | Memory to Register | 把"只在函数内用、不被取地址"的**局部内存**提升成 SSA 值 |

顺序通常是先 SROA 再 mem2reg：先把聚合拆开（`c[4]` → 四个标量），
再把标量从内存里"提"到寄存器（`alloca` + load/store → SSA 值 + phi）。

### 为什么"常量下标"是关键

`opt -passes='sroa,mem2reg'` 之所以能把我们的数组消灭掉，是因为
**我们的数组下标全是编译期常量**（`a[0]`、`a[1]`、`a[2]`、`a[3]`）。

如果下标是运行期变量（比如 `a[i]`），SROA 就没法拆——它不知道要拆成几份、
也不知道谁对应谁。这时数组只能"老实待在内存里"，后续要么靠 GVN/DSE 部分优化，
要么就真的落到 local memory（SASS 里会出现 `LDL`/`STL`，还要占 local memory 带宽）。

**这就是 Tensor Core 编程里那条硬性经验的来源：**

> **不要让 fragment 数组被动态下标访问。**

Intel/AMD/NVIDIA 的手册都会提这一条，原因就在这里——不是风格问题，
而是"能不能被拆开、能不能进寄存器"的问题。

## 五、小结

1. 做中端实验必须先去 `optnone`（`-Xclang -disable-O0-optnone`），否则 `opt` 什么都不干。
2. **SROA 拆聚合、mem2reg 提内存**，两个一起把 `float c[4]` 变成 4 个 phi：
   **796 → 353 行**。
3. 这一切的前提是"**下标是常量**"。fragment 数组一旦被动态下标访问，
   就会掉回内存世界——这是 Tensor Core 编程里最重要的一条实践建议。

## 六、动手题

1. 看 `.sroa.` 的名字在两个阶段的残留：

   ```bash
   grep -c '\.sroa\.' dumps/15-after-sroa-mem2reg.ll   # SROA 刚跑完
   grep -c '\.sroa\.' dumps/02-device-O2.ll            # 完整 O2 之后
   ```
2. 把 kernel 里的 `a[0]`、`a[1]`、`a[2]`、`a[3]` 改成 `a[k & 3]`（动态下标），
   重新生成 O2 IR，看看 `alloca` 是不是回来了。

下一讲我们看 InstCombine——那个天天在做"小账本"的 pass。
