# 03-4 · 循环体逐行精读

这一讲我们做一件"笨功夫"：把 `mma_tc_manual` 的循环从头到尾读一遍。

为什么值得？因为读完之后，你看任何 IR 都不会再慌——你会知道每条指令
"为什么必须在这儿"，以及它对应源码的哪一行。

我们会按源码的执行顺序读，一共五段：

```
1. 循环前的准备      （算地址）
2. 进循环的条件分支   （K > 0 ?）
3. 循环头：五个 phi
4. 循环体：A 的装载 → B 的装载 → mma → 累加器回转
5. 写回块
```

## 一、循环前的准备（第 13–31 行）

```llvm
  %7  = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8  = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %9  = shl nuw nsw i32 %8, 4               ; tile_m = blockIdx.y * 16
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %11 = shl i32 %10, 3                      ; tile_n = blockIdx.x * 8
  %12 = lshr i32 %7, 2                      ; gid = lane >> 2
  %13 = add nuw nsw i32 %12, 8              ; gid + 8
  %14 = shl nuw nsw i32 %7, 1               ; lane * 2
  %15 = and i32 %14, 6                      ; (lane * 2) & 6  == tig * 2
  %16 = or disjoint i32 %15, 8              ; tig * 2 + 8
  %17 = icmp sgt i32 %5, 0                  ; K > 0 ?
  %18 = add nuw nsw i32 %9, %12             ; tile_m + gid
  br i1 %17, label %23, label %19
```

逐行对照第 01-2、01-3 讲的 fragment 坐标表，你会发现**每一行都在算一个下标**：

```
gid      = %12        tig*2 = %15
gid + 8  = %13        tig*2 + 8 = %16
tile_m   = %9         tile_n = %11
```

三处值得停下：

**第一，`lshr` 出现在本该是 `ashr` 的地方。**

源码里 `lane` 是 `int`，`lane >> 2` 按 C 语义是算术右移（`ashr`）。
但优化器能证明 `threadIdx.x` 非负（它的取值范围是已知的 0..1023），
于是把它降级成了**逻辑**右移 `lshr`——语义更弱、后端更好优化。

**第二，`%15 = and i32 %14, 6` 是 `(lane & 3) * 2` 的变形。**

你写的是 `tig * 2`，其中 `tig = lane & 3`。InstCombine 看到 `(lane & 3) * 2`，
把它改写成了 `(lane * 2) & 6`：乘 2 等于左移一位，低位一定是 0，
所以掩码可以从 `& 3` 提到移位之后变成 `& 6`。这类"少一条 LOP3"的小账，
就是 InstCombine 的日常（第 4 部分细讲）。

**第三，`or disjoint` 是个"没有进位的加法"提示。**

`%16 = or disjoint i32 %15, 8` 代替了 `add`。`disjoint` 告诉后端：
这两个数的置位区间不重叠，所以 `or` 不会产生进位——后端可以放心把它当加法折进地址计算。
它是 InstCombine 推出来的。

## 二、分支：为什么要先问"K > 0"

```llvm
br i1 %17, label %23, label %19
```

`K > 0` 才进主循环，否则直接跳到写回块（`%19` → `%38`），把 4 个 0 存进 D。

这个结构是 **loop-rotate** 这类变换的结果：把"循环入口的判断"提出来，
主循环里就少一次检查。你写的是朴素的 `for (k0 = 0; k0 < K; k0 += 16)`，
编译器把它转成了"先判断一次，再进 do-while"的形状。

SASS 里对应的是：

```
/*0020*/  ISETP.LT.AND P0, PT, RZ, c[0x0][0x180], PT ;
/*00b0*/  @!P0 BRA 0xd80 ;
```

（`c[0x0][0x180]` 就是 K —— kernel 参数在常量内存里。第 8 部分会讲这个细节。）

## 三、地址准备块（`%23`，第 33–49 行）

```llvm
23:
  %24 = mul nuw nsw i32 %5, %18             ; K * (tile_m + gid)
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %25   ; &A[row][0]
  %27 = add nuw nsw i32 %13, %9
  %28 = mul nuw nsw i32 %5, %27
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds nuw [2 x i8], ptr %0, i64 %29   ; &A[row+8][0]
  %31 = zext nneg i32 %15 to i64
  %32 = zext nneg i32 %16 to i64
  %33 = sext i32 %11 to i64
  %34 = getelementptr [2 x i8], ptr %1, i64 %33                ; &B[0][tile_n]
  %35 = zext nneg i32 %12 to i64
  %36 = getelementptr [2 x i8], ptr %34, i64 %35               ; &B[0][tile_n + gid]
  %37 = sext i32 %4 to i64                                     ; N（跨行用）
```

**这一段的精髓是 GEP。**

`getelementptr inbounds nuw [2 x i8], ptr %0, i64 %25` 的意思是：
"以 `[2 x i8]`（也就是一个 `__half`）为单位，从 `%0` 往前走 `%25` 个元素"。
因为它知道一个 `__half` 是 2 字节，这一步隐式地包含了 `× 2`。

**在 IR 里你永远看不到显式的 `× 2`**——这就是"类型化指针"的代价和便利。

两个细节：

1. **GEP 的元素类型写的是 `[2 x i8]` 而不是 `%struct.__half`。**
   优化器会把元素类型"规格化"成它认为方便的形式，只要**大小**不变，语义就不变
   （`%struct.__half` = `{ i16 }` = 2 字节 = `[2 x i8]`）。
   **读 IR 时看尺寸，不要看名字。**
2. **`%34` / `%36` 没有 `inbounds`。**
   因为循环里会用它加上 `%37`（= N）去取 `B[k+1][n]`，编译器无法证明一定不越界，
   于是不敢加 `inbounds`（加了就等于承诺"绝不越界"，一旦越界就是 UB）。

## 四、循环头：五个 phi（第 76–81 行）

```llvm
58:                                    ; preds = %23, %58
  %59 = phi i32   [ 0, %23 ],         [ %96, %58 ]      ; k0
  %60 = phi float [ 0.000000e+00, %23 ], [ %92, %58 ]   ; c[0]
  %61 = phi float [ 0.000000e+00, %23 ], [ %93, %58 ]   ; c[1]
  %62 = phi float [ 0.000000e+00, %23 ], [ %94, %58 ]   ; c[2]
  %63 = phi float [ 0.000000e+00, %23 ], [ %95, %58 ]   ; c[3]
```

**五个 phi = 源码里五个"跨迭代存活"的量**：循环变量 `k0` 和累加器 `c[0..3]`。

这里能看出一个关键事实：**`float c[4]` 这个数组在 O2 之后彻底不存在了**，
它变成了 4 个 phi。第 03-5 讲我们会用 O0 对照，看它是怎么消失的。

## 五、循环体：A 的装载（第 82–92 行）

```llvm
  %64 = zext nneg i32 %59 to i64            ; k0
  %65 = getelementptr inbounds nuw [2 x i8], ptr %26, i64 %64   ; &A[row][k0]
  %66 = getelementptr inbounds nuw [2 x i8], ptr %30, i64 %64   ; &A[row+8][k0]
  %67 = getelementptr inbounds nuw [2 x i8], ptr %65, i64 %31   ; + tig*2
  %68 = load i32, ptr %67, align 4, !tbaa !13                   ; a[0]
  %69 = getelementptr inbounds nuw [2 x i8], ptr %66, i64 %31
  %70 = load i32, ptr %69, align 4, !tbaa !13                   ; a[1]
  %71 = getelementptr inbounds nuw [2 x i8], ptr %65, i64 %32   ; + tig*2+8
  %72 = load i32, ptr %71, align 4, !tbaa !13                   ; a[2]
  %73 = getelementptr inbounds nuw [2 x i8], ptr %66, i64 %32
  %74 = load i32, ptr %73, align 4, !tbaa !13                   ; a[3]
```

四条 `load i32`，对应 A fragment 的 4 个 `.b32`。**这是"用 32 位视角看 half"的胜利**：
如果按 `__half` 逐个取，会是 8 条 `load i16`（第 02-7 讲）。

注意 `!tbaa !13`——类型化别名分析信息，告诉优化器这些访问的类型不会互相别名。

## 六、循环体：B 的装载（第 93–108 行）

```llvm
  %75 = or disjoint i32 %59, %15                  ; k0 + tig*2
  %76 = mul nsw i32 %75, %4
  %77 = sext i32 %76 to i64
  %78 = getelementptr [2 x i8], ptr %36, i64 %77  ; &B[k0+tig*2][tile_n+gid]
  %79 = or disjoint i32 %59, %16                  ; k0 + tig*2 + 8
  %80 = mul nsw i32 %79, %4
  %81 = sext i32 %80 to i64
  %82 = getelementptr [2 x i8], ptr %36, i64 %81
  %83 = load i16, ptr %78, align 2, !tbaa !14     ; B[k][n]
  %84 = getelementptr inbounds [2 x i8], ptr %78, i64 %37   ; + N
  %85 = load i16, ptr %84, align 2, !tbaa !14     ; B[k+1][n]
  %86 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %83, i16 %85) #3, !srcloc !16
  ...
  %90 = tail call i32 asm "{  mov.b32 $0, {$1,$2};}\0A", "=r,h,h"(i16 %87, i16 %89) #3, !srcloc !16
```

画风明显不同于 A：**两条 16 位加载 + 一条汇编打包**，而且出现了两次。

为什么 B 不能像 A 那样一次 `load i32`？回看第 01-3 讲的：**B 的同一个寄存器里
两个 half 是"同一列相邻的两个 K"**，在 row-major 的内存里隔了 N 个元素。
所以只能分开取，再打包。

`%84` 那条 GEP 里那个 `+ %37`（= N）就是"跨一行"的具体体现。

## 七、循环体：mma（第 110–116 行）

```llvm
  %91 = tail call contract { float, float, float, float } asm sideeffect
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};\0A",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
        (i32 %68, i32 %70, i32 %72, i32 %74, i32 %86, i32 %90,
         float %60, float %61, float %62, float %63) #4, !srcloc !17
  %92 = extractvalue { float, float, float, float } %91, 0
  %93 = extractvalue { float, float, float, float } %91, 1
  %94 = extractvalue { float, float, float, float } %91, 2
  %95 = extractvalue { float, float, float, float } %91, 3
  %96 = add nuw nsw i32 %59, 16
  %97 = icmp slt i32 %96, %5
  br i1 %97, label %58, label %38, !llvm.loop !18
```

**这一小段是整份文件的"闭环"**，请慢慢看：

- `mma` 的 A 操作数是 `%68, %70, %72, %74`——正是前面那四条 `load i32`；
- B 操作数是 `%86, %90`——两条 `mov.b32` 打包的结果；
- C 是 `%60..%63`——四个 phi；
- 输出经过 `extractvalue` 变成 `%92..%95`——而它们正是下一轮 `%60..%63` 的 phi 输入。

**这个"数据绕一圈回到 phi"的闭环，就是 `D = A*B + C` 的累加形式在 SSA 下的样子。**

最后那条 `br`：`%96 = k0 + 16`，`%97 = (k0+16) < K`，满足就回 `%58`。
这就是 `for (int k0 = 0; k0 < K; k0 += 16)` 的完整形态——**它还在，没有展开**。

为什么不展开？因为 `K` 是运行期参数，trip count 未知。
第 4 部分会用三组实验把这件事讲透（顺便说：如果 `K` 是编译期常量，
这一版 LLVM 会把它完全展开）。

## 八、写回块（`%38`，第 50–75 行）

循环结束后跳到 `%38`。这个块里又有一组 phi：

```llvm
38:                                    ; preds = %58, %19
  %39 = phi i32 [ %22, %19 ], [ %27, %58 ]
  %40 = phi i64 [ %21, %19 ], [ %31, %58 ]
  %41 = phi i64 [ %20, %19 ], [ %33, %58 ]
  %42 = phi float [ 0.000000e+00, %19 ], [ %95, %58 ]
  %43 = phi float [ 0.000000e+00, %19 ], [ %94, %58 ]
  %44 = phi float [ 0.000000e+00, %19 ], [ %93, %58 ]
  %45 = phi float [ 0.000000e+00, %19 ], [ %92, %58 ]
```

**为什么写回块还需要 phi？** 因为执行流可以从两个地方到达这里：

- 从 `%19`（`K <= 0`，一次循环都没进）——这时累加器是初值 `0`；
- 从 `%58`（循环正常结束）——这时累加器是 `%92..%95`。

**"谁跳过来的"决定了用哪个值——这正是 phi 的定义。**

然后就是四条 `store`：

```llvm
  store float %45, ptr %50, align 4, !tbaa !11
  %56 = getelementptr inbounds nuw i8, ptr %50, i64 4
  store float %44, ptr %56, align 4, !tbaa !11
  store float %43, ptr %55, align 4, !tbaa !11
  %57 = getelementptr inbounds nuw i8, ptr %55, i64 4
  store float %42, ptr %57, align 4, !tbaa !11
```

注意 `%50` 和 `%56` 的差是 **4 字节**（一个 float）——这正是第 01-4 讲里
"`c[0]` 和 `c[1]` 在内存里相邻"这件事在 IR 里的样子。

## 九、小结

1. 循环前的准备块全是**下标算术**：`gid`/`tig`/`tile_m`/`tile_n` 各自算好，
   `lshr` 和 `or disjoint` 都是优化器改出来的写法。
2. 地址计算全靠 **GEP**：它把 `× 2`（一个 `__half` 的大小）藏进了类型里；
   元素类型被规格化成 `[2 x i8]`，读 IR 时看尺寸不要看名字。
3. 循环头的 5 个 phi 就是 `k0` 和 `c[0..3]`；
   `mma` 的输出经过 `extractvalue` 又回到 phi——**这就是累加在 SSA 下的形状**。

## 十、动手题

1. 在 `dumps/02-device-O2.ll` 里找到 `%37`（= N），说出它在循环体里被用了几次、
   每次是为了干什么。
2. 试着把 `K` 从"函数参数"改成 `constexpr int K = 64`，重新生成 O2 IR
   （`code/unroll_demo.cu` 就是干这个的），看看循环结构发生了什么变化。
   这就是第 4 部分"循环展开"那一讲的入口。

下一讲我们收尾：元数据速查，以及 O0 与 O2 的对照——数组到底去哪了。
