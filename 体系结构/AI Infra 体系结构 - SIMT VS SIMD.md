# 📝AI Infra 体系结构：SIMT vs SIMD


# AI Infra —— 体系结构 SIMT vs SIMD

## 一、Flynn分类法：并行计算的4种模型

Flynn分类法按照**指令流**和**数据流**的数量，把计算机架构分为4类：

| 分类 | 指令流 | 数据流 | 说明 | 代表 |
|------|--------|--------|------|------|
| **SISD** | 单 | 单 | 传统单核CPU | 早期计算机 |
| **SIMD** | 单 | 多 | 一条指令同时处理多个数据 | CPU的SSE/AVX指令 |
| **MISD** | 多 | 单 | 多条指令处理同一数据（理论上的） | 几乎不存在 |
| **MIMD** | 多 | 多 | 多条指令处理多个数据 | 多核CPU、分布式系统 |

> 💡 重点关注SIMD和SIMT（SIMT是NVIDIA在SIMD基础上的演进）。

## 二、SIMD：一条指令，多个数据

SIMD = Single Instruction, Multiple Data

### 核心思想

用**一条指令**同时对**多个数据**执行相同操作。

```
传统方式（逐个计算）：
  a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]  → 4条指令

SIMD方式（一次搞定）：
  VADD [a[0],a[1],a[2],a[3]], [b[0],b[1],b[2],b[3]]  → 1条指令
```

### CPU上的SIMD

- **SSE**：128位寄存器，可同时处理4个float
- **AVX2**：256位寄存器，可同时处理8个float
- **AVX-512**：512位寄存器，可同时处理16个float

### SIMD的特点

- ✅ 所有数据**必须执行完全相同的操作**
- ✅ 硬件层面是**真正的锁步执行**（lockstep）
- ✅ 如果遇到分支（if-else），只能选一条路走
- ❌ 不能让不同数据走不同的执行路径

## 三、SIMT：GPU的并行模型

SIMT = Single Instruction, Multiple Threads（NVIDIA提出）

### 核心思想

大量**线程**被分成小组（Warp），同一个Warp内的线程执行**相同的指令**，但每个线程有**自己的寄存器和地址**。

### Warp：GPU调度的基本单位

- 一个Warp = **32个线程**
- 同一个Warp内的线程在同一时刻执行同一条指令
- 但每个线程可以访问不同的数据地址
- 硬件以Warp为单位进行调度

```
Warp（32个线程）：
  Thread 0:  执行指令A，数据地址 addr_0
  Thread 1:  执行指令A，数据地址 addr_1
  Thread 2:  执行指令A，数据地址 addr_2
  ...
  Thread 31: 执行指令A，数据地址 addr_31
```

### Warp Divergence：SIMT的"代价"

当Warp内的线程遇到分支（if-else）时：

```c
if (threadIdx.x < 16) {
    do_A();  // 前16个线程走这
} else {
    do_B();  // 后16个线程走这
}
```

- SIMD：直接不支持，编译器必须消除分支
- SIMT：**两个分支都会执行**，不走这条路的线程被mask掉（闲置等待）

这就是**Warp Divergence**——Warp内线程走了不同分支，导致串行执行，性能下降。

> 💡 面试高频考点：什么是Warp Divergence？怎么避免？
> 答：尽量让同一个Warp内的线程走相同的分支路径。

## 四、SIMT vs SIMD 核心区别

| 对比维度 | SIMD | SIMT |
|---------|------|------|
| 提出者 | 通用概念 | NVIDIA |
| 执行单位 | 向量寄存器（128/256/512位） | Warp（32个线程） |
| 线程概念 | 无独立线程，是数据通道 | 每个线程有独立PC、寄存器、栈 |
| 分支处理 | 不支持（需编译器处理） | 支持（Warp Divergence，串行执行） |
| 编程模型 | 显式向量指令（intrinsics） | 标量代码自动并行（CUDA） |
| 灵活性 | 低（必须完全相同操作） | 高（允许分支，但有性能代价） |
| 典型硬件 | CPU（SSE/AVX） | GPU（CUDA Core） |

### 一句话总结

- **SIMD**：一条指令操作一个宽向量，数据通道没有独立性
- **SIMT**：看起来像多个独立线程，实际上以Warp为单位执行相同指令，兼顾灵活性和并行度

> 💡 SIMT可以理解为"对程序员更友好的SIMD"——你写的是标量代码（一个线程的逻辑），硬件帮你并行化。

## 五、在 AI Infra 中的应用

### GPU为什么适合跑大模型？

大模型的核心运算是**矩阵乘法**（GEMM），矩阵乘法的特点：
- 大量数据需要做相同操作 → 天然适合SIMT
- 数据并行度极高 → 可以喂饱GPU的成千上万个线程

### CUDA编程与SIMT

- CUDA的**kernel**就是在SIMT模型上运行
- 你写的每个kernel函数描述的是**单个线程**的行为
- GPU硬件自动把线程组织成Warp并行执行
- 理解SIMT才能优化CUDA kernel（避免Warp Divergence、优化内存合并访问）

### 相关优化知识

| 概念 | 和SIMT的关系 |
|------|-------------|
| Warp Divergence | SIMT分支处理的代价 |
| Bank Conflict | 共享内存访问和Warp调度相关 |
| Coalesced Memory Access | Warp内线程的内存访问模式 |
| Occupancy | SM上活跃Warp的比例 |

> 📌 下一篇预告：体系结构篇总结！一张图带你回顾从CPU到GPU的全部知识点 🗺️
