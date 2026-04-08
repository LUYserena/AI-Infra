# 🚀 AI Infra 学习经验

> 希望大家都能找到自己擅长喜欢的事情！

---

## 一、基础知识篇

### 1. Transformer

这肯定是理解 LLM 的基础，面试也问的挺多的，你得知道 Transformer 是怎么运行的，里面具体每个组件都是怎么工作的，如果有能力也可以尽量自己手搓一个出来跑跑。主要可以从下面几点入手：

- **Transformer 运行的整个过程**：Encoder-Decoder 架构，Decoder-Only 架构
- **Tokenizer** 是什么，Position 位置编码（问的不多，得了解）
- **LLM 自回归预测**：LLM 是怎么自回归预测下一个 Token 的
- **Attention 机制**：MHA、GQA、MLA（有个很基础的问题：*为什么没有 Q Cache？*）
- ……

### 2. 计算机体系结构 & 操作系统

作为系统人怎么能不学习体系结构，这部分还问的挺多的：

- **CPU**：构成、流水线、Cache、地址映射（全相联 / 组相联）、替换策略（LRU）
- **GPU**：结构、SM、Tensor Core、内存结构、调度
- **SIMT vs SIMD**
- **进程与线程**：以及它们的通信方式
- **虚拟内存**
- ……

---

## 二、框架调度篇

> 这里主要介绍关于具体的 AI Infra 需要学些什么。

### 1. vLLM

如果要做 LLM 相关的，了解 vLLM 还是很重要的，大家可以拉下来自己用 debug 模式跑一跑，知道整个具体过程。关于这个可以去看知乎**猛猿**的帖子，真神！

- **PagedAttention**
- **调度策略**
- **AsyncLLM**：如何实现异步推理
- **Prefix Caching**
- **vLLM 和 SGLang 的对比**
- ……

### 2. 常见的 Infra 知识

除了 vLLM 还有一些需要储备的知识点：

| # | 知识点 | 说明 |
|---|--------|------|
| 1 | **Chunked Prefill** | 分块预填充 |
| 2 | **并行策略** | 专家并行、张量并行、数据并行、流水线并行、序列并行 |
| 3 | **MoE 架构** | Mixture of Experts |
| 4 | **FlashAttention** | 高效注意力算法 |
| 5 | **Continuous Batching** | 连续批处理 |
| 6 | **量化** | INT8、BF16、FP32 等 |
| 7 | **采样方法** | 投机采样（Speculative Decoding）等 |
| 8 | **分布式编程** | 通信函数：`all_reduce` 等 |
| 9 | **Roofline 模型** | 性能分析模型 |

### 3. CUDA

CUDA 是个好东西，但不适合我这样的数学小笨蛋学习，我从 GEMM 入门到放弃 😂。如果大家能学会请教教我。

- **基本的 CUDA 编程模型**：Kernel、内存模型、线程模型
- **Bank Conflict**（问就是我不会 🤣）
- **GEMM 的 CUDA 编程**（这个最基础了）
- **`all_reduce` 算子**
- **FlashAttention 算子**
- ……

---

> 📝 目前主包能想到的就这些了，如果后面还想到了就二编吧。
