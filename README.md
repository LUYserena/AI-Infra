# AI-Infra
Suggestions for How to Learn AI Infra

AI Infra学习经验
一、 基础知识篇
	
1.  Transformer
这肯定是理解LLM的基础，面试也问的挺多的，你得知道Transformer是怎么运行的，里面具体每个组件都是怎么工作的，如果有能力也可以尽量自己手搓一个出来跑跑。主要可以从下面几点入手：
（1）Tranfromer运行的整个过程是怎么样的，encoder-decoder架构，decoder only架构。
（2）Tokenizer是什么，position位置编码（问的不多，得了解）
（3）LLM是怎么自回归预测下一个token的
（4）Attention是怎么做的，MHA，GQA，MLA（有个很基础的问题：为什么没有Q Cache）
.....
	
2.  计算机体系结构 & 操作系统
作为系统人怎么能不学习体系结构，这部分还问的挺多的：
（1）CPU构成、流水线，Cache，地址映射（全相联组相联）、替换策略（LRU）
（2）GPU的结构，SM、Tensor core、内存结构、调度
（3）SIMT  VS  SIMD
（4）进程、线程以及他们的通信
（5）虚拟内存
.....


二、框架调度篇
这里主要介绍关于具体的ai infra需要学些什么。
1.  vLLM
如果要做LLM相关的了解vLLM还是很重要的，大家可以拉下来自己用debug模式跑一跑，知道整个具体过程。关于这个可以去看知乎猛yuan的帖子，真神！
（1）paged attention
（2）调度策略
（3）AsynLLM如何实现异步推理
（4）prefix caching
（5）vLLM和SGLang的对比
......

2.  常见的infra知识：
除了vLLM还有一些需要储备的知识点
（1）chunked prefill
（2）并行策略：专家并行、张量并行、数据并行、流水线并行、序列并行
（3）MoE架构
（4）flashattention
（5）continuous batching
（6）量化 INT8 BF16 BF32...
（7）采样方法，投机采样
（8）分布式编程及其通信函数，all_reduce等
（9）roofline模型
.....

3.  CUDA
CUDA是个好东西，但不适合我这样的数学小笨蛋学习，我从GEMM入门到放弃，如果大家能学会请教教我。
（1）基本的CUDA编程模型，kernel、内存模型、线程模型
（2）bank conflict（问就是我不会）
（3）GEMM的CUDA编程（这个最基础了）
（4）all_reduce算子
（5）flash attention算子
.....

目前主包能想到的就这些了，如果后面还想到了就二编吧，希望大家都能找到自己擅长喜欢的事情！

