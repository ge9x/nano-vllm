# Nano-vLLM 学习路线图 (Backend Engineer 定制版)

作为一名后端工程师，你在系统架构、并发模型、内存管理和调度算法上有着天然的优势。`nano-vLLM` 是一个极简的大模型推理框架，它的很多核心思想（如 PagedAttention、Continuous Batching）可以直接映射到后端领域中的操作系统内存分页和任务队列调度上。

结合 [Zhihu 文章](https://zhuanlan.zhihu.com/p/2010638958783131701) 和 `nano-vLLM` 的代码结构，以下为你规划的进阶学习路线：

---

## 阶段一：建立大局观 (从 API 到 Request Lifecycle)

**目标**：理解 LLM 推理的基本流程，将 LLM 推理看作是一个特殊的“后端服务”。

1. **大模型推理的两种形态**：
   - **Prefill (预填充)**：处理用户的 prompt，是一次性计算大量 token。相当于“初始化”。
   - **Decode (解码)**：自回归生成，每次根据前面的内容生成下一个 token。相当于“流式响应”。
2. **理解痛点**：为什么普通的 PyTorch `model.generate()` 慢？为什么需要 vLLM？（内存碎片、GPU 显存利用率低、无法高效 Batching）。
3. **阅读入口代码**：
   - 查看 `example.py`，了解它是如何初始化的。
   - 追踪 `nanovllm/llm.py`，了解引擎 (`LLMEngine`) 是如何接受请求并驱动生成的。

---

## 阶段二：攻克调度层 (Backend 舒适区)

**目标**：理解 PagedAttention 思想在调度层的体现，这部分是后端工程师最容易上手且最有意思的系统设计。

1. **Continuous Batching (持续批处理)**：
   - **后端映射**：类似多路复用 (Multiplexing) 或消息队列的动态打包。传统的 Batch 是等一堆请求一起结束，Continuous Batching 是“先完先走，随进随出”。
2. **Block Manager 与 KV Cache 内存管理**：
   - **核心概念**：PagedAttention 借用了 OS 的**虚拟内存分页**思想。把显存（KV Cache）划分成固定大小的 Block。
   - **后端映射**：这就像是你手写了一个内存池（Memory Pool），管理逻辑块（Logical Token Blocks）到物理块（Physical Blocks）的映射表。
3. **深入代码**：
   - 重点阅读 `nanovllm/engine/` 下的调度器代码 (Scheduler)。
   - 研究 `Waiting Queue`（等待队列）和 `Running Queue`（运行队列）的转换逻辑。
   - 观察请求是如何在容量不足时被抢占 (Preempt) 或换出 (Swap) 的。

---

## 阶段三：张量并行与进程间通信 (分布式系统)

**目标**：理解大模型如何拆分到多张 GPU 上运行（Tensor Parallelism）。

1. **多进程模型 (Multiprocessing)**：
   - 文章提到 `nano-vLLM` 使用 Python 的 `multiprocessing` 启动多个 Worker（而不是 RPC）。
   - **后端映射**：类似 Nginx 的 Master-Worker 架构。
2. **SharedMemory 通信**：
   - Worker 之间如何同步数据？了解它是如何利用共享内存 (`multiprocessing.shared_memory`) 传递输入数据的。
3. **深入代码**：
   - 重点查看 `nanovllm/engine/model_runner.py`。
   - 理解矩阵乘法是如何被切块 (Shard) 分发到不同的 GPU 上，然后通过 `All-Reduce` 操作合并结果的。

---

## 阶段四：深入模型与算子 (探索深度学习)

**目标**：剥开 Transformer 的黑盒，了解自定义 CUDA kernel（Triton）的作用。

1. **Qwen 模型结构**：
   - 大致了解 `Qwen` 模型的结构，特别是它的注意力机制（Attention）。
   - 查看 `nanovllm/models/` 目录，看权重是如何被加载 (safetensors) 并映射到 PyTorch 层的。
2. **KV Cache 与 FlashAttention**：
   - **KV Cache**：就是空间换时间，把历史计算出的 Key 和 Value 缓存下来，避免 Decode 阶段重复计算。
   - **FlashAttention**：一种硬件友好（Hardware-aware）的加速算法，减少 GPU HBM 和 SRAM 之间的数据搬运。
3. **Triton 自定义算子**：
   - 文章提到了为了 Paged KV Cache 编写了自定义的 Triton kernel（而不是 C++ CUDA）。
   - 查看 `nanovllm/layers/` 里的 triton kernel 代码。Triton 是类 Python 语法，对于后端工程师来说，这是一种学习 GPU 并发编程（Thread block, Grid）的极佳切入点。

---

## 阶段五：动手实践与 Debug (闭环)

**目标**：让代码跑起来，并通过断点和打日志验证认知。

1. **环境搭建**：安装必要的依赖（PyTorch, Triton, FlashAttention 等）。
2. **单步调试**：
   - 在 `LLMEngine.step()` 里打断点。
   - 走通一个请求从进入队列 -> 物理块分配 -> Model Runner 前向传播 -> 采样 (Sampling) 的全生命周期。
3. **修改与实验**：
   - 尝试在 `bench.py` 中改变并发请求数，观察 Scheduler 的日志输出，看看内存是如何被分配和回收的。

**总结语**：作为后端，你不需要一开始就搞懂矩阵微分或复杂的数学推导。把握住**数据流转、内存分配策略和并发调度**，你就能掌控 70% 的 vLLM 核心原理，剩下的 30% 就是逐步了解 Transformer 的组件即可。
