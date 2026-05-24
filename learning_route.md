# Nano-vLLM 学习路线图 (Backend Engineer 定制版)

作为一名后端工程师，你在系统架构、并发模型、内存管理和调度算法上有着天然的优势。`nano-vLLM` 是一个极简的大模型推理框架，它的很多核心思想（如 PagedAttention、Continuous Batching）可以直接映射到后端领域中的操作系统内存分页和任务队列调度上。

结合 [Zhihu 文章](https://zhuanlan.zhihu.com/p/2010638958783131701) 和 `nano-vLLM` 的代码结构，以下为你规划的进阶学习路线：

---

## 阶段一：建立大局观 (从 API 到 Request Lifecycle)

**目标**：理解 LLM 推理的基本流程，将 LLM 推理看作是一个特殊的“后端服务”。

1. **大模型推理的两种形态**：
   - **Prefill (预填充)**：处理用户的 prompt，是一次性计算大量 token。相当于“初始化”。
   - **Decode (解码)**：自回归生成，每次根据前面的内容生成下一个 token。相当于“流式响应”。
2. **理解痛点**：为什么普通的 PyTorch `model.generate()` 慢？为什么需要 vLLM？（内存碎片、MPS 统一内存利用率低、无法高效 Batching）。
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

## 阶段三：MPS 执行器与模型前向传播

**目标**：理解大模型如何在 Apple Metal 后端上以单设备方式运行。

1. **单执行器模型**：
   - MPS 分支使用单进程 `ModelRunner`，避免多设备同步与跨进程通信复杂度。
   - **后端映射**：类似一个独立工作进程承接调度器发来的批处理任务。
2. **统一内存与 KV Cache**：
   - 关注 `torch.mps.recommended_max_memory()` 与预分配 KV Cache 的关系。
3. **深入代码**：
   - 重点查看 `nanovllm/engine/model_runner.py`。
   - 理解输入张量、位置编码和 KV Cache 如何被放到 `mps` 设备上执行。

---

## 阶段四：深入模型与算子 (探索深度学习)

**目标**：剥开 Transformer 的黑盒，了解纯 PyTorch 注意力如何适配 MPS。

1. **Qwen 模型结构**：
   - 大致了解 `Qwen` 模型的结构，特别是它的注意力机制（Attention）。
   - 查看 `nanovllm/models/` 目录，看权重是如何被加载 (safetensors) 并映射到 PyTorch 层的。
2. **KV Cache 与注意力计算**：
   - **KV Cache**：就是空间换时间，把历史计算出的 Key 和 Value 缓存下来，避免 Decode 阶段重复计算。
   - **Scaled Dot-Product Attention**：MPS 分支使用 PyTorch 原生接口完成注意力计算，优先保证可运行性和可读性。
3. **分页 KV Cache 写入**：
   - 查看 `nanovllm/layers/attention.py`，理解逻辑 token 如何通过 block table 映射到物理 KV Cache 槽位。

---

## 阶段五：动手实践与 Debug (闭环)

**目标**：让代码跑起来，并通过断点和打日志验证认知。

1. **环境搭建**：使用 `environment-mps.yml` 创建 conda 环境，并安装 PyTorch、Transformers、xxhash。
2. **单步调试**：
   - 在 `LLMEngine.step()` 里打断点。
   - 走通一个请求从进入队列 -> 物理块分配 -> Model Runner 前向传播 -> 采样 (Sampling) 的全生命周期。
3. **修改与实验**：
   - 尝试在 `bench.py` 中改变并发请求数，观察 Scheduler 的日志输出，看看内存是如何被分配和回收的。

**总结语**：作为后端，你不需要一开始就搞懂矩阵微分或复杂的数学推导。把握住**数据流转、内存分配策略和并发调度**，你就能掌控 70% 的 vLLM 核心原理，剩下的 30% 就是逐步了解 Transformer 的组件即可。
