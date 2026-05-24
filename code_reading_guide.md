# Nano-vLLM 源码阅读指南

为了让你更高效地阅读 `nano-vLLM` 源码，我已经在几个核心文件中添加了中文注释。以下是为你推荐的阅读路线及重点关注内容：

## 1. 入口点与请求生命周期 (请求是如何被接管的？)

**阅读文件**: 
- `example.py`
- `nanovllm/engine/llm_engine.py` (重点看 `__init__` 和 `step()` 方法)

**重点关注**:
- **单执行器架构**: 在 `llm_engine.py` 的初始化中，你可以看到调度器如何直接绑定一个 MPS `ModelRunner`。这更接近一个独立工作进程承接批处理任务。
- **Event Loop 机制**: 观察 `LLMEngine.generate()` 方法，它本质上是一个 `while not self.is_finished():` 的死循环。在每一次循环 (`step()`) 里，它完成“调度 -> 推理 -> 后处理”的闭环。

## 2. 调度与队列模型 (请求是如何排队的？)

**阅读文件**: 
- `nanovllm/engine/scheduler.py`

**重点关注**:
- **Continuous Batching (持续批处理)**: 在 `schedule()` 方法里，仔细看它是如何分别处理 `prefill` (预填充，类似新请求初始化) 和 `decode` (解码，类似流式返回结果) 的。它不会等所有请求都结束，只要 `running` 队列里的请求生成了一个 token，它就会马上再次调度。
- **抢占机制 (Preemption)**: 看看 `preempt()` 函数。当内存不足时，正在运行的请求会被强制放回 `waiting` 队列头部。这是非常典型的后端任务调度思维。

## 3. 内存管理 (MPS 统一内存怎么用才不浪费？)

**阅读文件**: 
- `nanovllm/engine/block_manager.py`

**重点关注**:
- **PagedAttention 思想**: 忘记矩阵相乘，把它当成一个“操作系统虚拟内存”。
- **Block 类**: 相当于一个物理内存页 (Physical Page)。
- **BlockManager**: 就是一个内存池。重点看 `allocate()` 和 `deallocate()` 方法，理解引用计数 (`ref_count`) 是如何被用来回收空闲物理内存块的。通过这种方式，不管请求的 token 长度如何变化，都尽量避免 KV Cache 内存碎片。

## 💡 下一步建议

完成上述三个核心文件的阅读后，你就已经掌握了 vLLM 架构的“骨架”。如果你想深入神经网络本身，可以再去阅读 `nanovllm/engine/model_runner.py` (看看数据是如何放到 MPS 设备并驱动前向传播的)，以及 `nanovllm/layers/` 下的模型定义。

现在，你可以打开对应文件查看我为你补充的中文注释了！如果有哪一段代码的逻辑不够清晰，随时告诉我，我们可以逐行分析。
