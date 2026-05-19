import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    # [Backend Analogy]: 类似于网关的“最大请求体大小限制”。决定了一次 prefill 阶段最多处理多少个 token。
    max_num_batched_tokens: int = 16384
    # [Backend Analogy]: 类似于 Nginx 的 worker_connections，最大并发连接数。
    max_num_seqs: int = 512
    # [Backend Analogy]: 单个请求的最大超时/长度限制。
    max_model_len: int = 4096
    # [Backend Analogy]: JVM 的 -Xmx，控制内存池 (KV Cache) 最多能占用多少 GPU 显存。
    gpu_memory_utilization: float = 0.9
    # [Backend Analogy]: 分库分表的节点数 (分布式推理使用的 GPU 数量)。
    tensor_parallel_size: int = 1
    # 是否强制禁用 CUDA Graph (执行计划缓存)
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
