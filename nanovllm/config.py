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
    # [Backend Analogy]: 控制内存池 (KV Cache) 最多能占用多少统一内存。
    gpu_memory_utilization: float = 0.3
    # MPS 后端始终使用 eager 执行
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
