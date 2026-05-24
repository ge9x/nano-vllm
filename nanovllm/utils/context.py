from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # [Backend Analogy]: 类似于 Web 框架中的 ThreadLocal 或 Request Context。
    # 注意力层需要拿到全局调度状态，通过 Context 透传避免在每一层方法调用中塞满参数。
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    
    # Cached CPU lists to completely avoid host-device synchronizations in forward loops
    cu_seqlens_q_cpu: list[int] | None = None
    cu_seqlens_k_cpu: list[int] | None = None
    context_lens_cpu: list[int] | None = None
    block_tables_cpu: list[list[int]] | None = None

    # Lazy caches for fully vectorized paged attention
    decode_slots: torch.Tensor | None = None
    decode_attn_mask: torch.Tensor | None = None

    def get_decode_slots_and_mask(self, block_size: int, device: torch.device):
        if self.decode_slots is None:
            block_tables_cpu = self.block_tables_cpu
            context_lens_cpu = self.context_lens_cpu
            cu_seqlens_k_cpu = self.cu_seqlens_k_cpu
            
            num_seqs = len(block_tables_cpu) if block_tables_cpu is not None else 0
            
            lens = []
            if context_lens_cpu is not None:
                lens = context_lens_cpu
            elif cu_seqlens_k_cpu is not None:
                for i in range(num_seqs):
                    lens.append(cu_seqlens_k_cpu[i + 1] - cu_seqlens_k_cpu[i])
            else:
                raise ValueError("Both context_lens and cu_seqlens_k are None")
                
            max_len = max(lens) if lens else 0
            
            slots_list = []
            for i in range(num_seqs):
                length = lens[i]
                blocks = block_tables_cpu[i]
                seq_slots = []
                for block_id in blocks:
                    if block_id < 0 or len(seq_slots) >= length:
                        break
                    start = block_id * block_size
                    seq_slots.extend(range(start, start + block_size))
                seq_slots = seq_slots[:length]
                seq_slots.extend([0] * (max_len - len(seq_slots)))
                slots_list.append(seq_slots)
                
            self.decode_slots = torch.tensor(slots_list, dtype=torch.int64, device=device)
            
            col_indices = torch.arange(max_len, device=device).view(1, max_len)
            lens_tensor = torch.tensor(lens, device=device).view(-1, 1)
            mask = col_indices < lens_tensor
            self.decode_attn_mask = mask.unsqueeze(1).unsqueeze(1)
            
        return self.decode_slots, self.decode_attn_mask

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist() if cu_seqlens_q is not None else None
    cu_seqlens_k_cpu = cu_seqlens_k.cpu().tolist() if cu_seqlens_k is not None else None
    context_lens_cpu = context_lens.cpu().tolist() if context_lens is not None else None
    block_tables_cpu = block_tables.cpu().tolist() if block_tables is not None else None
    
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q_cpu=cu_seqlens_q_cpu,
        cu_seqlens_k_cpu=cu_seqlens_k_cpu,
        context_lens_cpu=context_lens_cpu,
        block_tables_cpu=block_tables_cpu,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

