import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.context import get_context


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    valid = slot_mapping >= 0
    k_cache = k_cache.view(-1, *k_cache.shape[2:])
    v_cache = v_cache.view(-1, *v_cache.shape[2:])
    k_cache[slot_mapping[valid].long()] = key[valid]
    v_cache[slot_mapping[valid].long()] = value[valid]


def repeat_kv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    return x.repeat_interleave(num_heads // x.size(1), dim=1)


def varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, scale: float):
    outputs = []
    cu_seqlens_q = context.cu_seqlens_q_cpu
    cu_seqlens_k = context.cu_seqlens_k_cpu
    for i in range(len(cu_seqlens_q) - 1):
        qs, qe = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        ks, ke = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        qi, ki, vi = repeat_kv(qi, qi.size(1)), repeat_kv(ki, qi.size(1)), repeat_kv(vi, qi.size(1))
        mask = torch.ones(qe - qs, ke - ks, dtype=torch.bool, device=q.device).tril(diagonal=ke - ks - (qe - qs))
        out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=scale)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, context, scale: float):
    block_size = k_cache.shape[1]
    k_cache = k_cache.view(-1, *k_cache.shape[2:])
    v_cache = v_cache.view(-1, *v_cache.shape[2:])
    
    slots, mask = context.get_decode_slots_and_mask(block_size, q.device)
    
    k_batched = k_cache[slots].transpose(1, 2)  # (num_seqs, num_kv_heads, max_len, head_dim)
    v_batched = v_cache[slots].transpose(1, 2)  # (num_seqs, num_kv_heads, max_len, head_dim)
    
    qi = q.unsqueeze(2)  # (num_seqs, num_heads, 1, head_dim)
    
    ki = repeat_kv(k_batched, qi.size(1))
    vi = repeat_kv(v_batched, qi.size(1))
    
    out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=scale)
    return out.squeeze(2)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                o = paged_attention(q, k_cache, v_cache, context, self.scale)
            else:
                o = varlen_attention(q, k, v, context, self.scale)
        else:    # decode
            o = paged_attention(q, k_cache, v_cache, context, self.scale)
        return o
