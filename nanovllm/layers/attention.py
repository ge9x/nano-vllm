import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# [Backend Analogy]: 类似于自己写了一段嵌入式的汇编代码。
# Triton 允许用 Python 语法写 GPU 底层算子 (Kernel)。
# 这里的 kernel 是为了把当前计算出的 Key/Value 写回到 BlockManager 分配的离散物理显存块中。
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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
            
        # 检测 GPU 算力是否低于 8.0 (即非 Ampere 架构，不支持 FlashAttention)
        if torch.cuda.get_device_capability()[0] < 8:
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache (前缀缓存)
                    k, v = k_cache, v_cache
                
                # 预热/首 Token 阶段 (Prefill)
                batch_size = len(context.cu_seqlens_q) - 1
                outputs = []
                block_size = k_cache.shape[1] if k_cache.numel() else 0
                
                for i in range(batch_size):
                    # 1. 提取当前序列的 Query 范围
                    start_q, end_q = int(context.cu_seqlens_q[i]), int(context.cu_seqlens_q[i+1])
                    seq_q = q[start_q:end_q]  # [L_q, num_heads, head_dim]
                    
                    # 2. 提取当前序列的 Key 和 Value
                    if context.block_tables is not None:
                        # 存在前缀缓存，从离散块中重构完整的 key/value 序列
                        start_k, end_k = int(context.cu_seqlens_k[i]), int(context.cu_seqlens_k[i+1])
                        L_k = end_k
                        num_blocks = (L_k + block_size - 1) // block_size
                        blocks = context.block_tables[i, :num_blocks]
                        
                        # 重构为: [L_k, num_kv_heads, head_dim]
                        seq_k = k[blocks].view(-1, self.num_kv_heads, self.head_dim)[:L_k]
                        seq_v = v[blocks].view(-1, self.num_kv_heads, self.head_dim)[:L_k]
                    else:
                        start_k, end_k = int(context.cu_seqlens_k[i]), int(context.cu_seqlens_k[i+1])
                        seq_k = k[start_k:end_k]  # [L_k, num_kv_heads, head_dim]
                        seq_v = v[start_k:end_k]  # [L_k, num_kv_heads, head_dim]
                    
                    # 3. 形状转换以匹配 SDPA: [1, num_heads, seq_len, head_dim]
                    seq_q = seq_q.transpose(0, 1).unsqueeze(0)
                    seq_k = seq_k.transpose(0, 1).unsqueeze(0)
                    seq_v = seq_v.transpose(0, 1).unsqueeze(0)
                    
                    # 4. GQA (Grouped-Query Attention) 广播扩展
                    if self.num_heads != self.num_kv_heads:
                        num_queries_per_kv = self.num_heads // self.num_kv_heads
                        seq_k = seq_k.repeat_interleave(num_queries_per_kv, dim=1)
                        seq_v = seq_v.repeat_interleave(num_queries_per_kv, dim=1)
                    
                    # 5. 构造完美的因果注意力掩码 (Causal Mask)
                    # 对于 q_idx 和 k_idx，只允许关注当前及过去的位置 (即 k_idx <= q_idx + 历史偏移)
                    L_q, L_k = seq_q.shape[2], seq_k.shape[2]
                    q_indices = torch.arange(L_q, device=q.device).view(-1, 1)
                    k_indices = torch.arange(L_k, device=q.device).view(1, -1)
                    attn_mask = k_indices <= q_indices + (L_k - L_q)
                    
                    # 6. 使用 PyTorch 原生高度优化的 SDPA 进行计算
                    out_seq = torch.nn.functional.scaled_dot_product_attention(
                        seq_q, seq_k, seq_v, attn_mask=attn_mask, scale=self.scale
                    )
                    
                    # 7. 恢复原始形状: [L_q, num_heads, head_dim]
                    out_seq = out_seq.squeeze(0).transpose(0, 1)
                    outputs.append(out_seq)
                
                o = torch.cat(outputs, dim=0)
            else:
                # 逐 Token 生成阶段 (Decode)
                batch_size = q.shape[0]
                outputs = []
                block_size = k_cache.shape[1]
                
                for i in range(batch_size):
                    # 1. 提取当前序列的 Query (当前 Step 生成的单个 Token)
                    seq_q = q[i]  # [num_heads, head_dim]
                    
                    # 2. 从块中重构完整的 key/value 序列 (包含当前已写入的 Token)
                    L_k = int(context.context_lens[i])
                    num_blocks = (L_k + block_size - 1) // block_size
                    blocks = context.block_tables[i, :num_blocks]
                    
                    seq_k = k_cache[blocks].view(-1, self.num_kv_heads, self.head_dim)[:L_k]
                    seq_v = v_cache[blocks].view(-1, self.num_kv_heads, self.head_dim)[:L_k]
                    
                    # 3. 形状转换以匹配 SDPA: [1, num_heads, 1, head_dim] 和 [1, num_kv_heads, L_k, head_dim]
                    seq_q = seq_q.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, head_dim]
                    seq_k = seq_k.transpose(0, 1).unsqueeze(0)  # [1, num_kv_heads, L_k, head_dim]
                    seq_v = seq_v.transpose(0, 1).unsqueeze(0)  # [1, num_kv_heads, L_k, head_dim]
                    
                    # 4. GQA (Grouped-Query Attention) 广播扩展
                    if self.num_heads != self.num_kv_heads:
                        num_queries_per_kv = self.num_heads // self.num_kv_heads
                        seq_k = seq_k.repeat_interleave(num_queries_per_kv, dim=1)
                        seq_v = seq_v.repeat_interleave(num_queries_per_kv, dim=1)
                    
                    # 5. 使用 PyTorch 原生高度优化的 SDPA 进行计算 (此处无需因果掩码，因为 Query 长度为 1 且包含过去所有状态)
                    out_seq = torch.nn.functional.scaled_dot_product_attention(
                        seq_q, seq_k, seq_v, scale=self.scale
                    )
                    
                    # 6. 恢复原始形状: [num_heads, head_dim]
                    out_seq = out_seq.squeeze(2).squeeze(0)
                    outputs.append(out_seq)
                
                # 重新堆叠回 [batch_size, 1, num_heads, head_dim] 以匹配 FlashAttention 接口返回值形状
                o = torch.stack(outputs, dim=0).unsqueeze(1)
        else:
            # [Backend Analogy]: FlashAttention 是目前最高效的注意力加速库。
            # 这里区分了 prefill (一次性计算大量新 token) 和 decode (每次只增加一个 token)。
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:    # decode
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        return o
