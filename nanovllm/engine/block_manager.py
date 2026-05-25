from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    # [Backend Analogy]: 类似于操作系统中的物理内存页 (Physical Page)。
    # 这里存储的是大模型计算过程中的 KV Cache。
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 引用计数，用于内存回收
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    # [Backend Analogy]: 类似于操作系统的内存池管理器 (Memory Pool Manager)。
    # 它负责把连续的请求打散分配到不连续的物理 Block 中 (PagedAttention 核心思想)，
    # 彻底解决显存碎片化的问题。
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        
        # 空闲的 block 列表 (类似 free page list)
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用的 block 集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配一个物理 block，返回其 block_id
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 判断一个 Sequence 的前缀有多少个 block 可以复用（已在 hash_to_block_id 中），
    # 检查当前空闲物理 block 是否足够分配该序列剩余需要的新 block。若空闲不足则返回 -1
    # 否则返回可复用的前缀块数（num_cached_blocks）
    def can_allocate(self, seq: Sequence) -> int:
        h = -1 # 前缀块的哈希值
        num_cached_blocks = 0 # 可复用的前缀块数
        num_new_blocks = seq.num_blocks # 该序列总共需要的块数
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i) # 取出该块的 token ids
            h = self.compute_hash(token_ids, h) # 计算该块的哈希值(包含前一个块的哈希值以形成链式依赖)
            block_id = self.hash_to_block_id.get(h, -1) # 查找是否有物理块的哈希值与该块匹配
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break # 没有找到可复用的块
            num_cached_blocks += 1 # 该块可复用
            if block_id in self.used_block_ids:
                num_new_blocks -= 1 # 该块使用中，不需要分配新的块（否则即映射到的是一个当前空闲的物理块)
        if len(self.free_block_ids) < num_new_blocks:
            return -1 # 空闲物理块不足以分配剩余需要的新块
        return num_cached_blocks

    # 为给定的 Sequence 分配物理 block，复用前缀已存在的 block 并为其余块从空闲池分配新块
    # 同时更新该 Sequence 的 block_table 和 num_cached_tokens
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 将某个 Sequence 的物理 block 释放回空闲池，回收该序列占用的 KV cache 资源。
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 判断 seq 是否可以在当前批次继续生成下一个 token（即是否有空闲的 block 可以分配给它）
    def can_append(self, seq: Sequence) -> bool:
        # len(seq) 返回当前 seq 的 token 数
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 将 seq 的下一个 token 分配到一个 block 中(如果当前 block 已满了)
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 将 seq 的新块（由 prefill 产生）记录到 hash_to_block_id 中以便未来复用
    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # h 是 seq 前缀块的哈希值
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            # 计算该块的哈希值(包含前一个块的哈希值以形成链式依赖
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
