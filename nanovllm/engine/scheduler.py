from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 单次调度允许的最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 一次 prefill 阶段允许的最大 token 数
        self.eos = config.eos
        self.block_size = config.kvcache_block_size # 每个 Block 管理的 token 数量 (KV cache 的块大小)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # [Backend Analogy]: 类似于操作系统的任务队列。
        # waiting 队列：存放刚接收到，还未分配资源的请求 (类似就绪队列)
        self.waiting: deque[Sequence] = deque()
        # running 队列：存放正在生成 token 的请求 (类似运行队列)
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill (预填充阶段)：处理新的请求。
        # [Backend Analogy]: 类似处理刚到达的请求，需要分配大量的计算资源（Token）。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: # 优先处理 waiting 队列直到达到并发数
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens # 本次批次剩余可用 token 配额；若为 0 则退出循环
            if remaining == 0:
                break
            if not seq.block_table: # seq.block_table 为空表示该请求之前未被调度过，需要分配新的 Block；否则表示该请求之前被抢占过，正在等待重新调度，此时优先使用之前分配的 Block（如果还在）以减少内存碎片
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1: # 若当前 free blocks 不足以分配所需新块 → 暂停 prefill
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size # 根据可复用块数计算本次仍需填充的 token 数. 
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens # 已经调度过但被抢占的请求，继续填充剩余 token 数
            # 本批次剩余 token 不足且已经有请求被调度了 → 
            # 暂停 prefill 以先执行已调度的请求，减少等待时间
            if remaining < num_tokens and scheduled_seqs:
                break 
            # 分配 seq 的 block_table（如果之前未被调度过）并设置本次预填充的 token 数
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining) # min(该请求剩余需要填充的 token 数, 本批次剩余可用 token 数)
            num_batched_tokens += seq.num_scheduled_tokens # 累加本批次已调度的 token 数
            # 如果 seq 的所有 token 都已被调度过了（即本次 prefill 后该请求就可以进入 decode 阶段了），
            # 则将其状态改为 RUNNING 并从 waiting 队列移到 running 队列
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode (解码阶段)：处理正在生成的请求。
        # [Backend Analogy]: 类似流式返回结果。这是 Continuous Batching (持续批处理) 的核心：
        # 只要有空闲资源，就让 running 队列里的请求继续生成下一个 token，
        # 而不是等所有请求都处理完了再一起返回。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # can_append 判断是否有空闲的 block 可以分配给 seq 以生成下一个 token
            while not self.block_manager.can_append(seq):
                # 如果没有空闲 block 可用，说明 GPU 内存已满了，此时需要抢占 (Preempt) 当前正在运行的请求，释放它占用的内存块，并把它放回 waiting 队列的头部，等待下次资源充足时重新调度。
                if self.running:
                    self.preempt(self.running.pop())
                # 如果 running 队列已经空了，说明没有其他请求可以抢占了，此时只能先让 seq 自己被抢占，直到有资源可用再继续调度它生成下一个 token。 
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1 # decode 阶段每次只生成 1 个 token
                seq.is_prefill = False
                self.block_manager.may_append(seq) # 如果当前 block 已满了，分配一个新 block 给 seq 以生成下一个 token
                scheduled_seqs.append(seq) # 将 seq 放入本次调度的列表中，等待被 ModelRunner 执行生成下一个 token 的前向传播
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # [Backend Analogy]: 抢占 (Preemption)。
        # 当显存 (KV Cache 空间) 不足时，强制将当前请求暂停，释放其持有的物理内存块，
        # 并把它放回 waiting 队列的头部，等待下次资源充足时重新调度。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True # 被抢占后物理 KV block 被释放，重新调度时必须把缺失的 prompt token 再填回 KV（即 prefill）
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 将模型生成的 token_id 存入对应请求的状态中，并判断该请求是否已经完成
    # （生成了 EOS token 或者达到了 max_tokens 限制），
    # 如果完成了则释放其占用的内存块并从 running 队列移除。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0 # 本次调度的 token 已经被生成了，重置 num_scheduled_tokens 以便下次调度时重新计算
            # prefill 阶段且 seq 还没有被完全填充完（即 KV Cache 里还缺 token） 
            # → 继续等待后续 prefill 来填充剩余 token
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 否则 decode 阶段或 prefill 后 seq 已经被完全填充完了 
            # → 生成下一个 token 并判断是否完成
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
