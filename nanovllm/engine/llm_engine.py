import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        
        # [Backend Analogy]: 类似于 Nginx 的 Master-Worker 多进程架构。
        # 为了支持多 GPU 张量并行 (Tensor Parallelism)，启动多个子进程。
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # ModelRunner 是实际跑在 GPU 上的执行器
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
            
        # 主进程自己也跑一个 ModelRunner (rank 0)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # [Backend Analogy]: 核心调度器，类似于微服务网关中的流量分配器。
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 1. 调度：从队列中挑选一批请求 (决定谁能上 GPU 执行)
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        
        # 2. 执行：调用 ModelRunner 触发一次神经网络的前向传播
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 3. 后处理：将生成的 Token 存入请求的状态中，并判断请求是否结束，如果结束则释放内存
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 如果只传入单个 SamplingParams，复制为与 prompts 等长，便于 zip 配对
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 将每个输入封装为 Sequence 并提交给调度器（异步等待执行）
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}  # 临时映射：seq_id -> 生成的 token ids
        prefill_throughput = decode_throughput = 0.

        # 主循环：不断调用 step() 驱动调度与模型执行，直到所有请求完成
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            # 根据 num_tokens 的正负判断当前步骤为 prefill 还是 decode，并计算吞吐率
            if num_tokens > 0:
                # prefill 阶段（向模型输入上下文、填充 K/V 缓存）
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # decode 阶段（模型生成新 token）
                decode_throughput = -num_tokens / (perf_counter() - t)

            # 在进度条上显示吞吐率，帮助调试与性能分析
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 处理本次 step() 返回的已完成请求，将结果记入 outputs 并更新进度条
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()

        # 按 seq_id 的升序恢复为与输入顺序一致的列表
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 将 token ids 解码为字符串并返回包含文本与 token ids 的 dict 列表
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
