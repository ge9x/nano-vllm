import torch
from torch import nn


class Sampler(nn.Module):
    # [Backend Analogy]: 就像是一个“带权重的随机负载均衡算法”。
    # 根据模型输出的打分 (Logits)，加上温度参数 (Temperature) 来控制随机性，
    # 最终决定生成哪个 Token。
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
