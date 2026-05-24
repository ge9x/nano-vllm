import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # [Backend Analogy]: 类似于协议序列化工具 (Serializer)。
    # Tokenizer 负责把人类可读的字符串 (String) 编码成模型认识的数字 ID 数组 (Tokens)。
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # [Backend Analogy]: 类似于初始化整个后台微服务 (启动 Scheduler，分配内存池)。
    llm = LLM(path, enforce_eager=True)

    # [Backend Analogy]: 请求配置参数 (Request Options)。
    # 控制生成策略：temperature 控制随机性，max_tokens 类似 max_response_length。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # [Backend Analogy]: 报文组装 (Message Formatting)。
    # 大模型通常需要特定的 Prompt 模板（例如加上 <|im_start|>user 这样的标记）。
    # apply_chat_template 就像是把你传入的 JSON payload 封装成标准 HTTP/RPC 报文结构。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # [Backend Analogy]: 发送批量请求并等待处理完成。
    # 这里会阻塞 (Blocking call)，直到引擎处理完这批所有的 prompt。
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
