import openai
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_API_BASE,SIM_PARAMS

# class LLMClient:
#     """
#     LLM 接口调用封装模块
#     用于统一处理 GPT / DeepSeek 等模型的调用
#     """
#     def __init__(self):
#         openai.api_key = OPENAI_API_KEY
#         self.client = OpenAI(api_key=OPENAI_API_KEY,base_url=OPENAI_API_BASE)
#
#         mtype = SIM_PARAMS['gpt_type']
#         if mtype == 0:
#             self.model = 'gpt-4o-mini-2024-07-18'
#             # self.model = 'gpt-3.5-turbo-1106'
#         elif mtype == 1:
#             self.model = 'gpt-4-1106-preview'
#         # elif mtype == 2:
#         #     self.model = 'deepseek-chat'
#         else:
#             raise ValueError(f"Unsupported gpt_type: {mtype}")
#
#     def call(self, messages):
#         print(messages)
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=0,
#             timeout=120,
#             max_tokens=2048
#         )
#         return resp.choices[0].message.content

import time
import openai
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout, APIConnectionError

class LLMClient:
    """
    LLM 接口调用封装模块
    用于统一处理 GPT / DeepSeek 等模型的调用
    """
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

        mtype = SIM_PARAMS['gpt_type']
        if mtype == 0:
            self.model = 'gpt-4o-mini-2024-07-18'
        elif mtype == 1:
            self.model = 'gpt-4-1106-preview'
        else:
            raise ValueError(f"Unsupported gpt_type: {mtype}")

    def call(self, messages, max_retries=5):
        print(messages)
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    timeout=120,
                    max_tokens=2048
                )
                return resp.choices[0].message.content

            except (RateLimitError, APIError, APIConnectionError, Timeout) as e:  ####添加了睡眠策略 ，避免GPT连续多次访问，GPT罢工
                wait_time = 2 ** attempt  # 指数退避策略：1s, 2s, 4s, 8s...
                print(f"⚠️ 第 {attempt+1} 次尝试失败：{str(e)}，将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        raise RuntimeError("❌ 多次尝试后仍无法成功调用 LLM 接口。")


# ✅ 测试  test_llm_client.py
def test_llm_name():
    client = LLMClient()

    messages = [
        {"role": "system", "content": "你是一个喜欢简洁回答的机器人。"},
        {"role": "user", "content": "你的名字是什么？"}
    ]

    response = client.call(messages)
    print("模型回答：", response)
    assert isinstance(response, str) and len(response) > 0
    print("✅ 测试通过：模型成功返回内容。")

if __name__ == "__main__":
    test_llm_name()


# ####可以调用本地开源大模型的代码
# import os
# import openai
# import json
# from openai import OpenAI
# from config import OPENAI_API_KEY, OPENAI_API_BASE, SIM_PARAMS
#
# from transformers import AutoTokenizer
# from vllm import LLM as VLLMClient, SamplingParams
#
# class LLMClient:
#     """
#     通用 LLM 调用模块，支持 OpenAI GPT 与 本地 vLLM 模型（如 LLaMA、DeepSeek）
#     """
#
#     def __init__(self):
#         self.gpt_type = SIM_PARAMS['gpt_type']
#         self.temperature = SIM_PARAMS.get('temperature', 0)
#         self.max_tokens = SIM_PARAMS.get('max_tokens', 2048)
#
#         if self.gpt_type in [0, 1]:  # GPT系列：OpenAI
#             openai.api_key = OPENAI_API_KEY
#             self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
#             if self.gpt_type == 0:
#                 self.model = 'gpt-4o-mini-2024-07-18'
#             elif self.gpt_type == 1:
#                 self.model = 'gpt-4-1106-preview'
#
#         elif self.gpt_type == 2:  # 本地 VLLM: LLaMA
#             self.model_path = SIM_PARAMS['LLaMA_local_model_path']  # 例如 "/data/llama3"
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
#             self.client = VLLMClient(
#                 model=self.model_path,
#                 trust_remote_code=True,
#                 dtype="float16",
#                 max_model_len=2048
#             )
#         else:
#             raise ValueError(f"Unsupported gpt_type: {self.gpt_type}")
#
#     def call(self, messages):
#         print(messages)
#         if self.gpt_type in [0, 1]:
#             # OpenAI 模型调用
#             print("🔵 使用 GPT 模型调用：", self.model)
#             resp = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 temperature=self.temperature,
#                 timeout=120,
#                 max_tokens=self.max_tokens
#             )
#             return resp.choices[0].message.content
#
#         elif self.gpt_type == 2:
#             # vLLM 本地模型调用
#             # os.environ["CUDA_VISIBLE_DEVICES"] = SIM_PARAMS.get("cuda_device", "4")  # 默认为 GPU 0
#             print("🟢 使用本地 vLLM 模型：", self.model_path)
#
#             # 兼容 message 格式转换为 prompt string
#             role_map = {"system": "[System]", "user": "[User]", "assistant": "[Assistant]"}
#             prompt = ""
#             for msg in messages:
#                 role_tag = role_map.get(msg["role"], "")
#                 prompt += f"{role_tag} {msg['content']}\n"
#             prompt += "[Assistant] "
#
#             sampling_params = SamplingParams(
#                 temperature=self.temperature,
#                 top_p=0.9,
#                 max_tokens=self.max_tokens,
#                 stop=["[User]", "[System]", "[Assistant]"]
#             )
#
#             results = self.client.generate([prompt], sampling_params)
#             return results[0].outputs[0].text.strip()
#
#         else:
#             raise ValueError("Unsupported gpt_type")
#
# def test_llm_name():
#     client = LLMClient()
#
#     messages = [
#         {"role": "system", "content": "你是一个喜欢简洁回答的机器人。"},
#         {"role": "user", "content": "你的名字是什么？"}
#     ]
#
#     response = client.call(messages)
#     print("模型回答：", response)
#     assert isinstance(response, str) and len(response.strip()) > 0
#     print("✅ 测试通过：模型成功返回内容。")
#
#
# if __name__ == "__main__":
#     test_llm_name()


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import openai
# from openai import OpenAI
# from config import OPENAI_API_KEY, OPENAI_API_BASE, SIM_PARAMS
#
#
# class LLMClient:
#     """
#     LLM 接口调用封装模块
#     用于统一处理 GPT / DeepSeek 等模型的调用
#     """
#     def __init__(self):
#         mtype = SIM_PARAMS['gpt_type']
#
#         if mtype == 0:
#             openai.api_key = OPENAI_API_KEY
#             self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
#             self.model = 'gpt-4o-mini-2024-07-18'
#             self.provider = 'openai'
#
#         elif mtype == 1:
#             openai.api_key = OPENAI_API_KEY
#             self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
#             self.model = 'gpt-4-1106-preview'
#             self.provider = 'openai'
#
#         elif mtype == 2:
#             self.provider = 'deepseek'
#             model_dir = "/data/share_weight/DeepSeek-R1-Distill-Qwen-1.5B"
#             max_memory = {
#                 0: "10GiB",
#                 1: "10GiB",
#                 2: "10GiB",
#                 "cpu": "20GiB"
#             }
#             self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_dir,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 max_memory=max_memory,
#                 trust_remote_code=True
#             )
#             self.model.eval()
#         else:
#             raise ValueError(f"Unsupported gpt_type: {mtype}")
#
#     def call(self, messages):
#         print("📨 输入消息：", messages)
#
#         if self.provider == 'openai':
#             resp = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 temperature=0,
#                 timeout=120,
#                 max_tokens=2048
#             )
#             return resp.choices[0].message.content
#
#         elif self.provider == 'deepseek':
#             # 构造 prompt（忽略 system）
#             prompt = ""
#             for msg in messages:
#                 if msg["role"] == "user":
#                     prompt += f"用户：{msg['content']}\n"
#                 elif msg["role"] == "assistant":
#                     prompt += f"助手：{msg['content']}\n"
#                 elif msg["role"] == "system":
#                     continue
#             prompt += "助手："
#
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#             with torch.no_grad():
#                 output = self.model.generate(
#                     **inputs,
#                     max_new_tokens=256,
#                     do_sample=False,
#                     temperature=0
#                 )
#             decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
#             # 提取回答内容
#             response = decoded[len(prompt):].strip()
#             return response
#
#
# def test_llm_name():
#     client = LLMClient()
#
#     messages = [
#         {"role": "system", "content": "你是一个喜欢简洁回答的机器人。"},
#         {"role": "user", "content": "你的名字是什么？"}
#     ]
#
#     response = client.call(messages)
#     print("🤖 模型回答：", response)
#     assert isinstance(response, str) and len(response) > 0
#     print("✅ 测试通过：模型成功返回内容。")
#
# if __name__ == "__main__":
#     test_llm_name()
