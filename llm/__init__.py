"""LLM 层：OpenAI 兼容后端的 JSON 调用封装 + 磁盘缓存 + 离线复现。

对上只暴露 LLMClient / get_llm。Agent 只依赖 call_json(system, user)，
不感知具体后端、重试、缓存与推理模型的正文/思维链分离。
"""
from llm.client import LLMClient, get_llm

__all__ = ["LLMClient", "get_llm"]
