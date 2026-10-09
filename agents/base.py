"""Agent 抽象基类。

每个 Agent = 一段专用 system prompt + 一次结构化 JSON 决策。
基类统一：LLM 注入、JSON 解析兜底、调用日志（供论文里画协作轨迹）。

AgentLog 的唯一定义在 state.py，这里再导出以兼容既有 import。
"""
from __future__ import annotations

from typing import Any

from llm.client import LLMClient, get_llm
from state import AgentLog

__all__ = ["AgentLog", "BaseAgent"]


class BaseAgent:
    name: str = "BaseAgent"
    system: str = ""

    def __init__(self, llm: LLMClient | None = None,
                 trace: list[AgentLog] | None = None) -> None:
        self.llm = llm or get_llm()
        self.trace = trace if trace is not None else []

    def _decide(self, user: str, max_tokens: int | None = None,
                temperature: float | None = None, note: str = "") -> Any:
        """决策调用，支持动态temperature和max_tokens"""
        out = self.llm.call_json(
            self.system, user,
            max_tokens=max_tokens,
            temperature=temperature
        )
        self.trace.append(AgentLog(agent=self.name, output=out, note=note))
        return out
