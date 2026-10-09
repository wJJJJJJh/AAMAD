"""OpenAI 兼容 LLM 客户端：JSON 调用 + 磁盘缓存 + 离线复现 + 稳健解析。

面向本项目的两点约束做了专门处理：
  1) 推理模型（deepseek-v4-pro 等）会把思维链放在 reasoning_content，正文在 content；
     只解析 content，且给足 max_tokens 避免正文被推理挤空；
  2) 审稿人复现：LLM_OFFLINE=1 时缓存未命中直接报错，绝不打网络——保证「同一份
     冻结的 .cache/ 必得同一组预测」。

安全：API 密钥只从环境变量读取（见 config.LLM_API_KEY），绝不写死在源码里。

缓存键 = sha256(model + system + user + max_tokens + temperature)，落盘到
CACHE_DIR/llm/；命中即返回，不计入网络请求。统计量 calls / cache_hits /
total_tokens 供 run.py 报告成本。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import config

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)


def _extract_json(text: str) -> dict:
    """从（可能含推理噪声/代码围栏/前后缀说明的）模型正文里稳健抽出 JSON 对象。

    依次尝试：整体 parse → 代码围栏内 parse → 首个平衡花括号块 parse。
    全部失败返回 {}，由调用方按业务语义降级（如置 type=None）。"""
    s = (text or "").strip()
    if not s:
        return {}
    # 1) 整体就是 JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_list": obj}
    except Exception:  # noqa: BLE001
        pass
    # 2) ```json ... ``` 围栏
    for m in _FENCE.finditer(s):
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else {"_list": obj}
        except Exception:  # noqa: BLE001
            continue
    # 3) 扫描首个平衡的 {...} 块（容忍字符串内花括号）
    block = _first_json_object(s)
    if block:
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else {"_list": obj}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _first_json_object(s: str) -> str:
    """返回首个大括号平衡的子串；找不到返回空串。识别字符串与转义，避免被
    字符串内的括号/引号误导。"""
    start = s.find("{")
    if start < 0:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""


class LLMClient:
    """OpenAI 兼容后端的 JSON 调用封装（单例经 get_llm 取用）。

    call_json(system, user) 是唯一对外方法：返回解析好的 dict。
    统计量：calls（真实网络请求数）/ cache_hits / total_tokens。"""

    def __init__(self) -> None:
        self.calls = 0
        self.cache_hits = 0
        self.total_tokens = 0
        self._client: Any = None          # 懒加载 openai.OpenAI
        self._cache_dir = config.CACHE_DIR / "llm"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- 缓存
    def _cache_key(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        """生成缓存键，包含temperature以区分不同温度的调用"""
        raw = "\x00".join([config.LLM_MODEL, system, user,
                           str(max_tokens), str(temperature)])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> dict | None:
        f = self._cache_path(key)
        if not f.exists():
            return None
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            return rec.get("parsed") if isinstance(rec, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def _cache_write(self, key: str, parsed: dict, raw: str) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps({"parsed": parsed, "raw": raw}, ensure_ascii=False),
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # -------------------------------------------------------------- 后端
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not config.LLM_API_KEY:
            raise RuntimeError(
                "未设置 API 密钥。请设置环境变量 DEEPSEEK_API_KEY（或 DASHSCOPE_API_KEY）；"
                "密钥绝不写入源码。离线复现请设 LLM_OFFLINE=1 并使用已有 .cache/。")
        from openai import OpenAI
        self._client = OpenAI(api_key=config.LLM_API_KEY,
                              base_url=config.LLM_BASE_URL,
                              timeout=config.LLM_TIMEOUT)
        return self._client

    def _raw_completion(self, system: str, user: str, max_tokens: int,
                        temperature: float) -> tuple[str, int]:
        """一次带重试的补全，返回 (content 正文, tokens)。

        对于推理模型（deepseek-reasoner/deepseek-v4-pro），优先从 reasoning_content 提取，
        因为JSON输出可能在思维链里而不是 content 字段。
        """
        client = self._ensure_client()
        last_err: Exception | None = None
        for attempt in range(config.LLM_MAX_RETRY):
            try:
                resp = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                msg = resp.choices[0].message

                # 优先从 content 提取（格式化输出），空时退化到 reasoning_content
                # 推理模型可能把思维链放在 reasoning_content，最终答案放在 content
                content = (msg.content or "").strip()
                if not content and hasattr(msg, 'reasoning_content'):
                    content = (msg.reasoning_content or "").strip()

                tokens = 0
                if getattr(resp, "usage", None) is not None:
                    tokens = int(getattr(resp.usage, "total_tokens", 0) or 0)
                return content, tokens
            except Exception as e:  # noqa: BLE001
                last_err = e
                # response_format 不被某些端点支持时，退回无约束再试一次
                if "response_format" in str(e).lower() and attempt == 0:
                    try:
                        resp = client.chat.completions.create(
                            model=config.LLM_MODEL,
                            messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                            temperature=temperature,
                            max_tokens=max_tokens)
                        msg = resp.choices[0].message

                        # 优先从 reasoning_content 提取（推理模型）
                        content = ""
                        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                            content = msg.reasoning_content
                        else:
                            content = (msg.content or "")

                        tokens = int(getattr(getattr(resp, "usage", None),
                                             "total_tokens", 0) or 0)
                        return content, tokens
                    except Exception as e2:  # noqa: BLE001
                        last_err = e2
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM 调用失败（重试 {config.LLM_MAX_RETRY} 次）: {last_err}")

    # -------------------------------------------------------------- 对外
    def call_json(self, system: str, user: str, max_tokens: int | None = None,
                  temperature: float | None = None) -> dict:
        """发一次 JSON 调用并返回解析好的 dict。

        - 命中磁盘缓存直接返回（cache_hits++，不计网络）；
        - LLM_OFFLINE=1 且未命中 → 报错（绝不打网络，保证离线复现确定性）；
        - 正文解析失败返回 {}，由调用方按业务降级。"""
        mt = int(max_tokens or config.LLM_MAX_TOKENS)
        temp = float(temperature if temperature is not None else config.LLM_TEMPERATURE)
        key = self._cache_key(system, user, mt, temp)

        if config.LLM_CACHE:
            cached = self._cache_read(key)
            if cached is not None:
                self.cache_hits += 1
                return cached

        if config.LLM_OFFLINE:
            raise RuntimeError(
                f"LLM_OFFLINE=1 且缓存未命中（key={key[:12]}…）。离线复现需使用随仓库冻结的 "
                ".cache/llm/；如需重新生成请取消离线模式并配置 API 密钥。")

        content, tokens = self._raw_completion(system, user, mt, temp)
        self.calls += 1
        self.total_tokens += tokens
        parsed = _extract_json(content)
        if config.LLM_CACHE:
            self._cache_write(key, parsed, content)
        return parsed


_SINGLETON: LLMClient | None = None


def get_llm() -> LLMClient:
    """进程内单例：全流程共享同一组统计量与缓存句柄。"""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LLMClient()
    return _SINGLETON

