"""JSON-only model client, versioned caches and measured per-call usage.

Only final response content is parsed. Private reasoning is not used as an
alternative prediction. Offline mode requires a matching cached response.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
import config

def _extract_json(text: str) -> dict:
    s = str(text or "").strip()
    fence = chr(96) * 3
    if s.startswith(fence):
        s = re.sub("^" + fence + r"(?:json)?\s*|\s*" + fence + "$", "", s, flags=re.I)
    try:
        value = json.loads(s)
    except ValueError as exc:
        raise ValueError("The model did not return a valid final JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from the model")
    return value

class LLMClient:
    def __init__(self):
        self.calls = self.cache_hits = self.total_tokens = 0
        self.logical_calls = self.network_requests = self.recorded_tokens = 0
        self._client = None

    def snapshot(self) -> dict:
        return {name: getattr(self, name) for name in
                ("calls", "cache_hits", "total_tokens", "logical_calls", "network_requests", "recorded_tokens")}

    def _cache_key(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        value = ["paper-aligned-v3", config.LLM_BASE_URL, config.LLM_MODEL,
                 system, user, max_tokens, temperature]
        return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()

    def _ensure_client(self):
        if self._client is None:
            if not config.LLM_API_KEY:
                raise RuntimeError("Set DEEPSEEK_API_KEY, or set LLM_OFFLINE=1 with compatible caches.")
            from openai import OpenAI
            self._client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                                  timeout=config.LLM_TIMEOUT, max_retries=0)
        return self._client

    def _raw_completion(self, system: str, user: str, max_tokens: int, temperature: float) -> tuple[str, int]:
        client = self._ensure_client()
        params = dict(model=config.LLM_MODEL,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                      temperature=temperature, max_tokens=max_tokens,
                      response_format={"type": "json_object"})
        last_error = None
        for attempt in range(config.LLM_MAX_RETRY):
            try:
                self.network_requests += 1
                response = client.chat.completions.create(**params)
                self.calls += 1
                tokens = int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)
                self.total_tokens += tokens
                if getattr(response.choices[0], "finish_reason", "") == "length":
                    raise ValueError("Model output was truncated. Increase LLM_MAX_TOKENS and rerun.")
                return response.choices[0].message.content or "", tokens
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                last_error = exc
                if "response_format" in str(exc).lower():
                    params.pop("response_format", None)
                if attempt + 1 < config.LLM_MAX_RETRY:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Model request failed after {config.LLM_MAX_RETRY} attempts") from last_error

    def call_json(self, system: str, user: str, max_tokens: int | None = None,
                  temperature: float | None = None) -> dict:
        max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        key = self._cache_key(system, user, max_tokens, temperature)
        path = config.CACHE_DIR / "llm" / (key + ".json")
        self.logical_calls += 1
        if config.LLM_CACHE and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            self.recorded_tokens += int(record.get("tokens", 0))
            return record["parsed"]
        if config.LLM_OFFLINE:
            raise RuntimeError(f"Offline model cache miss: {key}. Caches are not supplied with this source release.")
        raw, tokens = self._raw_completion(system, user, max_tokens, temperature)
        parsed = _extract_json(raw)
        self.recorded_tokens += tokens
        if config.LLM_CACHE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"parsed": parsed, "raw": raw, "tokens": tokens},
                                       ensure_ascii=False), encoding="utf-8")
        return parsed

_SINGLETON = None
def get_llm() -> LLMClient:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LLMClient()
    return _SINGLETON
