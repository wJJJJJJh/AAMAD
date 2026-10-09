"""全局配置"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
TRAIN_XML = DATA_DIR / "train" / "XML"
TRAIN_PDF = DATA_DIR / "train" / "PDF"
TEST_XML = DATA_DIR / "test" / "XML"
TEST_PDF = DATA_DIR / "test" / "PDF"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"

CACHE_DIR = ROOT / ".cache"
OUT_DIR = ROOT / "outputs"
for _d in (CACHE_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 模型
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_API_KEY = DEEPSEEK_API_KEY  # 别名，供llm/client.py使用
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
LLM_BASE_URL = DEEPSEEK_BASE_URL  # 别名
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.0  # 默认温度
LLM_CACHE = True  # 启用LLM缓存
LLM_OFFLINE = False  # 是否离线模式
LLM_TIMEOUT = 60  # API超时时间（秒）
LLM_MAX_RETRY = 3  # 最大重试次数

# ---------------------------------------------------------------- 多轮辩论参数（AAMAD）
DEBATE_MAX_ROUNDS = 5  # 最多辩论轮次
DEBATE_CONVERGENCE_GAP = 0.3  # 收敛阈值
DEBATE_FINAL_TIE_POLICY = "EMIT"  # 保召回策略

# 动态增长策略：temperature和max_tokens每轮递增，增量指数递减
# Round 1: 基础值（快速判断明显案例）
# Round 2-5: 逐步增加，探索更复杂的论证空间，增量减半
DEBATE_ROUND_PARAMS = {
    1: {"temperature": 0.0, "max_tokens": 512, "increment_t": 0.0, "increment_m": 0},
    2: {"temperature": 0.1, "max_tokens": 640, "increment_t": 0.1, "increment_m": 128},
    3: {"temperature": 0.15, "max_tokens": 704, "increment_t": 0.05, "increment_m": 64},
    4: {"temperature": 0.175, "max_tokens": 736, "increment_t": 0.025, "increment_m": 32},
    5: {"temperature": 0.19, "max_tokens": 752, "increment_t": 0.015, "increment_m": 16},
}

# ---------------------------------------------------------------- 消融实验开关
ABLATION = {}

ABLATION_PRESETS = {
    "discovery_only": {"SKIP_DEBATE": True, "SKIP_CLASSIFY": True},
    "conventional_debate": {"USE_STRUCTURED_DEBATE": False},
    "AAMAD": {"USE_STRUCTURED_DEBATE": True},
}

def apply_ablation(name: str) -> dict:
    """应用消融预设"""
    if name not in ABLATION_PRESETS:
        raise ValueError(f"未知ablation预设: {name}")
    ABLATION.clear()
    ABLATION.update(ABLATION_PRESETS[name])
    return dict(ABLATION)

# ---------------------------------------------------------------- 其他参数
MAX_FULLTEXT_CHARS = 50000
BULK_FILTER_THRESHOLD = 50
