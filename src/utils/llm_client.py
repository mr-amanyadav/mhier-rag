# src/utils/llm_client.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import GROQ_API_KEY

_client = None
_working_model = None

# Priority-ordered. First one that your Groq account has access to wins.
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found in .env. Add: GROQ_API_KEY=gsk_..."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _pick_model():
    """Pick the first fallback model that this account has access to."""
    global _working_model
    if _working_model:
        return _working_model

    try:
        available = {m.id for m in get_client().models.list().data}
    except Exception:
        available = set()

    for m in FALLBACK_MODELS:
        if m in available:
            _working_model = m
            print(f"[llm_client] Using model: {m}", flush=True)
            return m

    # Last resort: probe each model with a tiny call
    for m in FALLBACK_MODELS:
        try:
            get_client().chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            _working_model = m
            print(f"[llm_client] Using model (probe): {m}", flush=True)
            return m
        except Exception:
            continue

    raise RuntimeError(
        f"No Groq model available from {FALLBACK_MODELS}. "
        f"Check https://console.groq.com and update FALLBACK_MODELS."
    )


def call_llm(prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
    """Send a prompt to Groq. Auto-selects an available model.

    Reasoning models (gpt-oss-*) spend tokens on internal reasoning before
    emitting output, so we enforce a minimum budget to avoid empty replies.
    """
    model = _pick_model()

    is_reasoning = any(k in model for k in ["gpt-oss", "o1", "o3", "o4"])
    effective_max = max(max_tokens, 400) if is_reasoning else max_tokens

    resp = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=effective_max,
    )
    return (resp.choices[0].message.content or "").strip()