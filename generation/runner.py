"""OpenAI-compatible client (vLLM) or HuggingFace causal LM fallback."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from generation.prompts import PromptBundle


@dataclass
class GenResult:
    text: str
    latency_s: float
    raw: dict[str, Any]


def generate_openai(
    bundle: PromptBundle,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> GenResult:
    t0 = time.perf_counter()
    client = OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(timeout=600.0))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": bundle.system},
            {"role": "user", "content": bundle.user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    dt = time.perf_counter() - t0
    return GenResult(text=text.strip(), latency_s=dt, raw={"model": model})


def generate_hf_fallback(
    bundle: PromptBundle,
    *,
    model_name: str,
    max_new_tokens: int,
) -> GenResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    messages = [
        {"role": "system", "content": bundle.system},
        {"role": "user", "content": bundle.user},
    ]
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    dt = time.perf_counter() - t0
    return GenResult(text=text.strip(), latency_s=dt, raw={"model": model_name, "backend": "hf"})
