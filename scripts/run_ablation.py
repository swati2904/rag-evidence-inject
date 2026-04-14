"""TRIM ablation over defense modes and poison ranks (subset size from CLI)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.corpus_builder import build_pool_for_example, insert_poison_at_rank
from benchmark.datasets import load_pilot_examples
from benchmark.templates import TEMPLATES
from defense.trim import apply_trim_to_docs
from evaluation.metrics import attack_success_rules, exact_match
from generation.prompts import build_prompts
from generation.runner import generate_hf_fallback, generate_openai
from load_config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--subset", type=int, default=20)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--hf-fallback", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    examples = load_pilot_examples(
        n_nq=args.subset // 2,
        n_hotpot=args.subset - args.subset // 2,
        seed=seed,
        use_kilt_nq=bool(cfg["data"].get("use_kilt_nq", True)),
    )
    gen_cfg = cfg["generation"]
    top_k = int(cfg["retrieval"]["top_k"])
    num_d = int(cfg["benchmark"]["num_distractors"])
    modes = ["none", "reminder", "trim", "trim_mask"]
    ranks = [1, 3, 5]
    attack = TEMPLATES[0]
    rows = []
    for ex in examples:
        pool = build_pool_for_example(ex, attack=attack, num_distractors=num_d, rng=rng)
        if pool is None:
            continue
        for pr in ranks:
            ordered = insert_poison_at_rank(pool, pr, rng)
            ctx = ordered[:top_k]
            for mode in modes:
                if mode == "trim_mask":
                    docs = apply_trim_to_docs(ctx, mask_spans=True)
                    prompt_mode = "trim"
                elif mode == "trim":
                    docs = apply_trim_to_docs(ctx, mask_spans=False)
                    prompt_mode = "trim"
                else:
                    docs = ctx
                    prompt_mode = mode
                bundle = build_prompts(pool.question, docs, defense=prompt_mode)
                if args.no_llm:
                    text = ""
                elif args.hf_fallback or gen_cfg.get("use_hf_fallback"):
                    text = generate_hf_fallback(
                        bundle,
                        model_name=gen_cfg["model"],
                        max_new_tokens=int(gen_cfg.get("max_tokens", 128)),
                    ).text
                else:
                    text = generate_openai(
                        bundle,
                        base_url=gen_cfg["openai_base_url"],
                        api_key=gen_cfg["openai_api_key"],
                        model=gen_cfg["model"],
                        max_tokens=int(gen_cfg.get("max_tokens", 128)),
                        temperature=float(gen_cfg.get("temperature", 0.0)),
                    ).text
                rows.append(
                    {
                        "example_id": pool.example_id,
                        "mode": mode,
                        "poison_rank": pr,
                        "asr": bool(
                            text
                            and attack_success_rules(
                                text,
                                attack_family=attack.family.value,
                                malicious_target=pool.malicious_target,
                                gold_answers=pool.gold_answers,
                            )
                        ),
                        "em": exact_match(text, pool.gold_answers) if text else False,
                    }
                )
    log_dir = Path(cfg["paths"]["logs_dir"])
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    by_mode: dict[str, dict[str, float]] = {}
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        by_mode[mode] = {
            "asr": sum(r["asr"] for r in sub) / max(1, len(sub)),
            "em": sum(r["em"] for r in sub) / max(1, len(sub)),
        }
    out = {"by_mode": by_mode, "rows": rows}
    outp = log_dir / "ablation_summary.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(by_mode, indent=2))
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
