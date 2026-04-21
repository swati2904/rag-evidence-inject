"""Pilot benchmark: attacks, defenses, and metrics; LLM controlled by CLI flags."""
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
from evaluation.metrics import attack_success_rules, exact_match, f1_max
from evaluation.run_logger import RunLogger, StageTimer
from generation.prompts import build_prompts
from generation.runner import generate_hf_fallback, generate_openai
from load_config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--no-llm", action="store_true", help="Skip generation; ASR rule-based scores on empty output")
    ap.add_argument("--hf-fallback", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = use config pilot sizes")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    data = cfg["data"]
    n_nq = int(data.get("pilot_nq", 100))
    n_hp = int(data.get("pilot_hotpot", 100))
    if args.max_examples > 0:
        n_nq = min(n_nq, args.max_examples // 2)
        n_hp = min(n_hp, args.max_examples - n_nq)
    examples = load_pilot_examples(
        n_nq=n_nq,
        n_hotpot=n_hp,
        seed=seed,
        use_kilt_nq=bool(data.get("use_kilt_nq", True)),
    )
    log_dir = Path(cfg["paths"]["logs_dir"])
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    logger = RunLogger(log_dir)
    gen_cfg = cfg["generation"]
    defenses = ["none", "reminder", "boundary", "trim", "trim_mask"]
    poison_ranks = list(cfg["benchmark"].get("poison_rank_candidates", [1, 5]))
    num_d = int(cfg["benchmark"]["num_distractors"])
    top_k = int(cfg["retrieval"]["top_k"])
    rows = []
    for ex in examples:
        attack = TEMPLATES[rng.randint(0, len(TEMPLATES) - 1)]
        pool = build_pool_for_example(ex, attack=attack, num_distractors=num_d, rng=rng)
        if pool is None:
            continue
        for pr in poison_ranks:
            ordered = insert_poison_at_rank(pool, pr, rng)
            ctx = ordered[:top_k]
            gold_in_topk = any(d.role == "gold" for d in ctx)
            poison_in_topk = any(d.role == "poison" for d in ctx)
            observed_poison_rank = next(
                (i + 1 for i, d in enumerate(ctx) if d.role == "poison"), None
            )
            for defense in defenses:
                docs = ctx
                mask_hits = 0
                if defense == "trim_mask":
                    docs, mask_hits = apply_trim_to_docs(ctx, mask_spans=True)
                    defense_prompt = "trim"
                elif defense == "trim":
                    docs, mask_hits = apply_trim_to_docs(ctx, mask_spans=False)
                    defense_prompt = "trim"
                else:
                    defense_prompt = defense
                bundle = build_prompts(pool.question, docs, defense=defense_prompt)
                timer = StageTimer()
                if args.no_llm:
                    text = ""
                    lat = 0.0
                elif args.hf_fallback or gen_cfg.get("use_hf_fallback"):
                    res = generate_hf_fallback(
                        bundle,
                        model_name=gen_cfg["model"],
                        max_new_tokens=int(gen_cfg.get("max_tokens", 256)),
                    )
                    text, lat = res.text, res.latency_s
                else:
                    res = generate_openai(
                        bundle,
                        base_url=gen_cfg["openai_base_url"],
                        api_key=gen_cfg["openai_api_key"],
                        model=gen_cfg["model"],
                        max_tokens=int(gen_cfg.get("max_tokens", 256)),
                        temperature=float(gen_cfg.get("temperature", 0.0)),
                    )
                    text, lat = res.text, res.latency_s
                timer.split()
                asr = (
                    attack_success_rules(
                        text,
                        attack_family=pool.attack_family,
                        malicious_target=pool.malicious_target,
                        gold_answers=pool.gold_answers,
                    )
                    if text
                    else False
                )
                em = exact_match(text, pool.gold_answers) if text else False
                f1 = f1_max(text, pool.gold_answers) if text else 0.0
                row = {
                    "example_id": pool.example_id,
                    "attack": attack.family.value,
                    "attack_template": attack.name,
                    "poison_rank": pr,
                    "observed_poison_rank": observed_poison_rank,
                    "gold_in_topk": gold_in_topk,
                    "poison_in_topk": poison_in_topk,
                    "defense": defense,
                    "mask_hits": mask_hits,
                    "exact_match": em,
                    "f1": f1,
                    "asr_rules": asr,
                    "latency_s": lat,
                }
                rows.append(row)
                logger.log_generation(
                    example_id=f"{pool.example_id}|pr{pr}|{defense}",
                    prompt=bundle.system + "\n\n" + bundle.user,
                    response=text,
                    retrieved_doc_ids=[d.doc_id for d in docs],
                    ranks=list(range(1, len(docs) + 1)),
                    latency_s=lat,
                    extra={"attack": attack.family.value, "defense": defense},
                )
    logger.close()
    def _mean(xs: list, key: str) -> float:
        return sum(x[key] for x in xs) / max(1, len(xs))

    summary = {
        "n_rows": len(rows),
        "mean_asr": _mean(rows, "asr_rules"),
        "mean_em": _mean(rows, "exact_match"),
        "mean_f1": _mean(rows, "f1"),
        "gold_in_topk_rate": _mean(rows, "gold_in_topk"),
        "poison_in_topk_rate": _mean(rows, "poison_in_topk"),
        "by_defense": {},
        "by_family": {},
        "by_rank": {},
        "by_defense_family": {},
        "mask_hits": {
            "rows_with_any_hit": sum(1 for r in rows if r.get("mask_hits", 0) > 0),
            "rows_trim_mask": sum(1 for r in rows if r["defense"] == "trim_mask"),
            "mean_hits_trim_mask": (
                sum(r.get("mask_hits", 0) for r in rows if r["defense"] == "trim_mask")
                / max(1, sum(1 for r in rows if r["defense"] == "trim_mask"))
            ),
        },
    }
    for d in defenses:
        sub = [r for r in rows if r["defense"] == d]
        summary["by_defense"][d] = {
            "n": len(sub),
            "asr": _mean(sub, "asr_rules"),
            "em": _mean(sub, "exact_match"),
            "f1": _mean(sub, "f1"),
        }
    families = sorted({r["attack"] for r in rows})
    for fam in families:
        sub = [r for r in rows if r["attack"] == fam]
        summary["by_family"][fam] = {
            "n": len(sub),
            "asr": _mean(sub, "asr_rules"),
            "em": _mean(sub, "exact_match"),
        }
    for r_k in sorted({r["poison_rank"] for r in rows}):
        sub = [r for r in rows if r["poison_rank"] == r_k]
        summary["by_rank"][str(r_k)] = {
            "n": len(sub),
            "asr": _mean(sub, "asr_rules"),
            "em": _mean(sub, "exact_match"),
        }
    for d in defenses:
        summary["by_defense_family"][d] = {}
        for fam in families:
            sub = [r for r in rows if r["defense"] == d and r["attack"] == fam]
            summary["by_defense_family"][d][fam] = {
                "n": len(sub),
                "asr": _mean(sub, "asr_rules"),
                "em": _mean(sub, "exact_match"),
            }
    outp = log_dir / "pilot_summary.json"
    outp.write_text(
        json.dumps(
            {"rows_sample": rows[:30], "summary": summary, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
