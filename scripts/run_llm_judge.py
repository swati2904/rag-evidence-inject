"""Run an LLM judge on a stratified sample of pilot rows; report agreement with rule-based ASR.

Usage (on a host that has logs/ and the OPENAI_API_KEY env var set):

    export OPENAI_API_KEY=sk-...
    python scripts/run_llm_judge.py \
        --pilot-summary logs/pilot_summary.json \
        --n 200 --model gpt-5.4-mini \
        --out logs/llm_judge.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from openai import OpenAI  # noqa: E402

from benchmark.datasets import load_pilot_examples  # noqa: E402
from evaluation.llm_judge import judge_row  # noqa: E402
from load_config import load_config, pilot_example_counts  # noqa: E402


def _build_response_map(jsonl_paths: list[Path]) -> dict[str, str]:
    """Walk all per-row JSONL files; index response previews by combined example_id."""
    out: dict[str, str] = {}
    for jp in jsonl_paths:
        with jp.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "generation":
                    continue
                ex_id = rec.get("example_id")
                if not ex_id:
                    continue
                out[ex_id] = rec.get("response_preview", "") or ""
    return out


def _stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["defense"], r["attack"])].append(r)
    n_per_cell = max(1, n // max(1, len(by_cell)))
    sampled: list[dict] = []
    seen_ids: set[int] = set()
    for cell, group in by_cell.items():
        rng.shuffle(group)
        for r in group[:n_per_cell]:
            sampled.append(r)
            seen_ids.add(id(r))
    if len(sampled) < n:
        leftover = [r for r in rows if id(r) not in seen_ids]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])
    return sampled


def _agreement_metrics(judged: list[dict]) -> dict:
    clear = [j for j in judged if j["judge_label"] in ("yes", "no")]
    n_clear = len(clear)
    if n_clear == 0:
        return {"error": "no clear judgments", "n_clear": 0, "n_total": len(judged)}
    rules = [bool(j["asr_rules"]) for j in clear]
    judge = [j["judge_label"] == "yes" for j in clear]
    tp = sum(1 for r, j in zip(rules, judge) if r and j)
    tn = sum(1 for r, j in zip(rules, judge) if not r and not j)
    # Treat the LLM judge as the reference label for this validation slice.
    # A false positive means the rule flags an attack the judge rejects; a
    # false negative means the judge sees an attack the rule misses.
    fp = sum(1 for r, j in zip(rules, judge) if r and not j)
    fn = sum(1 for r, j in zip(rules, judge) if not r and j)
    po = (tp + tn) / n_clear
    n2 = n_clear * n_clear
    pe_yes = ((tp + fp) * (tp + fn)) / n2 if n2 else 0.0
    pe_no = ((fn + tn) * (fp + tn)) / n2 if n2 else 0.0
    pe = pe_yes + pe_no
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "n_total": len(judged),
        "n_clear": n_clear,
        "n_unclear": len(judged) - n_clear,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": po,
        "cohen_kappa": kappa,
        "rules_precision_vs_judge": precision,
        "rules_recall_vs_judge": recall,
        "rules_f1_vs_judge": f1,
        "asr_judge": sum(judge) / n_clear,
        "asr_rules": sum(rules) / n_clear,
        "interpretation": (
            "rule ASR is a lower-bound proxy vs. this judge"
            if sum(rules) < sum(judge)
            else "rule ASR is an upper-bound proxy vs. this judge"
            if sum(rules) > sum(judge)
            else "rule ASR matches the judge-positive rate on this slice"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-summary", default=str(ROOT / "logs" / "pilot_summary.json"))
    ap.add_argument(
        "--jsonl-glob",
        default="logs/run_*.jsonl",
        help="Glob (relative to repo root) for per-row JSONL run logs",
    )
    ap.add_argument("--out", default=str(ROOT / "logs" / "llm_judge.json"))
    ap.add_argument("--n", type=int, default=200, help="stratified sample size")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    pilot_path = Path(args.pilot_summary)
    if not pilot_path.exists():
        print(f"ERROR: {pilot_path} not found", file=sys.stderr)
        sys.exit(1)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    rows = pilot.get("rows", [])
    if not rows:
        print("ERROR: no rows in pilot summary", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} rows from {pilot_path}")

    jsonl_paths = sorted(ROOT.glob(args.jsonl_glob))
    if not jsonl_paths:
        print(f"ERROR: no JSONL run logs at {args.jsonl_glob}", file=sys.stderr)
        sys.exit(1)
    response_map = _build_response_map(jsonl_paths)
    print(f"Indexed {len(response_map)} responses from {len(jsonl_paths)} JSONL file(s)")

    cfg = load_config(args.config)
    data = cfg["data"]
    n_nq, n_hp = pilot_example_counts(data)
    examples = load_pilot_examples(
        n_nq=n_nq,
        n_hotpot=n_hp,
        seed=int(cfg.get("seed", 42)),
        use_kilt_nq=bool(data.get("use_kilt_nq", True)),
    )
    ex_by_id = {ex.example_id: ex for ex in examples}
    print(f"Loaded {len(examples)} pilot examples")

    rng = random.Random(args.seed)
    sampled = _stratified_sample(rows, args.n, rng)
    print(f"Stratified-sampled {len(sampled)} rows for judging")

    client = OpenAI(api_key=api_key, http_client=httpx.Client(timeout=120.0))

    judged: list[dict] = []
    skipped = 0
    resolved_models: dict[str, int] = {}
    for i, r in enumerate(sampled):
        ex = ex_by_id.get(r["example_id"])
        if ex is None:
            skipped += 1
            continue
        key = f"{r['example_id']}|pr{r['poison_rank']}|{r['defense']}"
        response = response_map.get(key, "")
        if not response:
            skipped += 1
            continue
        label, raw, resolved = judge_row(
            client,
            args.model,
            question=ex.question,
            response=response,
            gold=ex.gold_answers,
            attack_family=r["attack"],
        )
        if resolved:
            resolved_models[resolved] = resolved_models.get(resolved, 0) + 1
        judged.append(
            {
                "example_id": r["example_id"],
                "attack": r["attack"],
                "attack_template": r.get("attack_template", ""),
                "defense": r["defense"],
                "poison_rank": r["poison_rank"],
                "asr_rules": bool(r["asr_rules"]),
                "exact_match": bool(r.get("exact_match", False)),
                "question": ex.question,
                "gold": ex.gold_answers,
                "response_preview": response,
                "judge_label": label,
                "judge_raw": raw,
                "model_resolved": resolved,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(sampled)} (skipped {skipped})")

    agreement = _agreement_metrics(judged)
    model_resolved_majority = (
        max(resolved_models.items(), key=lambda kv: kv[1])[0] if resolved_models else ""
    )
    output = {
        "model": args.model,
        "model_resolved": model_resolved_majority,
        "model_resolved_counts": resolved_models,
        "n_judged": len(judged),
        "n_skipped": skipped,
        "agreement": agreement,
        "rows": judged,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": args.model, "agreement": agreement}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
