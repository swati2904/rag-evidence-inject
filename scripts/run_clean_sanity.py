"""Gold-in-top-k retrieval check (BM25; optional dense). No LLM.

Runs over the full pilot split by default so the reported recall is on the
same 200 questions the main evaluation uses. Per-dataset and per-source
breakdowns are emitted so the paper can show retrieval recall is not the
bottleneck on the evidence-present subset.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.corpus_builder import build_clean_pool
from benchmark.datasets import load_pilot_examples
from evaluation.metrics import gold_in_topk
from load_config import load_config, pilot_example_counts
from retrieval.context_builder import retrieve_ids_bm25, retrieve_ids_dense


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Cap the number of pilot examples to score. The default of 0 means "
            "'use the full pilot split from the config' (200 examples in the "
            "standard pilot.yaml), so the sanity check covers the same "
            "questions the main evaluation uses."
        ),
    )
    p.add_argument("--dense", action="store_true", help="Also run dense retrieval (loads embedding model)")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--test-split",
        default=None,
        help=(
            "Optional path to a frozen split JSON (see scripts/freeze_split.py). "
            "When provided, loaded examples are filtered to the listed "
            "example_ids in the split's stable order."
        ),
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    data = cfg["data"]
    top_k = int(cfg["retrieval"]["top_k"])
    num_d = int(cfg["benchmark"]["num_distractors"])
    pn, ph = pilot_example_counts(data)
    if args.limit > 0:
        half = max(1, args.limit // 2)
        n_nq = min(pn, half)
        n_hp = min(ph, max(1, args.limit - half))
    else:
        n_nq, n_hp = pn, ph
    examples = load_pilot_examples(
        n_nq=n_nq,
        n_hotpot=n_hp,
        seed=seed,
        use_kilt_nq=bool(data.get("use_kilt_nq", True)),
    )
    if args.test_split:
        split_path = Path(args.test_split)
        if not split_path.is_absolute():
            split_path = ROOT / split_path
        split = json.loads(split_path.read_text(encoding="utf-8"))
        wanted = list(split.get("example_ids", []))
        wanted_set = set(wanted)
        by_id = {ex.example_id: ex for ex in examples if ex.example_id in wanted_set}
        examples = [by_id[i] for i in wanted if i in by_id]
    rows = []
    for ex in examples:
        rng = random.Random(seed + hash(ex.example_id) % 10000)
        docs = build_clean_pool(ex, num_distractors=num_d, rng=rng)
        gold_ids = {d.doc_id for d in docs if d.role == "gold"}
        bm_ids = retrieve_ids_bm25(docs, ex.question, top_k)
        row = {
            "example_id": ex.example_id,
            "source": ex.source,
            "bm25_gold_in_topk": gold_in_topk(gold_ids, bm_ids),
        }
        if args.dense:
            dn = cfg["retrieval"]["dense_model"]
            d_ids = retrieve_ids_dense(docs, ex.question, top_k, dn, args.device)
            row["dense_gold_in_topk"] = gold_in_topk(gold_ids, d_ids)
        rows.append(row)
    n = len(rows)
    bm_hits = sum(1 for r in rows if r["bm25_gold_in_topk"])
    by_source: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "bm25_hits": 0, "dense_hits": 0}
    )
    for r in rows:
        src = r.get("source", "unknown")
        by_source[src]["n"] += 1
        if r["bm25_gold_in_topk"]:
            by_source[src]["bm25_hits"] += 1
        if r.get("dense_gold_in_topk"):
            by_source[src]["dense_hits"] += 1
    by_source_clean: dict[str, dict[str, float]] = {}
    for src, vals in by_source.items():
        ns = vals["n"]
        by_source_clean[src] = {
            "n": int(ns),
            "bm25_recall_at_k": vals["bm25_hits"] / max(1, ns),
        }
        if args.dense:
            by_source_clean[src]["dense_recall_at_k"] = (
                vals["dense_hits"] / max(1, ns)
            )
    out = {
        "n": n,
        "top_k": top_k,
        "bm25_recall_at_k": bm_hits / max(1, n),
        "by_source": by_source_clean,
        # Keep the full per-example list now that we report over the whole
        # split: downstream analyses (per-rank, per-defense filtering) can
        # join on example_id.
        "per_example": rows,
    }
    if args.dense:
        out["dense_recall_at_k"] = sum(
            1 for r in rows if r.get("dense_gold_in_topk")
        ) / max(1, n)
    log_dir = Path(cfg["paths"]["logs_dir"])
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    outp = log_dir / "clean_sanity_summary.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_example"}, indent=2))
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
