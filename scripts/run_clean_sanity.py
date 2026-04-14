"""Gold-in-top-k retrieval check (BM25; optional dense). No LLM."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.corpus_builder import build_clean_pool
from benchmark.datasets import load_pilot_examples
from evaluation.metrics import gold_in_topk
from load_config import load_config
from retrieval.context_builder import retrieve_ids_bm25, retrieve_ids_dense


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--dense", action="store_true", help="Also run dense retrieval (loads embedding model)")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    data = cfg["data"]
    top_k = int(cfg["retrieval"]["top_k"])
    num_d = int(cfg["benchmark"]["num_distractors"])
    pn = int(data.get("pilot_nq", 100))
    ph = int(data.get("pilot_hotpot", 100))
    half = max(1, args.limit // 2)
    examples = load_pilot_examples(
        n_nq=min(pn, half),
        n_hotpot=min(ph, max(1, args.limit - half)),
        seed=seed,
        use_kilt_nq=bool(data.get("use_kilt_nq", True)),
    )
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
    out = {
        "n": n,
        "bm25_recall_at_k": bm_hits / max(1, n),
        "per_example": rows[:20],
    }
    if args.dense:
        out["dense_recall_at_k"] = sum(1 for r in rows if r.get("dense_gold_in_topk")) / max(1, n)
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
