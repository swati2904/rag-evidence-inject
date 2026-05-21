"""Clean mask-hit diagnostic for TRIM-mask.

Rebuilds the clean (poison-free) retrieval pools used by the clean-control run
and applies the TRIM-mask rule set to every retrieved passage. Reports the
fraction of rows on which at least one mask rule fires. This is the diagnostic
referenced by the clean-utility control in the paper; it is computed from
public code only (no LLM is invoked) and can be reproduced from the public
repository.
"""
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
from defense.trim import apply_trim_to_docs
from load_config import load_config, pilot_example_counts
from retrieval.context_builder import retrieve_ids_bm25


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    p.add_argument(
        "--test-split",
        default=str(ROOT / "data" / "test_split.json"),
        help="Frozen split JSON listing the 200 example_ids.",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    data = cfg["data"]
    top_k = int(cfg["retrieval"]["top_k"])
    num_d = int(cfg["benchmark"]["num_distractors"])
    pn, ph = pilot_example_counts(data)
    examples = load_pilot_examples(
        n_nq=pn,
        n_hotpot=ph,
        seed=seed,
        use_kilt_nq=bool(data.get("use_kilt_nq", True)),
    )
    split_path = Path(args.test_split)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    split = json.loads(split_path.read_text(encoding="utf-8"))
    wanted = list(split.get("example_ids", []))
    by_id = {ex.example_id: ex for ex in examples if ex.example_id in set(wanted)}
    examples = [by_id[i] for i in wanted if i in by_id]
    rows = []
    n_rows_with_hit = 0
    total_hits = 0
    for ex in examples:
        rng = random.Random(seed + hash(ex.example_id) % 10000)
        docs = build_clean_pool(ex, num_distractors=num_d, rng=rng)
        bm_ids = retrieve_ids_bm25(docs, ex.question, top_k)
        id_to_doc = {d.doc_id: d for d in docs}
        retrieved = [id_to_doc[i] for i in bm_ids if i in id_to_doc]
        _, hits = apply_trim_to_docs(retrieved, mask_spans=True)
        if hits["mask_hits_total"] > 0:
            n_rows_with_hit += 1
        total_hits += hits["mask_hits_total"]
        rows.append({
            "example_id": ex.example_id,
            "mask_hits_total": hits["mask_hits_total"],
            "mask_hits_benign": hits["mask_hits_benign"],
        })
    n = len(rows)
    out = {
        "n_rows": n,
        "top_k": top_k,
        "defense": "trim_mask",
        "context": "clean (poison-free) pools, BM25 retrieval",
        "rows_with_any_mask_hit": n_rows_with_hit,
        "mask_hit_row_rate": n_rows_with_hit / max(1, n),
        "total_mask_hits": total_hits,
        "mean_mask_hits_per_row": total_hits / max(1, n),
        "per_row": rows,
    }
    log_dir = Path(cfg["paths"]["logs_dir"])
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    outp = log_dir / "clean_mask_hit_diagnostic.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {k: v for k, v in out.items() if k != "per_row"}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
