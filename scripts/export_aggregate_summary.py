"""Export a GitHub-readable aggregate view of the main pilot summary.

The full ``logs/pilot_summary.json`` file intentionally contains every
per-row generation result, so it is large enough that GitHub's web viewer may
not render it. This script extracts the headline evidence-present aggregates
into a small companion JSON file while leaving the full per-row log unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import bootstrap_question_clustered_ci
from load_config import load_config

DEFENSE_ORDER = ["none", "reminder", "boundary", "trim", "trim_mask"]


def _repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _filter_evidence_present(rows: list[dict]) -> list[dict]:
    return [
        r
        for r in rows
        if r.get("gold_in_topk", False) and r.get("poison_in_topk", False)
    ]


def _stats(rows: list[dict], *, include_ci: bool = False) -> dict:
    n = len(rows)
    if n == 0:
        out: dict[str, object] = {"n": 0, "asr": 0.0, "em": 0.0, "f1": 0.0}
    else:
        out = {
            "n": n,
            "asr": sum(r["asr_rules"] for r in rows) / n,
            "em": sum(r["exact_match"] for r in rows) / n,
            "f1": sum(r.get("f1", 0.0) for r in rows) / n,
        }
    if include_ci:
        _, lo, hi = bootstrap_question_clustered_ci(
            [r["asr_rules"] for r in rows],
            [r["example_id"] for r in rows],
        )
        out["asr_ci"] = [lo, hi]
    return out


def _group_stats(
    rows: list[dict],
    key: str,
    *,
    order: Iterable[str] | None = None,
    include_ci: bool = False,
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get(key))].append(r)
    keys = list(order) if order is not None else sorted(grouped)
    return {
        k: _stats(grouped[k], include_ci=include_ci)
        for k in keys
        if k in grouped
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    p.add_argument(
        "--pilot-summary",
        default=str(ROOT / "logs" / "pilot_summary.json"),
        help="Full per-row pilot summary JSON.",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "logs" / "pilot_summary_aggregate.json"),
        help="Small aggregate JSON to write.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    source_path = Path(args.pilot_summary)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    data = json.loads(source_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    evidence_rows = _filter_evidence_present(rows)
    by_defense = _group_stats(
        evidence_rows,
        "defense",
        order=DEFENSE_ORDER,
        include_ci=True,
    )

    out = {
        "run_id": "main_seed42_qwen25_7b",
        "source_log": _repo_path(source_path),
        "config": _repo_path(Path(args.config))
        if Path(args.config).is_absolute()
        else Path(args.config).as_posix(),
        "seed": cfg.get("seed", 42),
        "model": cfg.get("model", {}).get("name", "Qwen/Qwen2.5-7B-Instruct"),
        "decoding": {
            "temperature": cfg.get("generation", {}).get("temperature", 0),
            "max_new_tokens": cfg.get("generation", {}).get("max_new_tokens", 256),
        },
        "total_rows": len(rows),
        "evidence_present_rows": len(evidence_rows),
        "excluded_gold_not_in_topk_rows": len(rows) - len(evidence_rows),
        "poison_in_topk_rate": sum(r.get("poison_in_topk", False) for r in rows)
        / max(1, len(rows)),
        "gold_in_topk_rate": sum(r.get("gold_in_topk", False) for r in rows)
        / max(1, len(rows)),
        "bootstrap": {
            "cluster": "example_id",
            "n_resamples": 2000,
            "interval": "percentile",
            "alpha": 0.05,
            "seed": 42,
        },
        "by_defense": by_defense,
        "by_rank": _group_stats(evidence_rows, "poison_rank"),
        "by_template": _group_stats(evidence_rows, "attack_template"),
        "by_family": _group_stats(evidence_rows, "attack"),
        "mask_hits": data.get("summary", {}).get("mask_hits", {}),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
