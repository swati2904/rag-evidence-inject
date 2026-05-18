"""Load pilot/ablation JSON summaries and emit LaTeX table snippets.

By default, the primary tables are computed on the **evidence-present
subset** of pilot rows -- rows where both ``gold_in_topk`` and
``poison_in_topk`` are true -- so the headline numbers match the paper's
threat model. Pass ``--filter-gold-in-topk all`` to fall back to the
pre-aggregated ``summary.by_*`` blocks (the older "all rows" view) for
appendix comparisons.

Confidence intervals reported in the primary tables are
question-clustered bootstrap intervals; per-row Wilson intervals are kept
as a comparator in the rank/family tables when n is small.
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

from evaluation.metrics import (  # noqa: E402
    bootstrap_question_clustered_ci,
    wilson_ci,
)


def _safe(name: str) -> str:
    return name.replace("_", r"\_")


def _fmt_rate_ci(rate: float, n: int) -> str:
    """Render ``rate`` with a Wilson 95% CI as ``0.296 (0.269, 0.325)``."""
    if n <= 0:
        return f"{rate:.3f}"
    k = int(round(rate * n))
    lo, hi = wilson_ci(k, n)
    return f"{rate:.3f} ({lo:.3f}, {hi:.3f})"


def _fmt_rate_clustered(rate: float, lo: float, hi: float) -> str:
    return f"{rate:.3f} ({lo:.3f}, {hi:.3f})"


def _filter_rows(rows: list[dict], mode: str) -> list[dict]:
    """Filter the per-row log for the requested analysis subset.

    Modes:
        ``evidence``  - keep rows where ``gold_in_topk`` AND ``poison_in_topk``.
                        This is the primary evidence-present subset and the
                        default reported in the paper body.
        ``gold``      - keep rows where ``gold_in_topk`` only (no poison
                        constraint). Useful for the clean-utility control
                        when poison_in_topk is irrelevant.
        ``all``       - no filtering; report on every row.
    """
    if mode == "all":
        return rows
    if mode == "gold":
        return [r for r in rows if r.get("gold_in_topk", False)]
    if mode == "evidence":
        return [
            r
            for r in rows
            if r.get("gold_in_topk", False) and r.get("poison_in_topk", False)
        ]
    raise ValueError(f"unknown filter mode: {mode}")


def _aggregate_from_rows(
    rows: list[dict],
    group_keys: Iterable[str] | None,
) -> dict:
    """Re-aggregate per-row records into the same shape ``summary.by_*`` uses.

    If ``group_keys`` is None, returns the overall aggregate. Otherwise,
    returns a nested dict keyed by the cross-product of ``group_keys`` values
    that appear in the data.
    """
    def _stats(sub: list[dict]) -> dict:
        n = len(sub)
        if n == 0:
            return {"n": 0, "asr": 0.0, "em": 0.0, "f1": 0.0}
        return {
            "n": n,
            "asr": sum(r["asr_rules"] for r in sub) / n,
            "em": sum(r["exact_match"] for r in sub) / n,
            "f1": sum(r.get("f1", 0.0) for r in sub) / n,
        }

    if not group_keys:
        return _stats(rows)
    keys = list(group_keys)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[tuple(r.get(k) for k in keys)].append(r)
    return {keys: _stats(sub) for keys, sub in buckets.items()}


def _per_defense(rows: list[dict], defenses: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in defenses:
        sub = [r for r in rows if r["defense"] == d]
        out[d] = {
            "n": len(sub),
            "asr": sum(r["asr_rules"] for r in sub) / max(1, len(sub)),
            "em": sum(r["exact_match"] for r in sub) / max(1, len(sub)),
            "f1": sum(r.get("f1", 0.0) for r in sub) / max(1, len(sub)),
            "_rows": sub,
        }
    return out


def make_attack_table_clustered(
    rows: list[dict],
    defenses: list[str],
) -> str:
    """Per-defense ASR with question-clustered bootstrap CIs, plus EM and F1."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Defense & ASR (rules; clustered 95\% CI) & EM & F1 \\",
        r"\hline",
    ]
    for d in defenses:
        sub = [r for r in rows if r["defense"] == d]
        if not sub:
            lines.append(f"{_safe(d)} & -- & -- & -- \\\\")
            continue
        rate, lo, hi = bootstrap_question_clustered_ci(
            [r["asr_rules"] for r in sub],
            [r["example_id"] for r in sub],
        )
        em = sum(r["exact_match"] for r in sub) / len(sub)
        f1 = sum(r.get("f1", 0.0) for r in sub) / len(sub)
        lines.append(
            f"{_safe(d)} & {_fmt_rate_clustered(rate, lo, hi)} & "
            f"{em:.3f} & {f1:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_attack_table_legacy(title_map: dict[str, dict[str, float]]) -> str:
    """Backwards-compatible per-defense table from a pre-aggregated dict."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Defense & ASR (rules; Wilson 95\% CI) & EM & F1 \\",
        r"\hline",
    ]
    for name, vals in title_map.items():
        n = int(vals.get("n", 0))
        lines.append(
            f"{_safe(name)} & {_fmt_rate_ci(vals.get('asr', 0), n)} & "
            f"{vals.get('em', 0):.3f} & {vals.get('f1', 0):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_ablation_table(title_map: dict[str, dict[str, float]]) -> str:
    """Ablation table: ASR + EM only (the ablation runner does not record F1)."""
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Defense & ASR (rules) & EM \\",
        r"\hline",
    ]
    for name, vals in title_map.items():
        lines.append(
            f"{_safe(name)} & {vals.get('asr', 0):.3f} & {vals.get('em', 0):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_clean_table(title_map: dict[str, dict[str, float]]) -> str:
    """Matched clean-utility table: EM, F1, refusal rate (no attack)."""
    has_mask_hits = any("mask_hit_rate" in vals for vals in title_map.values())
    if has_mask_hits:
        lines = [
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Defense & EM & F1 & Refusal & Mask-hit rows \\",
            r"\hline",
        ]
        for name, vals in title_map.items():
            lines.append(
                f"{_safe(name)} & {vals.get('em', 0):.3f} & "
                f"{vals.get('f1', 0):.3f} & {vals.get('refusal_rate', 0):.3f} & "
                f"{vals.get('mask_hit_rate', 0):.3f} \\\\"
            )
        lines.extend([r"\hline", r"\end{tabular}", ""])
        return "\n".join(lines)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Defense & EM & F1 & Refusal \\",
        r"\hline",
    ]
    for name, vals in title_map.items():
        lines.append(
            f"{_safe(name)} & {vals.get('em', 0):.3f} & "
            f"{vals.get('f1', 0):.3f} & {vals.get('refusal_rate', 0):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_family_table(rows: list[dict]) -> str:
    """Per-attack-family table: n, ASR with clustered CI, EM."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Attack family & $n$ & ASR (rules; clustered 95\% CI) & EM \\",
        r"\hline",
    ]
    families = sorted({r["attack"] for r in rows})
    for fam in families:
        sub = [r for r in rows if r["attack"] == fam]
        if not sub:
            continue
        rate, lo, hi = bootstrap_question_clustered_ci(
            [r["asr_rules"] for r in sub],
            [r["example_id"] for r in sub],
        )
        em = sum(r["exact_match"] for r in sub) / len(sub)
        lines.append(
            f"{_safe(fam)} & {len(sub)} & "
            f"{_fmt_rate_clustered(rate, lo, hi)} & {em:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_rank_table(rows: list[dict]) -> str:
    """Per-poison-rank table: n, ASR with clustered CI, EM."""
    lines = [
        r"\begin{tabular}{cccc}",
        r"\hline",
        r"Poison rank & $n$ & ASR (rules; clustered 95\% CI) & EM \\",
        r"\hline",
    ]
    ranks = sorted({r["poison_rank"] for r in rows})
    for rk in ranks:
        sub = [r for r in rows if r["poison_rank"] == rk]
        if not sub:
            continue
        rate, lo, hi = bootstrap_question_clustered_ci(
            [r["asr_rules"] for r in sub],
            [r["example_id"] for r in sub],
        )
        em = sum(r["exact_match"] for r in sub) / len(sub)
        lines.append(
            f"{rk} & {len(sub)} & "
            f"{_fmt_rate_clustered(rate, lo, hi)} & {em:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_defense_rank_table(rows: list[dict], defenses: list[str]) -> str:
    """ASR matrix: rows = defenses (fixed order), columns = poison ranks."""
    ranks = sorted({r["poison_rank"] for r in rows})
    if not ranks:
        return _placeholder("lc", "Defense & ASR (rules)")
    cols = "l" + "c" * len(ranks)
    header = " & ".join(["Defense"] + [f"rank {r}" for r in ranks]) + r" \\"
    lines = [
        rf"\begin{{tabular}}{{{cols}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for d in defenses:
        cells = [_safe(d)]
        for rk in ranks:
            sub = [r for r in rows if r["defense"] == d and r["poison_rank"] == rk]
            if not sub:
                cells.append("--")
            else:
                asr = sum(r["asr_rules"] for r in sub) / len(sub)
                cells.append(f"{asr:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_template_table(rows: list[dict]) -> str:
    """Per-template table: family, template, n, undefended ASR, overall ASR, EM."""
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Family & Template & $n$ & ASR (none) & ASR (all defenses) & EM \\",
        r"\hline",
    ]
    by_template: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_template[r["attack_template"]].append(r)
    items = []
    for name, sub in by_template.items():
        fam = sub[0]["attack"] if sub else ""
        n = len(sub)
        asr_all = sum(r["asr_rules"] for r in sub) / max(1, n)
        em = sum(r["exact_match"] for r in sub) / max(1, n)
        undef = [r for r in sub if r["defense"] == "none"]
        asr_undef = (
            sum(r["asr_rules"] for r in undef) / len(undef) if undef else 0.0
        )
        items.append((fam, name, n, asr_undef, asr_all, em))
    items.sort(key=lambda x: (x[0], -x[3]))
    for fam, name, n, asr_undef, asr_all, em in items:
        lines.append(
            f"{_safe(fam)} & {_safe(name)} & {n} & "
            f"{asr_undef:.3f} & {asr_all:.3f} & {em:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def _placeholder(columns: str, header: str) -> str:
    return (
        "% Populate after running experiment script.\n"
        f"\\begin{{tabular}}{{{columns}}}\n\\hline\n"
        f"{header} \\\\\n\\hline\n"
        "none & -- & -- \\\\\n\\hline\n\\end{tabular}\n"
    )


def write_if_exists(src: Path, out: Path, extractor, placeholder: str) -> None:
    if not src.exists():
        tex = placeholder
    else:
        data = json.loads(src.read_text(encoding="utf-8"))
        tex = extractor(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")


DEFENSES = ["none", "reminder", "boundary", "trim", "trim_mask"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-summary", default=str(ROOT / "logs" / "pilot_summary.json"))
    ap.add_argument("--pilot-out", default=str(ROOT / "paper" / "tables" / "pilot_metrics.tex"))
    ap.add_argument(
        "--pilot-family-out",
        default=str(ROOT / "paper" / "tables" / "pilot_by_family.tex"),
    )
    ap.add_argument(
        "--pilot-rank-out",
        default=str(ROOT / "paper" / "tables" / "pilot_by_rank.tex"),
    )
    ap.add_argument(
        "--pilot-template-out",
        default=str(ROOT / "paper" / "tables" / "pilot_by_template.tex"),
    )
    ap.add_argument(
        "--pilot-defense-rank-out",
        default=str(ROOT / "paper" / "tables" / "pilot_by_defense_rank.tex"),
    )
    ap.add_argument(
        "--clean-summary",
        default=str(ROOT / "logs" / "pilot_clean_summary.json"),
    )
    ap.add_argument(
        "--clean-out",
        default=str(ROOT / "paper" / "tables" / "pilot_clean_metrics.tex"),
    )
    ap.add_argument("--ablation-summary", default=str(ROOT / "logs" / "ablation_summary.json"))
    ap.add_argument("--ablation-out", default=str(ROOT / "paper" / "tables" / "ablation_metrics.tex"))
    ap.add_argument(
        "--filter-gold-in-topk",
        choices=["evidence", "gold", "all"],
        default="evidence",
        help=(
            "Subset of pilot rows to aggregate over. 'evidence' (default) "
            "requires both gold_in_topk and poison_in_topk -- the threat-"
            "model-consistent subset for the paper body. 'gold' requires "
            "only gold_in_topk. 'all' uses the pre-aggregated summary "
            "(legacy behaviour) for appendix comparisons."
        ),
    )
    args = ap.parse_args()

    pilot_path = Path(args.pilot_summary)
    pilot_rows: list[dict] = []
    if pilot_path.exists():
        pilot_doc = json.loads(pilot_path.read_text(encoding="utf-8"))
        pilot_rows = pilot_doc.get("rows", []) or []
        if args.filter_gold_in_topk == "all" and not pilot_rows:
            # Fall back to legacy code path that reads summary.by_*.
            write_if_exists(
                pilot_path,
                Path(args.pilot_out),
                lambda d: make_attack_table_legacy(
                    d.get("summary", {}).get("by_defense", {})
                ),
                _placeholder("lccc", "Defense & ASR (rules) & EM & F1"),
            )
        else:
            subset = _filter_rows(pilot_rows, args.filter_gold_in_topk)
            print(
                f"pilot rows: total={len(pilot_rows)}, "
                f"after filter={len(subset)} (mode={args.filter_gold_in_topk})"
            )
            Path(args.pilot_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.pilot_out).write_text(
                make_attack_table_clustered(subset, DEFENSES),
                encoding="utf-8",
            )
            print(f"Wrote {args.pilot_out}")
            Path(args.pilot_family_out).write_text(
                make_family_table(subset), encoding="utf-8"
            )
            print(f"Wrote {args.pilot_family_out}")
            Path(args.pilot_rank_out).write_text(
                make_rank_table(subset), encoding="utf-8"
            )
            print(f"Wrote {args.pilot_rank_out}")
            Path(args.pilot_template_out).write_text(
                make_template_table(subset), encoding="utf-8"
            )
            print(f"Wrote {args.pilot_template_out}")
            Path(args.pilot_defense_rank_out).write_text(
                make_defense_rank_table(subset, DEFENSES), encoding="utf-8"
            )
            print(f"Wrote {args.pilot_defense_rank_out}")
    else:
        for out_path, header in [
            (args.pilot_out, "Defense & ASR (rules) & EM & F1"),
            (args.pilot_family_out, "Attack family & $n$ & ASR (rules) & EM"),
            (args.pilot_rank_out, "Poison rank & $n$ & ASR (rules) & EM"),
            (
                args.pilot_template_out,
                "Family & Template & $n$ & ASR (none) & ASR (all) & EM",
            ),
            (args.pilot_defense_rank_out, "Defense & ASR by rank"),
        ]:
            cols = (
                "lccc"
                if "Defense & ASR (rules) & EM" in header
                else "lccc"
                if header.startswith("Attack family")
                else "cccc"
                if header.startswith("Poison rank")
                else "llcccc"
                if header.startswith("Family")
                else "lc"
            )
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(
                _placeholder(cols, header), encoding="utf-8"
            )
            print(f"Wrote placeholder {out_path}")

    write_if_exists(
        Path(args.clean_summary),
        Path(args.clean_out),
        lambda d: make_clean_table(d.get("summary", {}).get("by_defense", {})),
        _placeholder("lccc", "Defense & EM & F1 & Refusal"),
    )
    write_if_exists(
        Path(args.ablation_summary),
        Path(args.ablation_out),
        lambda d: make_ablation_table(d.get("by_mode", {})),
        _placeholder("lcc", "Defense & ASR (rules) & EM"),
    )


if __name__ == "__main__":
    main()
