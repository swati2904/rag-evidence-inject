"""Load pilot/ablation JSON summaries and emit LaTeX table snippets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _safe(name: str) -> str:
    return name.replace("_", r"\_")


def make_attack_table(title_map: dict[str, dict[str, float]]) -> str:
    """Attack-side per-defense table: ASR, EM, F1 (used when summaries include F1)."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Defense & ASR (rules) & EM & F1 \\",
        r"\hline",
    ]
    for name, vals in title_map.items():
        lines.append(
            f"{_safe(name)} & {vals.get('asr', 0):.3f} & "
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


def make_family_table(by_family: dict[str, dict[str, float]]) -> str:
    """Per-attack-family table: n, ASR, EM."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Attack family & $n$ & ASR (rules) & EM \\",
        r"\hline",
    ]
    for name, vals in by_family.items():
        lines.append(
            f"{_safe(name)} & {int(vals.get('n', 0))} & "
            f"{vals.get('asr', 0):.3f} & {vals.get('em', 0):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_rank_table(by_rank: dict[str, dict[str, float]]) -> str:
    """Per-poison-rank table: n, ASR, EM."""
    lines = [
        r"\begin{tabular}{cccc}",
        r"\hline",
        r"Poison rank & $n$ & ASR (rules) & EM \\",
        r"\hline",
    ]
    for name, vals in sorted(by_rank.items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"{name} & {int(vals.get('n', 0))} & "
            f"{vals.get('asr', 0):.3f} & {vals.get('em', 0):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def make_template_table(by_template: dict[str, dict[str, float]]) -> str:
    """Per-template table: family, template, n, undefended ASR, overall ASR, EM.

    Rows are grouped by family, then sorted within family by descending
    undefended ASR so the most-dangerous template per family appears first.
    """
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Family & Template & $n$ & ASR (none) & ASR (all) & EM \\",
        r"\hline",
    ]
    items = list(by_template.items())
    items.sort(
        key=lambda kv: (
            kv[1].get("family", ""),
            -float(kv[1].get("asr_undefended", 0.0)),
        )
    )
    for name, vals in items:
        lines.append(
            f"{_safe(vals.get('family', ''))} & {_safe(name)} & "
            f"{int(vals.get('n', 0))} & "
            f"{vals.get('asr_undefended', 0):.3f} & "
            f"{vals.get('asr', 0):.3f} & "
            f"{vals.get('em', 0):.3f} \\\\"
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
        "--clean-summary",
        default=str(ROOT / "logs" / "pilot_clean_summary.json"),
    )
    ap.add_argument(
        "--clean-out",
        default=str(ROOT / "paper" / "tables" / "pilot_clean_metrics.tex"),
    )
    ap.add_argument("--ablation-summary", default=str(ROOT / "logs" / "ablation_summary.json"))
    ap.add_argument("--ablation-out", default=str(ROOT / "paper" / "tables" / "ablation_metrics.tex"))
    args = ap.parse_args()

    write_if_exists(
        Path(args.pilot_summary),
        Path(args.pilot_out),
        lambda d: make_attack_table(d.get("summary", {}).get("by_defense", {})),
        _placeholder("lccc", "Defense & ASR (rules) & EM & F1"),
    )
    write_if_exists(
        Path(args.pilot_summary),
        Path(args.pilot_family_out),
        lambda d: make_family_table(d.get("summary", {}).get("by_family", {})),
        _placeholder("lccc", "Attack family & $n$ & ASR (rules) & EM"),
    )
    write_if_exists(
        Path(args.pilot_summary),
        Path(args.pilot_rank_out),
        lambda d: make_rank_table(d.get("summary", {}).get("by_rank", {})),
        _placeholder("cccc", "Poison rank & $n$ & ASR (rules) & EM"),
    )
    write_if_exists(
        Path(args.pilot_summary),
        Path(args.pilot_template_out),
        lambda d: make_template_table(d.get("summary", {}).get("by_template", {})),
        _placeholder(
            "llcccc",
            "Family & Template & $n$ & ASR (none) & ASR (all) & EM",
        ),
    )
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