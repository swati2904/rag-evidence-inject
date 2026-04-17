"""Load pilot/ablation JSON summaries and emit LaTeX table snippets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_table(title_map: dict[str, dict[str, float]]) -> str:
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Defense & ASR (rules) & EM \\",
        r"\hline",
    ]
    for name, vals in title_map.items():
        safe_name = name.replace("_", r"\_")
        lines.append(f"{safe_name} & {vals.get('asr', 0):.3f} & {vals.get('em', 0):.3f} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_if_exists(src: Path, out: Path, extractor) -> None:
    if not src.exists():
        tex = (
            "% Populate after running experiment script.\n"
            "\\begin{tabular}{lcc}\n\\hline\n"
            "Defense & ASR (rules) & EM \\\\\n\\hline\n"
            "none & -- & -- \\\\\n\\hline\n\\end{tabular}\n"
        )
    else:
        data = json.loads(src.read_text(encoding="utf-8"))
        tex = make_table(extractor(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-summary", default=str(ROOT / "logs" / "pilot_summary.json"))
    ap.add_argument("--pilot-out", default=str(ROOT / "paper" / "tables" / "pilot_metrics.tex"))
    ap.add_argument("--ablation-summary", default=str(ROOT / "logs" / "ablation_summary.json"))
    ap.add_argument("--ablation-out", default=str(ROOT / "paper" / "tables" / "ablation_metrics.tex"))
    args = ap.parse_args()

    write_if_exists(
        Path(args.pilot_summary),
        Path(args.pilot_out),
        lambda d: d.get("summary", {}).get("by_defense", {}),
    )
    write_if_exists(
        Path(args.ablation_summary),
        Path(args.ablation_out),
        lambda d: d.get("by_mode", {}),
    )


if __name__ == "__main__":
    main()