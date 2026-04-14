"""Load pilot/ablation JSON summaries and emit LaTeX table snippets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-summary", default=str(ROOT / "logs" / "pilot_summary.json"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "tables" / "pilot_metrics.tex"))
    args = ap.parse_args()
    p = Path(args.pilot_summary)
    if not p.exists():
        tex = (
            "% Populate after running scripts/run_pilot.py.\n"
            "\\begin{tabular}{lccc}\n\\hline\n"
            "Defense & ASR (rules) & EM \\\\\n\\hline\n"
            "none & -- & -- \\\\\n\\hline\n\\end{tabular}\n"
        )
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        summ = data.get("summary", {})
        byd = summ.get("by_defense", {})
        lines = [
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Defense & ASR (rules) & EM \\",
            r"\hline",
        ]
        for k, v in byd.items():
            lines.append(
                f"{k} & {v.get('asr', 0):.3f} & {v.get('em', 0):.3f} \\\\"
            )
        lines.extend([r"\hline", r"\end{tabular}", ""])
        tex = "\n".join(lines)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(tex, encoding="utf-8")
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
