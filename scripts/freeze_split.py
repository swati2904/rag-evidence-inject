"""Freeze the test split used by the current pilot run for reproducibility.

Reads ``logs/pilot_summary.json`` (or another pilot summary file), extracts
unique ``example_id`` values in stable order, and writes them to
``data/test_split.json``.

Run once after a pilot run you want to lock as the canonical evaluation split:

    python scripts/freeze_split.py

Then point future runs at the file via:

    python scripts/run_pilot.py --test-split data/test_split.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _unique_in_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pilot-summary",
        default=str(ROOT / "logs" / "pilot_summary.json"),
        help="Source pilot summary JSON whose example_ids define the split.",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "data" / "test_split.json"),
        help="Where to write the frozen split file.",
    )
    args = ap.parse_args()

    src = Path(args.pilot_summary)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)
    payload = json.loads(src.read_text(encoding="utf-8"))
    rows = payload.get("rows") or payload.get("rows_sample") or []
    if not rows:
        print("ERROR: no rows in pilot summary; nothing to freeze", file=sys.stderr)
        sys.exit(1)

    ids = _unique_in_order(r["example_id"] for r in rows)
    sources = sorted({"kilt_nq" if i.startswith("kilt_nq") else "hotpot" for i in ids})
    digest = hashlib.sha256(("|".join(ids)).encode("utf-8")).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n": len(ids),
                "sources": sources,
                "sha256": digest,
                "source_summary": str(src.relative_to(ROOT)),
                "example_ids": ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(ids)} ids; sha256={digest[:12]}...)")


if __name__ == "__main__":
    main()
