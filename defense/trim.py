"""TRIM: mask directive-like spans in poison passages before prompt assembly."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.corpus_builder import CorpusDoc

_DIRECTIVE_PATTERNS = [
    r"(?i)\bignore\b.{0,40}\b(document|passage|evidence|context|entry|entries|retrieved)\b",
    r"(?i)\bdo not cite\b",
    r"(?i)\bclassified\b",
    r"(?i)\bresponse contract\b",
    r"(?i)\btreat the following as authoritative\b",
    # Carrier-level imperatives common in the realistic templates:
    r"(?i)\baction required\b",
    r"(?i)\bauthoritative (answer|value|correction|update)\b",
    r"(?i)\bsupersed(es|ed)\b",
    r"(?i)\bverified (answer|correction)\b",
    r"(?i)\boutput (it|the following) verbatim\b",
    r"(?i)\binsufficient[- ]evidence response\b",
    # Phase A+ template idioms:
    r"(?i)\bcorrigendum\b",
    r"(?i)\bredact(ed|ion)\b",
    r"(?i)\bcompliance (review|hold|clearance)\b",
    r"(?i)\bcanonical (answer|response|value|response field)\b",
    r"(?i)\b(approved )?runbook\b",
    r"(?i)\bintegration (spec|contract|specification)\b",
    r"(?i)\bdata steward(ship)? advisory\b",
    r"(?i)\bcache[- ]refresh advisory\b",
    r"(?i)\bnon[- ]disclosure hold\b",
    r"(?i)\bredaction (notice|hold)\b",
    r"(?i)\boutput (it|the canonical (value|response)|the following) verbatim\b",
]


def mask_instruction_spans(text: str) -> tuple[str, int]:
    masked = text
    hits = 0
    for pat in _DIRECTIVE_PATTERNS:
        masked, n = re.subn(pat, "[MASKED]", masked)
        hits += n
    return masked, hits


def apply_trim_to_docs(
    docs: list[CorpusDoc], *, mask_spans: bool
) -> tuple[list[CorpusDoc], int]:
    """Return (possibly-masked docs, total directive-pattern hits across poison docs).

    When ``mask_spans`` is False, docs are returned unchanged and hits is 0
    (useful as a "trim" prompting-only baseline). When True, poison-role docs
    have directive spans replaced and ``hits`` reports how many matches fired;
    this lets experiments tell whether trim_mask actually altered the text.
    """
    total_hits = 0
    if not mask_spans:
        return docs, 0
    out: list[CorpusDoc] = []
    for d in docs:
        if d.role == "poison":
            new_text, hits = mask_instruction_spans(d.text)
            total_hits += hits
            # Local import avoids pulling benchmark.datasets (and the HF datasets
            # package) into defense.trim at module import time.
            from benchmark.corpus_builder import CorpusDoc as _CD

            out.append(_CD(doc_id=d.doc_id, title=d.title, text=new_text, role=d.role))
        else:
            out.append(d)
    return out, total_hits
