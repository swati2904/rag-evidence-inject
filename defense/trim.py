"""TRIM: mask directive-like spans in retrieved passages before prompt assembly.

The masker is **role-agnostic** at inference time: it scans every retrieved
passage with the same fixed pattern set, regardless of whether the corpus
builder labelled the passage as ``gold``, ``distractor`` or ``poison``. That
label is used only for evaluation logging (how many hits fired on poison vs.
benign passages), not to decide which text to rewrite. This is what makes
``trim_mask`` a deployable defense rather than an oracle defense.
"""
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
) -> tuple[list[CorpusDoc], dict[str, int]]:
    """Apply role-agnostic directive masking to retrieved passages.

    Args:
        docs: Ordered list of retrieved passages from the per-query pool.
        mask_spans: If False, docs are returned unchanged (the ``trim``
            prompting-only baseline). If True, every passage -- regardless of
            its corpus-builder role -- is scanned with the fixed pattern set
            and matching spans are replaced with ``[MASKED]``.

    Returns:
        ``(possibly-masked docs, hits)`` where ``hits`` is a dictionary with
        ``mask_hits_total`` (all passages), ``mask_hits_poison`` (passages
        labelled poison by the builder) and ``mask_hits_benign`` (gold and
        distractor passages combined). The split is for evaluation logging
        only; the masker does not look at the role to decide what to rewrite.
    """
    hits = {"mask_hits_total": 0, "mask_hits_poison": 0, "mask_hits_benign": 0}
    if not mask_spans:
        return docs, hits
    out: list[CorpusDoc] = []
    # Local import avoids pulling benchmark.datasets (and the HF datasets
    # package) into defense.trim at module import time.
    from benchmark.corpus_builder import CorpusDoc as _CD

    for d in docs:
        new_text, n = mask_instruction_spans(d.text)
        hits["mask_hits_total"] += n
        if d.role == "poison":
            hits["mask_hits_poison"] += n
        else:
            hits["mask_hits_benign"] += n
        out.append(_CD(doc_id=d.doc_id, title=d.title, text=new_text, role=d.role))
    return out, hits
