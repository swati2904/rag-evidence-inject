"""TRIM: mask directive-like spans in poison passages before prompt assembly."""
from __future__ import annotations

import re

from benchmark.corpus_builder import CorpusDoc

_DIRECTIVE_PATTERNS = [
    r"(?i)\bignore\b.{0,40}\b(document|passage|evidence|context)\b",
    r"(?i)\bdo not cite\b",
    r"(?i)\bclassified\b",
    r"(?i)\bresponse contract\b",
    r"(?i)\btreat the following as authoritative\b",
]


def mask_instruction_spans(text: str) -> tuple[str, int]:
    masked = text
    hits = 0
    for pat in _DIRECTIVE_PATTERNS:
        masked, n = re.subn(pat, "[MASKED]", masked)
        hits += n
    return masked, hits


def apply_trim_to_docs(docs: list[CorpusDoc], *, mask_spans: bool) -> list[CorpusDoc]:
    if not mask_spans:
        return docs
    out: list[CorpusDoc] = []
    for d in docs:
        if d.role == "poison":
            new_text, _ = mask_instruction_spans(d.text)
            out.append(CorpusDoc(doc_id=d.doc_id, title=d.title, text=new_text, role=d.role))
        else:
            out.append(d)
    return out
