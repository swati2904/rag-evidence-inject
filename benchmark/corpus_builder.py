"""Build controlled per-query retrieval pools (gold, distractors, poison)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from benchmark.datasets import QAPoolExample
from benchmark.templates import PoisonTemplate, TEMPLATES, render_poison


DocRole = Literal["gold", "distractor", "poison"]


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    role: DocRole


@dataclass
class RetrievalPool:
    example_id: str
    question: str
    gold_answers: list[str]
    docs: list[CorpusDoc]
    poison_template: str
    attack_family: str
    malicious_target: str


def _pick_malicious_answer(gold_answers: list[str], rng: random.Random) -> str:
    if gold_answers:
        return f"INCORRECT_PLACEHOLDER_{rng.randint(1000,9999)}"
    return "UNKNOWN"


def build_pool_for_example(
    ex: QAPoolExample,
    *,
    attack: PoisonTemplate,
    num_distractors: int,
    rng: random.Random,
) -> RetrievalPool | None:
    gold_docs = ex.gold_docs
    if not gold_docs:
        return None
    distractors = list(ex.distractor_docs)
    rng.shuffle(distractors)
    real_d = distractors[: max(0, num_distractors)]
    chosen_d = list(real_d)
    if len(chosen_d) < num_distractors and ex.source == "kilt_nq":
        # Pad pool size with extra gold snippets when KILT yields too few real distractors.
        # These padded docs are kept out of the poison-prefix source to avoid leaking gold
        # answer text into the poison passage (see paper limitations / Phase A confound fix).
        for i, g in enumerate(gold_docs[1:]):
            if len(chosen_d) >= num_distractors:
                break
            chosen_d.append(
                {
                    "id": f"synth_d_{ex.example_id}_{i}",
                    "title": g["title"] + " (related)",
                    "text": g["text"][:800] + " Additional context may be incomplete.",
                }
            )
    mal = _pick_malicious_answer(ex.gold_answers, rng)
    summary = real_d[0]["text"][:400] if real_d else ""
    poison_text = render_poison(
        attack,
        distractor_summary=summary,
        malicious_answer=mal,
    )
    poison_doc = CorpusDoc(
        doc_id=f"poison:{ex.example_id}",
        title="Related note",
        text=poison_text,
        role="poison",
    )
    docs: list[CorpusDoc] = []
    for g in gold_docs:
        docs.append(CorpusDoc(doc_id=g["id"], title=g["title"], text=g["text"], role="gold"))
    for d in chosen_d:
        docs.append(CorpusDoc(doc_id=d["id"], title=d["title"], text=d["text"], role="distractor"))
    docs.append(poison_doc)
    rng.shuffle(docs)
    return RetrievalPool(
        example_id=ex.example_id,
        question=ex.question,
        gold_answers=ex.gold_answers,
        docs=docs,
        poison_template=attack.name,
        attack_family=attack.family.value,
        malicious_target=mal,
    )


def build_clean_pool(ex: QAPoolExample, *, num_distractors: int, rng: random.Random) -> list[CorpusDoc]:
    """Gold and distractors only (no poison); used by the retrieval sanity script."""
    gold_docs = ex.gold_docs
    distractors = list(ex.distractor_docs)
    rng.shuffle(distractors)
    chosen_d = distractors[: max(0, num_distractors)]
    if len(chosen_d) < num_distractors and ex.source == "kilt_nq":
        for i, g in enumerate(gold_docs[1:]):
            if len(chosen_d) >= num_distractors:
                break
            chosen_d.append(
                {
                    "id": f"synth_d_{ex.example_id}_{i}",
                    "title": g["title"] + " (related)",
                    "text": g["text"][:800] + " Additional context may be incomplete.",
                }
            )
    docs: list[CorpusDoc] = []
    for g in gold_docs:
        docs.append(CorpusDoc(doc_id=g["id"], title=g["title"], text=g["text"], role="gold"))
    for d in chosen_d:
        docs.append(CorpusDoc(doc_id=d["id"], title=d["title"], text=d["text"], role="distractor"))
    rng.shuffle(docs)
    return docs


def insert_poison_at_rank(
    pool: RetrievalPool,
    poison_rank: int,
    rng: random.Random,
) -> list[CorpusDoc]:
    """Return ordered docs with poison placed at 1-based rank (clamped)."""
    poison = next(d for d in pool.docs if d.role == "poison")
    others = [d for d in pool.docs if d.role != "poison"]
    rng.shuffle(others)
    rank = max(1, min(poison_rank, len(others) + 1))
    out = others[: rank - 1] + [poison] + others[rank - 1 :]
    return out
