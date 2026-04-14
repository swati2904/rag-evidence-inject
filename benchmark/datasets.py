"""Load KILT NQ and HotpotQA (distractor) for controlled RAG pools."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import load_dataset

_logger = logging.getLogger(__name__)


@dataclass
class QAPoolExample:
    example_id: str
    source: str  # "kilt_nq" | "hotpot"
    question: str
    gold_answers: list[str]
    gold_docs: list[dict[str, str]]  # id, title, text
    distractor_docs: list[dict[str, str]]


def _parse_kilt_output(output: Any) -> tuple[list[str], list[dict[str, str]]]:
    answers: list[str] = []
    gold_docs: list[dict[str, str]] = []
    if not output:
        return answers, gold_docs
    blocks = output if isinstance(output, list) else [output]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        raw_ans = block.get("answer")
        if isinstance(raw_ans, str) and raw_ans.strip():
            answers.append(raw_ans.strip())
        elif isinstance(raw_ans, list):
            answers.extend(str(x).strip() for x in raw_ans if str(x).strip())
        prov = block.get("provenance") or []
        for i, p in enumerate(prov):
            if not isinstance(p, dict):
                continue
            text = (p.get("text") or p.get("paragraph") or "").strip()
            title = p.get("title") or "wikipedia"
            section = p.get("section") or ""
            wid = str(p.get("wikipedia_id") or p.get("id") or f"gold_{i}")
            if not text:
                # facebook/kilt_tasks often omits raw passage text; surrogate for pipeline dev.
                # For publication runs, join facebook/kilt_wikipedia or another gold corpus.
                a = ""
                if isinstance(raw_ans, str):
                    a = raw_ans.strip()
                elif isinstance(raw_ans, list) and raw_ans:
                    a = str(raw_ans[0]).strip()
                text = (
                    f"Wikipedia article: {title}. {section}\n"
                    f"Gold answer associated with this evidence: {a}".strip()
                )
            gold_docs.append({"id": f"gold:{wid}", "title": str(title), "text": str(text)})
    return answers, gold_docs


def iter_kilt_nq(*, max_examples: int, seed: int) -> Iterator[QAPoolExample]:
    rng = random.Random(seed)
    try:
        ds = load_dataset("facebook/kilt_tasks", "nq", split="train", streaming=False)
    except Exception as e:
        _logger.warning("KILT NQ (facebook/kilt_tasks) unavailable (%s); skip kilt_nq stream.", e)
        return
    n = min(max_examples, len(ds))
    order = list(range(n))
    rng.shuffle(order)
    for idx in order:
        row = ds[idx]
        q = row.get("input") or row.get("question")
        output = row.get("output")
        answers, gold_docs = _parse_kilt_output(output)
        if not q or not gold_docs:
            continue
        if not answers:
            answers = [gold_docs[0]["text"][:32].strip() + " …"]
        eid = f"kilt_nq_{idx}"
        yield QAPoolExample(
            example_id=eid,
            source="kilt_nq",
            question=str(q),
            gold_answers=answers,
            gold_docs=gold_docs,
            distractor_docs=[],
        )


def iter_hotpot_distractor(*, max_examples: int, seed: int) -> Iterator[QAPoolExample]:
    ds = load_dataset("hotpot_qa", "distractor", split="train")
    n = min(max_examples, len(ds))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    for idx in order:
        row = ds[idx]
        question = row["question"]
        answer = row["answer"]
        titles, sents = row["context"]["title"], row["context"]["sentences"]
        supporting = row["supporting_facts"]
        sup_titles = {supporting["title"][i] for i in range(len(supporting["title"]))}
        gold_docs: list[dict[str, str]] = []
        distractors: list[dict[str, str]] = []
        for t, s_list in zip(titles, sents):
            text = " ".join(s_list)
            doc_id = f"hotpot:{t}"
            doc = {"id": doc_id, "title": t, "text": text}
            if t in sup_titles:
                gold_docs.append(doc)
            else:
                distractors.append(doc)
        if not gold_docs:
            continue
        eid = f"hotpot_{idx}"
        yield QAPoolExample(
            example_id=eid,
            source="hotpot",
            question=question,
            gold_answers=[answer],
            gold_docs=gold_docs,
            distractor_docs=distractors,
        )


def load_pilot_examples(
    *,
    n_nq: int,
    n_hotpot: int,
    seed: int,
    use_kilt_nq: bool,
) -> list[QAPoolExample]:
    """Return up to n_nq KILT NQ + n_hotpot Hotpot examples (Hotpot fills if KILT missing)."""
    out: list[QAPoolExample] = []
    if use_kilt_nq:
        for ex in iter_kilt_nq(max_examples=max(2000, n_nq * 20), seed=seed):
            out.append(ex)
            if sum(1 for x in out if x.source == "kilt_nq") >= n_nq:
                break
    nq_have = sum(1 for x in out if x.source == "kilt_nq")
    if nq_have < n_nq:
        _logger.warning(
            "KILT NQ yielded %s/%s examples; remaining NQ slots filled with Hotpot.",
            nq_have,
            n_nq,
        )
    hp: list[QAPoolExample] = []
    for ex in iter_hotpot_distractor(max_examples=max(2000, (n_hotpot + n_nq) * 5), seed=seed + 7):
        hp.append(ex)
        if len(hp) >= n_hotpot + max(0, n_nq - nq_have):
            break
    if nq_have < n_nq:
        need = n_nq - nq_have
        out.extend(hp[:need])
        hp = hp[need:]
    out.extend(hp[:n_hotpot])
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[: n_nq + n_hotpot]
