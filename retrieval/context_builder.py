"""Assemble model context from retrieval results with optional poison injection."""
from __future__ import annotations

import random

from benchmark.corpus_builder import CorpusDoc, RetrievalPool
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever


def _doc_join_text(d: CorpusDoc) -> str:
    return f"{d.title}\n{d.text}"


def retrieve_ids_bm25(docs: list[CorpusDoc], question: str, top_k: int) -> list[str]:
    tuples = [(d.doc_id, _doc_join_text(d)) for d in docs]
    r = BM25Retriever(tuples).retrieve(question, top_k)
    return r.doc_ids


def retrieve_ids_dense(
    docs: list[CorpusDoc],
    question: str,
    top_k: int,
    model_name: str,
    device: str | None,
) -> list[str]:
    tuples = [(d.doc_id, _doc_join_text(d)) for d in docs]
    r = DenseRetriever(model_name, tuples, device=device).retrieve(question, top_k)
    return r.doc_ids


def retrieve_ids_hybrid(
    docs: list[CorpusDoc],
    question: str,
    top_k: int,
    model_name: str,
    alpha: float,
    device: str | None,
) -> list[str]:
    tuples = [(d.doc_id, _doc_join_text(d)) for d in docs]
    bm25 = BM25Retriever(tuples)
    dense = DenseRetriever(model_name, tuples, device=device)
    return HybridRetriever(bm25, dense, alpha=alpha).retrieve(question, top_k).doc_ids


def ensure_poison_in_ids(
    ranked_ids: list[str],
    poison_id: str,
    *,
    top_k: int,
    rng: random.Random,
) -> list[str]:
    ids = list(ranked_ids)
    if poison_id in ids:
        return ids[:top_k]
    if len(ids) >= top_k:
        drop_idx = rng.randrange(top_k)
        ids = ids[:drop_idx] + ids[drop_idx + 1 : top_k]
    ids = (ids + [poison_id])[:top_k]
    return ids


def reorder_poison_rank(docs: list[CorpusDoc], poison_id: str, poison_rank: int) -> list[CorpusDoc]:
    """1-based poison_rank within len(docs)."""
    poison = next(d for d in docs if d.doc_id == poison_id)
    others = [d for d in docs if d.doc_id != poison_id]
    r = max(1, min(poison_rank, len(docs)))
    return others[: r - 1] + [poison] + others[r - 1 :]


def ids_to_docs(all_docs: list[CorpusDoc], ids: list[str]) -> list[CorpusDoc]:
    m = {d.doc_id: d for d in all_docs}
    return [m[i] for i in ids if i in m]
