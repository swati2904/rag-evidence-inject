"""Hybrid BM25 + dense by min-max normalizing scores then linear blend."""
from __future__ import annotations

from dataclasses import dataclass

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever


@dataclass
class HybridResult:
    doc_ids: list[str]
    scores: list[float]


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [1.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        *,
        alpha: float = 0.5,
        candidate_k: int = 50,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.candidate_k = candidate_k
        self._all_ids = bm25.doc_ids

    def retrieve(self, query: str, top_k: int) -> HybridResult:
        bm = self.bm25.retrieve(query, min(self.candidate_k, len(self._all_ids)))
        dn = self.dense.retrieve(query, min(self.candidate_k, len(self._all_ids)))
        id2bm = dict(zip(bm.doc_ids, bm.scores, strict=False))
        id2dn = dict(zip(dn.doc_ids, dn.scores, strict=False))
        ids = list(dict.fromkeys(list(id2bm.keys()) + list(id2dn.keys())))
        bmv = [id2bm.get(i, 0.0) for i in ids]
        dnv = [id2dn.get(i, 0.0) for i in ids]
        bmn = _minmax(bmv)
        dnn = _minmax(dnv)
        fused = [self.alpha * dnn[i] + (1 - self.alpha) * bmn[i] for i in range(len(ids))]
        ranked = sorted(range(len(ids)), key=lambda i: fused[i], reverse=True)[:top_k]
        return HybridResult(
            doc_ids=[ids[i] for i in ranked],
            scores=[float(fused[i]) for i in ranked],
        )
