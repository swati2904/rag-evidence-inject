"""BM25 retrieval over a fixed doc pool."""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().replace("\n", " ").split()


@dataclass
class BM25Result:
    doc_ids: list[str]
    scores: list[float]


class BM25Retriever:
    def __init__(self, docs: list[tuple[str, str]]) -> None:
        """docs: list of (doc_id, full_text_for_indexing)"""
        self.doc_ids = [d[0] for d in docs]
        self._corpus_tokens = [_tokenize(d[1]) for d in docs]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, query: str, top_k: int) -> BM25Result:
        q = _tokenize(query)
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return BM25Result(
            doc_ids=[self.doc_ids[i] for i in ranked],
            scores=[float(scores[i]) for i in ranked],
        )
