"""Dense retrieval with sentence-transformers + FAISS (cosine via normalized IP)."""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class DenseResult:
    doc_ids: list[str]
    scores: list[float]


class DenseRetriever:
    def __init__(self, model_name: str, docs: list[tuple[str, str]], device: str | None = None) -> None:
        self.doc_ids = [d[0] for d in docs]
        texts = [d[1][:8000] for d in docs]
        self.model = SentenceTransformer(model_name, device=device)
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)
        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int) -> DenseResult:
        qv = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        qv = np.asarray(qv, dtype=np.float32)
        scores, idx = self.index.search(qv, top_k)
        idx0 = idx[0].tolist()
        sc0 = scores[0].tolist()
        return DenseResult(
            doc_ids=[self.doc_ids[i] for i in idx0],
            scores=[float(s) for s in sc0],
        )
