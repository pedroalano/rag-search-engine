import os
from nltk.stem import PorterStemmer

from inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch

_STOPWORDS = {"a", "an", "the", "of", "on", "in", "to", "and"}


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        stemmer = PorterStemmer()
        self.idx = InvertedIndex(_STOPWORDS, stemmer)
        if not os.path.exists(os.path.join("cache", "index.pkl")):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        fetch = limit * 500

        bm25_raw = self._bm25_search(query, fetch)
        semantic_raw = self.semantic_search.search_chunks(query, fetch)

        bm25_scores = [s for _, s in bm25_raw]
        bm25_min, bm25_max = (min(bm25_scores), max(bm25_scores)) if bm25_scores else (0, 0)

        def norm_bm25(s):
            if bm25_max == bm25_min:
                return 1.0
            return (s - bm25_min) / (bm25_max - bm25_min)

        sem_scores = [r["score"] for r in semantic_raw]
        sem_min, sem_max = (min(sem_scores), max(sem_scores)) if sem_scores else (0, 0)

        def norm_sem(s):
            if sem_max == sem_min:
                return 1.0
            return (s - sem_min) / (sem_max - sem_min)

        combined = {}

        for doc_id, score in bm25_raw:
            doc = self.idx.docmap[doc_id]
            combined[doc_id] = {
                "id": doc_id,
                "title": doc["title"],
                "document": doc["description"][:100],
                "bm25_score": norm_bm25(score),
                "semantic_score": 0.0,
            }

        for result in semantic_raw:
            doc_id = result["id"]
            sem_norm = norm_sem(result["score"])
            if doc_id in combined:
                combined[doc_id]["semantic_score"] = sem_norm
            else:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": sem_norm,
                }

        for entry in combined.values():
            entry["hybrid_score"] = alpha * entry["bm25_score"] + (1 - alpha) * entry["semantic_score"]

        return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
