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
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
