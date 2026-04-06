import pickle
from pathlib import Path
from collections import Counter
import math
import string
import os


class InvertedIndex:
    BM25_K1 = 1.5

    BM25_B = 0.75

    def __init__(self, stopwords: set[str], stemmer):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")

    def _tokenize(self, text: str) -> list[str]:
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator).lower()

        tokens = text.split()

        tokens = [t for t in tokens if t and t not in self.stopwords]

        tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

    def _add_document(self, doc_id: int, text: str):
        tokens = self._tokenize(text)

        self.doc_lengths[doc_id] = len(tokens)

        self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        total = sum(self.doc_lengths.values())
        return total / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        term = self.stemmer.stem(term.lower())

        docs = self.index.get(term, set())
        return sorted(docs)

    def build(self, movies: list[dict]):
        for movie in movies:
            doc_id = movie["id"]

            text = f"{movie['title']} {movie['description']}"

            self.docmap[doc_id] = movie
            self._add_document(doc_id, text)

    def save(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        with open(cache_dir / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(cache_dir / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(cache_dir / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        cache_dir = Path("cache")

        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"
        tf_path = cache_dir / "term_frequencies.pkl"
        doc_lengths_path = cache_dir / "doc_lengths.pkl"

        if (
            not index_path.exists()
            or not docmap_path.exists()
            or not doc_lengths_path.exists()
        ):
            raise FileNotFoundError("Index files not found. Please run build first.")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = self._tokenize(term)

        if len(tokens) != 1:
            raise ValueError("Term must be a single token")

        token = tokens[0]

        return self.term_frequencies.get(doc_id, {}).get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        tokens = self._tokenize(term)

        if len(tokens) != 1:
            raise ValueError("Term must be a single token")

        token = tokens[0]

        N = len(self.docmap)  # total docs
        df = len(self.index.get(token, set()))  # docs com o termo

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)

        if tf == 0:
            return 0.0

        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_doc_len = self.__get_avg_doc_length()

        if avg_doc_len == 0:
            return 0.0

        norm = 1 - b + b * (doc_len / avg_doc_len)

        return (tf * (k1 + 1)) / (tf + k1 * norm)
