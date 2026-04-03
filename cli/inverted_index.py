import pickle
from pathlib import Path

class InvertedIndex:
    def __init__(self, stopwords: set[str], stemmer):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.stopwords = stopwords
        self.stemmer = stemmer


    def _tokenize(self, text:str) -> list[str]:
        import string

        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator).lower()

        tokens = text.split()

        tokens = [
            t for t in tokens
            if t and t not in self.stopwords
        ]

        tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

    def _add_document(self, doc_id: int, text: str):
        tokens = self._tokenize(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

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
