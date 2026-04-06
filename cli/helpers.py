import string
from pathlib import Path
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

STOPWORDS = {"a", "an", "the", "of", "on", "in", "to", "and"}

BM25_K1 = 1.5

BM25_B = 0.75

stemmer = PorterStemmer()


def normalize(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower()


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens = normalize(text).split()

    filtered = [t for t in tokens if t and len(t) >= 3 and t not in stopwords]

    stemmed = [stemmer.stem(t) for t in filtered]

    return stemmed


def is_match(query: str, title: str, stopwords: set[str]) -> bool:
    query_tokens = tokenize(query, stopwords)
    title_tokens = tokenize(title, stopwords)

    for q in query_tokens:
        for t in title_tokens:
            if len(q) >= 3 and (q in t or t in q):
                return True
    return False


def load_stopwords() -> set[str]:
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "../data/stopwords.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())


def bm25_idf_command(term: str) -> float:

    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    index = InvertedIndex(stopwords, stemmer)
    index.load()

    return index.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    index = InvertedIndex(stopwords, stemmer)
    index.load()

    return index.get_bm25_tf(doc_id, term, k1, b)
