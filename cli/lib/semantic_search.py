import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

SCORE_PRECISION = 4


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunk(text, max_chunk_size=4, overlap=1):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
        sentences = [text]
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    i = 0
    while i < len(sentences):
        chunks.append(" ".join(sentences[i : i + max_chunk_size]))
        i += max_chunk_size - overlap
    return chunks


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        cache_path = "cache/movie_embeddings.npy"
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Input text must not be empty or whitespace.")
        result = self.model.encode([text])
        return result[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        scores = [
            (cosine_similarity(query_embedding, self.embeddings[i]), self.documents[i])
            for i in range(len(self.documents))
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scores[:limit]
        ]


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def verify_embeddings():
    ss = SemanticSearch()
    data_path = os.path.join(os.path.dirname(__file__), "../../data/movies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = data["movies"]
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks = []
        chunk_metadata = []
        for movie_idx, doc in enumerate(documents):
            if not doc.get("description", "").strip():
                continue
            chunks = semantic_chunk(doc["description"], max_chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        emb_path = "cache/chunk_embeddings.npy"
        meta_path = "cache/chunk_metadata.json"
        if os.path.exists(emb_path) and os.path.exists(meta_path):
            self.chunk_embeddings = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, (chunk_emb, meta) in enumerate(
            zip(self.chunk_embeddings, self.chunk_metadata)
        ):
            score = cosine_similarity(query_embedding, chunk_emb)
            chunk_scores.append(
                {
                    "chunk_idx": meta["chunk_idx"],
                    "movie_idx": meta["movie_idx"],
                    "score": score,
                    "global_idx": i,
                }
            )
        movie_scores = {}
        for cs in chunk_scores:
            mid = cs["movie_idx"]
            if mid not in movie_scores or cs["score"] > movie_scores[mid]["score"]:
                movie_scores[mid] = cs
        sorted_scores = sorted(
            movie_scores.values(), key=lambda x: x["score"], reverse=True
        )[:limit]
        results = []
        for cs in sorted_scores:
            doc = self.documents[cs["movie_idx"]]
            meta = self.chunk_metadata[cs["global_idx"]]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(cs["score"], SCORE_PRECISION),
                    "metadata": meta or {},
                }
            )
        return results
