import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


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
