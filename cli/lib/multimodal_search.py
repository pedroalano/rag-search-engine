import json
import os

from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        if documents:
            self.texts = [
                f"{doc['title']}: {doc['description']}" for doc in documents
            ]
            self.text_embeddings = self.model.encode(
                self.texts, show_progress_bar=True
            )

    def embed_image(self, image_path):
        img = Image.open(image_path)
        embedding = self.model.encode([img])[0]
        return embedding

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(image_embedding, text_embedding)
            results.append(
                {
                    "id": self.documents[i]["id"],
                    "title": self.documents[i]["title"],
                    "description": self.documents[i]["description"],
                    "similarity": score,
                }
            )
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path):
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    data_path = os.path.join(os.path.dirname(__file__), "../../data/movies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        movies = json.load(f)["movies"]
    ms = MultimodalSearch(documents=movies)
    return ms.search_with_image(image_path)
