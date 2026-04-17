import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib.hybrid_search import HybridSearch


def load_movies():
    data_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["movies"]


def load_golden_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "../data/golden_dataset.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    movies = load_movies()
    hs = HybridSearch(movies)
    test_cases = load_golden_dataset()

    print(f"k={limit}\n")
    for tc in test_cases:
        query = tc["query"]
        relevant = set(tc["relevant_docs"])
        results = hs.rrf_search(query, 60, limit)[:limit]
        retrieved_titles = [r["title"] for r in results]
        hits = sum(1 for t in retrieved_titles if t in relevant)
        precision = hits / limit if limit > 0 else 0.0
        recall = hits / len(relevant) if len(relevant) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(tc['relevant_docs'])}")
        print()


if __name__ == "__main__":
    main()
