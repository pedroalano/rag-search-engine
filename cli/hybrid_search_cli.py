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


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ws_parser = subparsers.add_parser("weighted-search", help="Hybrid weighted search")
    ws_parser.add_argument("query", type=str, help="Search query")
    ws_parser.add_argument("--alpha", type=float, default=0.5, help="BM25 weight 0-1 (default 0.5)")
    ws_parser.add_argument("--limit", type=int, default=5, help="Number of results (default 5)")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="Scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results[:args.limit], 1):
                print(f"{i}. {r['title']}")
                print(f"  Hybrid Score: {r['hybrid_score']:.3f}")
                print(f"  BM25: {r['bm25_score']:.3f}, Semantic: {r['semantic_score']:.3f}")
                print(f"  {r['document']}...")
        case "normalize":
            scores = args.scores
            if not scores:
                pass
            elif min(scores) == max(scores):
                for _ in scores:
                    print(f"* {1.0:.4f}")
            else:
                lo, hi = min(scores), max(scores)
                for s in scores:
                    print(f"* {(s - lo) / (hi - lo):.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
