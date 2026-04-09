import argparse
import json
import os
from lib.semantic_search import SemanticSearch, verify_model, verify_embeddings, embed_text, embed_query_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model is loaded correctly")

    subparsers.add_parser("verify_embeddings", help="Build or verify movie embeddings cache")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a text input")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query string")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Query string")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results (default 5)")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            ss = SemanticSearch()
            data_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ss.load_or_create_embeddings(data["movies"])
            results = ss.search(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
