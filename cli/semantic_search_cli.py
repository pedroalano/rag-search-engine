import argparse
import json
import os
from lib.semantic_search import (
    SCORE_PRECISION,
    SemanticSearch,
    ChunkedSemanticSearch,
    semantic_chunk,
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
)


def load_movies():
    data_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["movies"]


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "verify", help="Verify the embedding model is loaded correctly"
    )

    subparsers.add_parser(
        "verify_embeddings", help="Build or verify movie embeddings cache"
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for a text input"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for a query string"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search movies by semantic similarity"
    )
    search_parser.add_argument("query", type=str, help="Query string")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results (default 5)"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size word chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Words per chunk (default 200)"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Words shared between consecutive chunks (default 0)",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into sentence-based chunks"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Max sentences per chunk (default 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Sentences shared between consecutive chunks (default 0)",
    )

    subparsers.add_parser(
        "embed_chunks", help="Build or load chunk embeddings for all movies"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunk embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Query string")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results (default 5)"
    )

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
        case "chunk":
            words = args.text.split()
            chunks = []
            i = 0
            while i < len(words):
                chunks.append(" ".join(words[i : i + args.chunk_size]))
                i += args.chunk_size - args.overlap
            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, 1):
                print(f"{idx}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, 1):
                print(f"{idx}. {chunk}")
        case "embed_chunks":
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(load_movies())
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            movies = load_movies()
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(movies)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
