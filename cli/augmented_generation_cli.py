import argparse
import json
import os
import sys

from dotenv import load_dotenv
from google import genai

sys.path.insert(0, os.path.dirname(__file__))
from lib.hybrid_search import HybridSearch


def load_movies():
    data_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["movies"]


def rag_search(query):
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, k=60, limit=5)[:5]

    print("Search Results:")
    for r in results:
        print(f"- {r['title']}")

    docs = "\n".join(
        f"- Title: {r['title']}\n  Description: {r['document']}" for r in results
    )

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

    print(f"\nRAG Response:\n{response.text}")


def summarize_search(query, limit):
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, k=60, limit=limit)[:limit]

    results_str = "\n".join(
        f"- Title: {r['title']}\n  Description: {r['document']}" for r in results
    )

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results_str}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

    print("Search Results:")
    for r in results:
        print(f"  - {r['title']}")

    print(f"\nLLM Summary:\n{response.text}")


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results using LLM"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to summarize"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_search(args.query)
        case "summarize":
            summarize_search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
