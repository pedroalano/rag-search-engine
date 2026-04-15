import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from lib.hybrid_search import HybridSearch


def spell_correct(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
    return response.text.strip()


def rewrite_query(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2014-2021"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
    return response.text.strip()


def expand_query(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
    additional = response.text.strip()
    return f"{query} {additional}"


def rerank_individual(results: list, query: str, limit: int) -> list:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]
    )
    print(f"Re-ranking top {len(results)} results using individual method...")
    for i, doc in enumerate(results):
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
        response = client.models.generate_content(
            model="gemma-3-27b-it", contents=prompt, config=config
        )
        try:
            doc["llm_score"] = float(response.text.strip())
        except ValueError, AttributeError:
            doc["llm_score"] = 0.0
        if i < len(results) - 1:
            time.sleep(3)
    return sorted(results, key=lambda r: r["llm_score"], reverse=True)[:limit]


def load_movies():
    data_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ws_parser = subparsers.add_parser("weighted-search", help="Hybrid weighted search")
    ws_parser.add_argument("query", type=str, help="Search query")
    ws_parser.add_argument(
        "--alpha", type=float, default=0.5, help="BM25 weight 0-1 (default 0.5)"
    )
    ws_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results (default 5)"
    )

    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid RRF search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k", type=int, default=60, help="RRF constant (default 60)"
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results (default 5)"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual"],
        help="Re-ranking method to apply after RRF search",
    )

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
            for i, r in enumerate(results[: args.limit], 1):
                print(f"{i}. {r['title']}")
                print(f"  Hybrid Score: {r['hybrid_score']:.3f}")
                print(
                    f"  BM25: {r['bm25_score']:.3f}, Semantic: {r['semantic_score']:.3f}"
                )
                print(f"  {r['document']}...")
        case "rrf-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            query = args.query
            if args.enhance == "spell":
                enhanced = spell_correct(query)
            elif args.enhance == "rewrite":
                enhanced = rewrite_query(query)
            elif args.enhance == "expand":
                enhanced = expand_query(query)
            else:
                enhanced = query
            if enhanced != query:
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
            query = enhanced
            fetch = args.limit * 5 if args.rerank_method else args.limit
            results = hs.rrf_search(query, args.k, fetch)
            if args.rerank_method == "individual":
                results = rerank_individual(results, query, args.limit)
                print(f"\nReciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. {r['title']}")
                    print(f"   Re-rank Score: {r['llm_score']:.3f}/10")
                    print(f"   RRF Score: {r['rrf_score']:.3f}")
                    print(
                        f"   BM25 Rank: {r['bm25_rank']}, Semantic Rank: {r['semantic_rank']}"
                    )
                    print(f"   {r['document']}...")
            else:
                for i, r in enumerate(results[: args.limit], 1):
                    print(f"{i}. {r['title']}")
                    print(f"  RRF Score: {r['rrf_score']:.3f}")
                    print(
                        f"  BM25 Rank: {r['bm25_rank']}, Semantic Rank: {r['semantic_rank']}"
                    )
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
