import argparse
import json
from helpers import is_match, load_stopwords

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    stopwords = load_stopwords()

    # ✅ load JSON file correctly
    with open("./data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    match args.command:
        case "search":
            results = []

            # iterate over movies
            for movie in data["movies"]:
                if is_match(args.query, movie["title"],stopwords):
                    results.append(movie)

            # limit to 5 results (already sorted by ID)
            results = results[:5]

            # print output
            print(f"Searching for: {args.query}")
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
