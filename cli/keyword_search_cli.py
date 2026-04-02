import argparse
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open("data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    match args.command:
        case "search":
            query = args.query.lower()
            results = []

            for movie in data["movies"]:
                if query in movie["title"].lower():
                    results.append(movie)

            results = results[:5]

            print("Searching for: " + args.query)
            for i,movie in enumerate(results,start=1):
                print(f"{i}. {movie['title']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
