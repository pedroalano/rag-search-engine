import argparse
import json
from nltk.stem import PorterStemmer
from helpers import is_match, load_stopwords
from inverted_index import InvertedIndex
import math

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)
    idf_parser = subparsers.add_parser("idf", help="Calculate IDF")
    idf_parser.add_argument("term", type=str)

    args = parser.parse_args()

    stopwords = load_stopwords()

    # ✅ load JSON file correctly
    with open("./data/movies.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    match args.command:
        case "search":
            stopwords = load_stopwords()
            stemmer = PorterStemmer()

            index = InvertedIndex(stopwords, stemmer)

            try:
                index.load()
            except FileNotFoundError as e:
                print(str(e)) 
                return

            query_tokens = index._tokenize(args.query)

            results = []
            seen = set()

            for token in query_tokens:
                doc_ids = index.get_documents(token)

                for doc_id in doc_ids:
                    if doc_id not in seen:
                        results.append(doc_id)
                        seen.add(doc_id)

                    if len(results) >= 5:
                        break

            print(f"Searching for: {args.query}")
            for i, doc_id in enumerate(results, start=1):
                movie = index.docmap[doc_id]
                print(f"{i}. {movie['title']}")
        case "tf":
            stopwords = load_stopwords()
            stemmer = PorterStemmer()

            index = InvertedIndex(stopwords, stemmer)

            try:
             index.load()
            except FileNotFoundError as e:
             print(str(e))
             return

            tf = index.get_tf(args.doc_id, args.term)
            print(tf)
        case "build":
            stopwords = load_stopwords()
            stemmer = PorterStemmer()

            index = InvertedIndex(stopwords, stemmer)
            index.build(data["movies"])
            index.save()

            print("Index build and saved successfully.")
        case "idf":
            stopwords = load_stopwords()
            stemmer = PorterStemmer()

            index = InvertedIndex(stopwords, stemmer)

            try:
             index.load()
            except FileNotFoundError as e:
             print(str(e))
             return

            total_doc_count = len(index.docmap)

            tokens = index._tokenize(args.term)

            if len(tokens) != 1:
                raise ValueError("Term must be a single token")

            token = tokens[0]

            term_match_doc_count = len(index.index.get(token, set()))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
