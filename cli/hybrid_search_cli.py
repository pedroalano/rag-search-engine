import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="Scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
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
