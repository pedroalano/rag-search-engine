import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib.multimodal_search import image_search_command, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding generation"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movies using an image"
    )
    image_search_parser.add_argument(
        "image_path", type=str, help="Path to image file"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for i, r in enumerate(results, 1):
                print(
                    f"{i}. {r['title']} (similarity: {r['similarity']:.3f})\n"
                    f"   {r['description'][:100]}...\n"
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
