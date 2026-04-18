import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding generation"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
