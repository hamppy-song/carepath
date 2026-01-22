# scripts/extract_embeddings.py
from carepath.config import parse_args
from carepath.extract import run_embedding_extraction


def main():
    args = parse_args()
    run_embedding_extraction(args)


if __name__ == "__main__":
    main()
