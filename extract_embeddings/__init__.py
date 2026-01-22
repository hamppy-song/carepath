"""
extract_embeddings package

Entry points:
- python -m extract_embeddings.main ...
"""

from .config import parse_args, parse_run_id
from .extract import save_embedding_files

__all__ = [
    "parse_args",
    "parse_run_id",
    "save_embedding_files",
]
