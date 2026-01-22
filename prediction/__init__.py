"""
prediction package

Entry points:
- python -m prediction.train_and_prediction ...
"""

from .config import parse_args

__all__ = [
    "parse_args",
]
