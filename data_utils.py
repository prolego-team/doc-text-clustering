"""
classes and utility functions related to
TextExample data
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Label:
    label: str
    score: float

@dataclass
class TextExample:
    text: str
    id: str
    embedding: List[int]
    labels: List[Label]
