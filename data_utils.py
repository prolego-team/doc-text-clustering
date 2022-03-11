"""
classes and utility functions related to
TextExample data
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Label:
    label: int
    score: float

@dataclass
class TextExample:
    text: str
    id: str
    embedding: Optional[np.ndarray]
    labels: Optional[List[Label]]
