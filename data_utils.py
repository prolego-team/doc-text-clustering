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
    embedding: Optional[np.ndarray] = None
    labels: Optional[List[Label]] = None


def read_test_labelled_topics() -> List[TextExample]:
    """
    convert test_data/labelled_topics.csv to a list of text examples
    """
    with open("test_data/labelled_topics.csv", "r") as f:
        data = f.readlines()
    labelled_examples = []
    for i, row in enumerate(data[1:]):
        label, text = row.split(",")
        text = text.strip()
        labelled_examples.append(TextExample(
            text,
            str(i),
            labels=[Label(label, 1.0)]
        ))
    return labelled_examples


def sorted_class_labels(examples: List[TextExample]) -> List[str]:
    """
    returns a sorted list of all unique labels in examples
    """
    label_list = [example.labels for example in examples if example]
    if len(label_list) == 0:
        return []
    return sorted(set([label.label for labels in label_list for label in labels]))
