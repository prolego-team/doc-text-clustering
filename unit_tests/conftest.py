"""
test fixtures shared among unit tests
"""

from copy import deepcopy
from typing import List

import pytest
import numpy as np

from data_utils import TextExample
from io_utils import read_txt
from splitters import NewLineSplitter


@pytest.fixture
def text() -> str:
    """
    sk8er boi lyrics
    """
    return read_txt("test_data/sk8er_boi.txt")


@pytest.fixture
def sbert_model_name() -> str:
    """
    name of a pre-trained SBERT model
    """
    return "all-MiniLM-L6-v2"


@pytest.fixture
def splitted_examples(text: str) -> List[TextExample]:
    """
    text split by new line
    """
    splitter = NewLineSplitter(id_prefix="sampledata-")
    return splitter(text)


@pytest.fixture
def examples_with_embeddings(
        splitted_examples: List[TextExample]) -> List[TextExample]:
    """
    text with (dummy, randomly generated) embeddings
    """
    out_examples = deepcopy(splitted_examples)
    for example in out_examples:
        example.embedding = np.random.random(384)
    return out_examples
