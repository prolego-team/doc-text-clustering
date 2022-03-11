"""
test fixtures shared among unit tests
"""

from typing import List

import pytest

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
