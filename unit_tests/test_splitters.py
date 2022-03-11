"""
unit tests for splitters.py
"""

import pytest

from data_utils import TextExample
import splitters


@pytest.mark.use_fixtures("text")
def test_NewLineSplitter(text: str):
    """
    """
    splitter = splitters.NewLineSplitter(id_prefix="sampleid-")
    examples = splitter(text)
    assert type(examples) == list
    for example in examples:
        assert type(example) == TextExample
        assert example.id.startswith("sampleid-")
        # print(repr(example.text))
