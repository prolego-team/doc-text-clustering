"""
unit tests for labellers.py
"""

from typing import List

import pytest

from data_utils import Label, TextExample
import labellers


@pytest.mark.use_fixtures("examples_with_embeddings")
def test_HDBScanLabeller(examples_with_embeddings: List[TextExample]) -> None:
    """
    test that the labeller populates labels for each example
    with the expected length and type of labels
    """
    labeller = labellers.HDBScanLabeller()
    labelled_examples = labeller(examples_with_embeddings)
    assert type(labelled_examples) == list
    assert len(labelled_examples) == len(examples_with_embeddings)
    for example in labelled_examples:
        assert example.labels is not None
        assert len(example.labels) == 1
        assert type(example.labels[0]) == Label
