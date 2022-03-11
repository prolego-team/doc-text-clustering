"""
unit tests for embedders.py
"""

from typing import List

import pytest

import embedders
from data_utils import TextExample


@pytest.mark.usefixtures("sbert_model_name")
@pytest.mark.usefixtures("splitted_examples")
def test_SBERTEmbedder(
        sbert_model_name: str,
        splitted_examples: List[TextExample]) -> None:
    """
    test that the SBERTEmbedder populates embedding
    for each TextExample that are the correct type
    and length
    """
    embedder = embedders.SBERTEmbedder(sbert_model_name)
    examples_with_embeddings = embedder(splitted_examples)
    assert type(examples_with_embeddings) == list
    assert len(examples_with_embeddings) == len(splitted_examples)
    for example in examples_with_embeddings:
        assert example.embedding is not None
        assert len(example.embedding) == 384
