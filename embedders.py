"""
embedders populate the embedding attribute of
each TextExample with transformer-based embeddings
"""

from copy import deepcopy
from typing import List

from sentence_transformers import SentenceTransformer

from data_utils import TextExample


class Embedder:
    """
    abstract base class
    """
    pass


class SBERTEmbedder:
    """
    compute embeddings using an SBERT model
    """
    def __init__(self, model_name: str):
        """
        Load the SentenceTransformer model
        A list of pre-trained models is available here:
           https://www.sbert.net/docs/pretrained_models.html
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, examples: List[TextExample]) -> List[TextExample]:
        """
        Compute embeddings for each example text
        Returns a new copy of examples with the embeddings populated
        """
        output_examples = deepcopy(examples)
        for example in output_examples:
            example.embedding = self.model.encode(
                example.text, convert_to_numpy=True)
        return output_examples
