"""
labellers assign a list of topic / outlier labels
to the text based on embeddings
"""

from copy import deepcopy
from typing import List

import numpy as np
import hdbscan

from data_utils import TextExample, Label


class Labeller():
    """
    abstract base class
    """
    pass


class HDBScanLabeller():
    """
    assign topic / outlier labels to text examples
    based on the embedding vectors using HDBScan
    """

    def __init__(self, min_cluster_size: int = 2):
        """
        cluster embeddings into topics and outliers using HDBScan
        """
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    def __call__(self, examples: List[TextExample]) -> List[TextExample]:
        """
        Run clustering on a list of examples
        Return and copy of the examples with labels populated
        """
        # perform clustering on embeddings
        embeddings = np.array([example.embedding for example in examples])
        cluster_labels = self.clusterer.fit_predict(embeddings)
        scores = self.clusterer.probabilities_

        # populate labels in examples
        out_examples = deepcopy(examples)
        for example, cluster_label, score in zip(out_examples, cluster_labels, scores):
            example.labels = [Label(label=cluster_label, score=score)]
        return out_examples


# TODO: look into hdbscan RobustSingleLinkage implementation
