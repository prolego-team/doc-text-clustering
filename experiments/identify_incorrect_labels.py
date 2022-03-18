"""
use embeddings and clustering to identify incorrectly labelled data
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Dict

import data_utils
from embedders import Embedder, SBERTEmbedder
from labellers import Labeller, HDBScanLabeller


SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


class LabelledTextClustering:
    """
    assign cluster labels to text by clustering embeddings
    """

    def __init__(self, embedder: Embedder, labeller: Labeller) -> None:
        self.embedder = embedder
        self.labeller = labeller

    def __call__(self, examples: List[data_utils.LabelledTextExample]) -> List[data_utils.LabelledTextExample]:
        """
        run embedding and labelling on a list of examples
        construct output labelled text examples
        """
        # embedding and clustering
        examples_with_embeddings = self.embedder(examples)
        examples_with_cluster_labels = self.labeller(examples_with_embeddings)

        # construct labelled text examples containing both the assigned
        # and cluster labels
        labelled_examples = [data_utils.LabelledTextExample(
            text=example.text,
            id=example.id,
            embedding=None,
            cluster_labels=labelled_example.cluster_labels,
            assigned_labels=example.assigned_labels
        ) for example, labelled_example in zip(examples, examples_with_cluster_labels)]

        return labelled_examples


def score_labels_by_frequency(label_counts: Dict[str, int]) -> Dict[str, int]:
    """
    assign a score indicating the likelihood that an instance of a topic
    is correctly labelled (high) or mislabelled (low), based on the
    relative counts of occurences of each topic
    """

    # count total number of instances of each label, excluding labels
    # that occur only once since they are assumed mislabelled by default
    total_count = sum([count for count in label_counts.values() if count > 1])

    # assign a score based on relative frequency of each label
    label_scores = {}
    for label, count in label_counts.items():
        if count == 1:
            # anything that is counted only once is likely mislabelled
            label_scores[label] = 0.0
        else:
            label_scores[label] = count / total_count

    return label_scores


def evaluate_assigned_labels(
        labelled_examples: List[data_utils.LabelledTextExample]) -> List[data_utils.LabelledTextExample]:
    """
    score the likelihood that each example should have each assigned label
    based on the prevalence of the assigned label in each cluster the example
    belongs to
    """

    # count frequency of each assigned label within each cluster label
    cluster_labels = data_utils.sorted_cluster_labels(labelled_examples)
    assigned_labels = data_utils.sorted_assigned_labels(labelled_examples)
    cluster_to_assigned_label_count = {cluster_label: {assigned_label: 0 for assigned_label in assigned_labels}
                                       for cluster_label in cluster_labels}
    for example in labelled_examples:
        for cluster_label in example.cluster_labels:
            for assigned_label in example.assigned_labels:
                cluster_to_assigned_label_count[cluster_label.label][assigned_label.label] += 1

    # TODO: handle label = -1 --> Likely out-of-set?

    # score likelihood that each assigned label is incorrect and generate output
    out_examples = deepcopy(labelled_examples)
    for example in out_examples:
        candidate_labels = []
        for cluster_label in example.cluster_labels:
            assigned_label_to_score = score_labels_by_frequency(
                cluster_to_assigned_label_count[cluster_label.label]
            )
            candidate_labels += [data_utils.Label(topic, score) for topic, score in assigned_label_to_score.items()]
        example.candidate_labels = candidate_labels
    return out_examples


def main():

    ground_truth_labelled_examples = data_utils.read_test_labelled_topics()

    # change some labels
    # ground_truth_labelled_examples[0].assigned_labels[0].label = "clothes"

    embedder = SBERTEmbedder(SBERT_MODEL_NAME)
    labeller = HDBScanLabeller()
    clusterer = LabelledTextClustering(embedder=embedder, labeller=labeller)
    labelled_examples = clusterer(ground_truth_labelled_examples)

    out_labelled_examples = evaluate_assigned_labels(labelled_examples)

    for example in out_labelled_examples:
        print("-------")
        print(example.text)
        print("Assigned Label:", example.assigned_labels[0].label)
        print("Proposed Labels:", [(label.label, label.score) for label in example.candidate_labels if label.score > 0])

if __name__ == "__main__":
    main()