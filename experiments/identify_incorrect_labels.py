"""
use embeddings and clustering to identify incorrectly labelled data
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import data_utils
from embedders import Embedder, SBERTEmbedder
from labellers import Labeller, HDBScanLabeller


SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class MislabelledLabel(data_utils.Label):
    labelled_score: float

@dataclass
class MislabelledTextExample(data_utils.TextExample):
    labels: Optional[List[MislabelledLabel]] = None
    candidate_labels: Optional[List[data_utils.Label]] = None


class TextLabeller:
    """
    label text by clustering embeddings
    """

    def __init__(self, embedder: Embedder, labeller: Labeller) -> None:
        self.embedder = embedder
        self.labeller = labeller

    def __call__(self, examples: List[data_utils.TextExample]) -> List[data_utils.TextExample]:
        """
        run embedding and labelling on a list of examples
        """
        examples_with_embeddings = self.embedder(examples)
        labelled_examples = self.labeller(examples_with_embeddings)
        return labelled_examples


def assign_labelled_score_to_topics(topic_counts: Dict[str, int]) -> Dict[str, int]:
    """
    assign a score indicating the likelihood that an instance of a topic
    is correctly labelled (high) or mislabelled (low), based on the
    relative counts of occurences of each topic
    """

    # anything that is counted only once is likely mislabelled
    topic_scores = {topic: 1.0 for topic, count in topic_counts.items()
                    if count == 1}

    # count total number of instances of each topic, excluding topics
    # that occur only once since they are assummed mislabelled by default
    total_count = sum([count for count in topic_counts.values() if count > 1])

    # assign a score based on relative frequency of each topic
    topic_scores = {}
    for topic, count in topic_counts.items():
        if count == 1:
            # anything that is counted only once is likely mislabelled
            topic_scores[topic] = 0.0
        else:
            topic_scores[topic] = count / total_count

    return topic_scores


def identify_incorrect_ground_truth(
        pred_examples: List[data_utils.TextExample], true_examples: List[data_utils.TextExample]) -> List[MislabelledTextExample]:
    """
    for each predicted label, score the likelihood that each occurence of a true topic
    is an outlier based on the prevalence of that topic in the labelled class
    """

    # count frequency of each topic within each label
    labels = data_utils.sorted_class_labels(pred_examples)
    topics = data_utils.sorted_class_labels(true_examples)
    topic_count_by_label = {label: {topic: 0 for topic in topics} for label in labels}
    for pred, true in zip(pred_examples, true_examples):
        for label in pred.labels:
            for topic in true.labels:
                topic_count_by_label[label.label][topic.label] += 1

    # TODO: handle label = -1

    # score likelihood that each label is incorrect and generate output
    out_examples = []
    for pred, true in zip(pred_examples, true_examples):
        new_labels = []
        candidate_labels = []
        for pred_label in pred.labels:
            topic_to_labelled_score = assign_labelled_score_to_topics(
                topic_count_by_label[pred_label.label])
            new_labels += [MislabelledLabel(topic.label, topic.score, topic_to_labelled_score[topic.label])
                           for topic in true.labels]
            candidate_labels += [data_utils.Label(topic, score) for topic, score in topic_to_labelled_score.items()]
        out_examples.append(MislabelledTextExample(
            true.text,
            true.id,
            true.embedding,
            new_labels,
            candidate_labels
        ))
    return out_examples


def main():
    ground_truth_labelled_examples = data_utils.read_test_labelled_topics()

    # change some labels
    # labelled_examples[0].labels[0].label = "clothes"

    embedder = SBERTEmbedder(SBERT_MODEL_NAME)
    labeller = HDBScanLabeller()
    text_labeller = TextLabeller(embedder=embedder, labeller=labeller)
    labelled_examples = text_labeller(ground_truth_labelled_examples)

    mislabelled_examples = identify_incorrect_ground_truth(labelled_examples, ground_truth_labelled_examples)

    for example in mislabelled_examples:
        print("-------")
        print(example.text)
        print("Assigned Label:", example.labels[0].label, example.labels[0].labelled_score)
        print("Proposed Labels:", [(label.label, label.score) for label in example.candidate_labels if label.score > 0])

if __name__ == "__main__":
    main()