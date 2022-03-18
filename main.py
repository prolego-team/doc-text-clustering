"""
Run text clustering workflow on a txt file
"""

import os

import click

from io_utils import read_txt
import splitters
import embedders
import labellers


SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


@click.command()
@click.argument("txt_filepath", type=click.Path(exists=True))
def main(txt_filepath: str) -> None:
    """
    """

    # read in txt file
    text = read_txt(txt_filepath)

    # split
    filename = os.path.split(txt_filepath)[1].split(".")[0]
    splitter = splitters.NewLineSplitter(id_prefix=filename + "-")
    examples = splitter(text)

    # embed
    embedder = embedders.SBERTEmbedder(SBERT_MODEL_NAME)
    examples_with_embeddings = embedder(examples)

    # label
    labeller = labellers.HDBScanLabeller()
    labelled_examples = labeller(examples_with_embeddings)

    # group by label
    labels_and_text = {}
    for example in labelled_examples:
        for label in example.cluster_labels:
            if label.label not in labels_and_text.keys():
                labels_and_text[label.label] = [example.text]
            else:
                labels_and_text[label.label].append(example.text)

    # pretty print results
    print("-----")
    for label, texts in labels_and_text.items():
        print("Label: ", label)
        for text in texts:
            print(repr(text))
        print("-----")


if __name__ == "__main__":
    main()
