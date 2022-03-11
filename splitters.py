"""
splitters split a text string into a list of examples (List[TextExample])
"""

from typing import List

from data_utils import TextExample


class Splitter():
    """
    abstract base class
    """
    pass


class NewLineSplitter(Splitter):
    """
    split text by newline character ("\n")
    """

    def __init__(self, id_prefix: str = ""):
        """
        if id_prefix is supplied, it is appended
        at the beginning of the TextExample id
        """
        self.id_prefix = id_prefix

    def __call__(self, text: str) -> List[TextExample]:
        """
        split input text by newline and populate a list
        of examples
        """
        splitted_text = text.split("\n")
        examples = []
        for i, text in enumerate(splitted_text):
            examples.append(
                TextExample(
                    text=text,
                    id=self.id_prefix + str(i),
                    embedding=None,
                    labels=None
                )
            )
        return examples
