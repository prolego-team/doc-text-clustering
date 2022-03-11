"""
utilty methods related to reading / writing data
"""

def read_txt(txt_filepath: str) -> str:
    """
    read a .txt file
    """
    with open(txt_filepath, "r") as f:
        text = f.read()
    return text
