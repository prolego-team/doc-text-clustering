import pytest

from io_utils import read_txt


@pytest.fixture
def text():
    """
    sk8er boi lyrics
    """
    return read_txt("test_data/sk8er_boi.txt")
