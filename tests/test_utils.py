import pytest
from pipegenie.utils import is_number, is_bool, is_tuple

@pytest.mark.parametrize(
    "value, expected",
    [
        ("10", True),
        ("10.5", True),
        ("-3.14", True),
        ("1e5", True),
        ("0", True),
        ("abc", False),
        ("", False),
        ("12a", False),
        (" ", False),
    ],
)
def test_is_number(value, expected):
    assert is_number(value) is expected

@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("false", True),
        ("True", True),
        ("False", True),
        ("TRUE", True),
        ("FALSE", True),
        ("yes", False),
        ("0", False),
        ("", False),
        ("falsee", False),
    ],
)
def test_is_bool(value, expected):
    assert is_bool(value) is expected

@pytest.mark.parametrize(
    "value, expected",
    [
        ("()", True),
        ("(1, 2)", True),
        ("(a, b, c)", True),
        ("(123)", True),
        ("(1,2", False),
        ("1,2)", False),
        ("[1,2]", False),
        ("", False),
        ("(", False),
        (")", False),
    ],
)
def test_is_tuple(value, expected):
    assert is_tuple(value) is expected
