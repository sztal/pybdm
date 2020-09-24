"""Tests for utility functions."""
import pytest
import numpy as np
from pybdm.utils import normalize_array, S2


@pytest.mark.parametrize('X,expected', [
    (np.arange(4), np.arange(4)),
    (np.arange(2, 7), np.arange(5)),
    (np.array([1, 1, 2, 0, 0, 3, 1]), np.array([0, 0, 1, 2, 2, 3, 0])),
    (np.array(['a', 'b', 'a', 'c', 'b', 'c']), np.array([0, 1, 0, 2, 1, 2]))
])
def test_normalize_array(X, expected):
    output = normalize_array(X)
    assert np.array_equal(output, expected)

@pytest.mark.parametrize('n,k,expected', [
    (1, 1, 1), (2, 1, 1), (2, 2, 1),
    (3, 1, 1), (3, 2, 3), (3, 3, 1),
    (4, 1, 1), (4, 2, 7), (4, 3, 6), (4, 4, 1),
    (5, 1, 1), (5, 2, 15), (5, 3, 25), (5, 4, 10), (5, 5, 1),
    (6, 1, 1), (6, 2, 31), (6, 3, 90), (6, 4, 65), (6, 5, 15), (6, 6, 1)
])
def test_S2(n, k, expected):
    output = S2(n, k)
    assert output == expected
