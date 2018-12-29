"""Tests for utility functions."""
import pytest
import numpy as np
from bdm.utils import get_reduced_shape, get_reduced_idx, slice_dataset


@pytest.mark.parametrize('x,shape,shift,length_only,expected', [
    (np.ones((50, 10)), (4, 4), 0, False, (13, 3)),
    (np.ones((50, 10)), (4, 4), 0, True, 39),
    (np.ones((50, 10)), (4, 4), 1, False, (47, 7)),
    (np.ones((4, 8)), (4, 4), 0, False, (1, 2)),
    (np.ones((4, 8)), (4, 4), 1, False, (1, 5))
])
def test_get_reduced_shape(x, shape, shift, length_only, expected):
    output = get_reduced_shape(x, shape, shift=shift, length_only=length_only)
    assert output == expected

@pytest.mark.parametrize('i,shape,expected', [
    (0, (2, 2, 2), (0, 0, 0)),
    (1, (2, 2, 2), (0, 0, 1)),
    (2, (2, 2, 2), (0, 1, 0)),
    (3, (2, 2, 2), (0, 1, 1)),
    (4, (2, 2, 2), (1, 0, 0)),
    (5, (2, 2, 2), (1, 0, 1)),
    (6, (2, 2, 2), (1, 1, 0)),
    (7, (2, 2, 2), (1, 1, 1)),
    (0, (1, 2), (0, 0)),
    (1, (1, 2), (0, 1))
])
def test_get_reduced_idx(i, shape, expected):
    output = get_reduced_idx(i, shape)
    assert output == expected

@pytest.mark.parametrize('X,shape,shift,expected', [
    (np.ones((10, 5)), (5, 5), 0, (np.ones((5, 5)), np.ones((5, 5)))),
    (np.ones((6, 6)), (5, 5), 1, (np.ones((5,5)), np.ones((5,5)), np.ones((5,5)), np.ones((5,5))))
])
def test_slice_dataset(X, shape, shift, expected):
    for o, e in zip(slice_dataset(X, shape, shift), expected):
        assert np.array_equal(o, e)
