"""Tests for utility functions."""
import pytest
from pytest import approx
import numpy as np
from bdm import BDM
from bdm.utils import get_reduced_shape, get_reduced_idx, slice_dataset
from bdm.utils import make_min_data, make_max_data


@pytest.mark.parametrize('x,shape,shift,size_only,expected', [
    (np.ones((50, 10)), (4, 4), 0, False, (13, 3)),
    (np.ones((50, 10)), (4, 4), 0, True, 39),
    (np.ones((50, 10)), (4, 4), 1, False, (47, 7)),
    (np.ones((4, 8)), (4, 4), 0, False, (1, 2)),
    (np.ones((4, 8)), (4, 4), 1, False, (1, 5))
])
def test_get_reduced_shape(x, shape, shift, size_only, expected):
    output = get_reduced_shape(x, shape, shift=shift, size_only=size_only)
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

@pytest.mark.parametrize('shape,expected', [
    ((5, 5), np.zeros((5, 5), dtype=int))
])
def test_make_min_data(shape, expected):
    output = make_min_data(shape)
    assert np.array_equal(output, expected)

@pytest.mark.parametrize('shape,part_shape,expected', [
    ((5, 5), (4, 4), 36.02279026553976),
    ((30,), (12,), 74.95847260483343)
])
def test_make_max_data(shape, part_shape, expected):
    # pylint: disable=protected-access
    bdm = BDM(len(shape))
    output = make_max_data(shape, part_shape, bdm._ctm)
    assert bdm.bdm(output) == approx(expected)
