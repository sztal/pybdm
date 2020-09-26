"""Unit tests for decompose module."""
import pytest
import numpy as np
from pybdm.blocks import get_block_shape, get_block_slice, get_block_idx
from pybdm.blocks import iter_block_slices, block_decompose


@pytest.mark.parametrize('X,shape,expected', [
    (np.ones((10, 10)), (2, 2), (5, 5)),
    (np.ones((10, 10)), (4, 2), (3, 5)),
    (np.ones((10, 10)), (4, 4), (3, 3)),
    (np.ones((1000, 1011)), (4, 4), (250, 253)),
])
def test_get_block_shape(X, shape, expected):
    output = get_block_shape(X, shape=shape)
    assert output == expected

@pytest.mark.parametrize('idx,shape,expected', [
    ((2, 2), (4, 4), (slice(8, 12), slice(8, 12))),
    ((1, 5), (4, 4), (slice(4, 8), slice(20, 24)))
])
def test_get_block_slice(idx, shape, expected):
    output = get_block_slice(idx, shape)
    assert output == expected

@pytest.mark.parametrize('idx,shape,expected', [
    ((55,), (12,), (4,)),
    ((3, 7), (4, 4), (0, 1)),
    (np.array([[1, 1], [5, 10]]), (4, 4), np.array([[0, 0], [1, 2]])),
    (np.array([[1, 1], [1, 2]]), (4, 4), np.array([[0, 0]]))
])
def test_get_block_idx(idx, shape, expected):
    output = get_block_idx(idx, shape)
    if isinstance(output, np.ndarray):
        assert np.array_equal(output, expected)
    else:
        assert output == expected

@pytest.mark.parametrize('X,shape,expected', [
    (np.arange(8).reshape(4, 2), (2, 2),
     [(slice(0, 2), slice(0, 2)), (slice(2, 4), slice(0, 2))])
])
def test_iter_block_slices(X, shape, expected):
    output = list(iter_block_slices(X, shape))
    assert output == expected

@pytest.mark.parametrize('X,shape,expected', [
    (np.arange(8).reshape(4, 2), (2, 2),
     [np.arange(4).reshape(2, 2), np.arange(4, 8).reshape(2, 2)])
])
def test_block_decompose(X, shape, expected):
    output = list(block_decompose(X, shape))
    assert all(np.array_equal(x, y) for x, y in zip(expected, output))
