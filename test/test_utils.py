"""Tests for utility functions."""
import pytest
import numpy as np
from pybdm.utils import slice_dataset


@pytest.mark.parametrize('X,shape,shift,expected', [
    (np.ones((10, 5)), (5, 5), 0, (np.ones((5, 5)), np.ones((5, 5)))),
    (np.ones((6, 6)), (5, 5), 1, [ np.ones((5,5)) for _ in range(4) ])
])
def test_slice_dataset(X, shape, shift, expected):
    output = list(slice_dataset(X, shape=shape, shift=shift))
    assert all(np.array_equal(o, e) for o, e in zip(output, expected))
