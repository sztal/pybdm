"""Unit tests for encoding/decoding functions."""
import pytest
import numpy as np
from pybdm.encoding import encode_sequences
from pybdm.encoding import decode_sequences
from pybdm.encoding import normalize_sequences


@pytest.mark.parametrize('arr,base,expected', [
    (np.array([[1,0,2,0], [0,1,2,3]]), 4, np.array([72, 27]))
])
def test_encode_sequences(arr, base, expected):
    output = encode_sequences(arr, base=base)
    assert np.array_equal(output, expected)

@pytest.mark.parametrize('arr,base,expected', [
    (np.array([[1,0,2,0],[0,1,2,3]]), 4, np.array([[0,1,2,1],[0,1,2,3]]))
])
def test_normalize_sequences(arr, base, expected):
    output = normalize_sequences(arr, base=base)
    assert np.array_equal(output, expected)

@pytest.mark.parametrize('codes,shape,base,expected', [
    (8, (4,), 2, np.array([1, 0, 0, 0])),
    (np.array([3, 8]), (4,), 2, np.array([[0, 0, 1, 1], [1, 0, 0, 0]])),
    (np.array([3, 8]), (4,), 4, np.array([[0, 0, 0, 3], [0, 0, 2, 0]])),
])
def test_decode_sequences(codes, shape, base, expected):
    output = decode_sequences(codes, shape=shape, base=base)
    assert np.array_equal(output, expected)
