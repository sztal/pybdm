"""Unit tests for encoding/decoding functions."""
import pytest
import numpy as np
from pybdm.encoding import encode_sequences
from pybdm.encoding import decode_array, decode_sequence
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

@pytest.mark.parametrize('code,shape,base,expected', [
    (0, None, 2, np.array([])),
    (9, None, 2, np.array([1, 0, 0, 1])),
    (28, None, 3, np.array([1, 0, 0, 1])),
    (447, (8,), 4, np.array([0, 0, 0, 1, 2, 3, 3, 3]))
])
def test_decode_sequence(code, shape, base, expected):
    output = decode_sequence(code, shape=shape, base=base)
    assert np.array_equal(output, expected)

@pytest.mark.parametrize('code,shape,base,expected', [
    (17, (3, 3), 2, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])),
    (7, (2, 2), 3, np.array([[0, 0], [2, 1]]))
])
def test_decode_array(code, shape, base, expected):
    output = decode_array(code, shape, base=base)
    assert np.array_equal(output, expected)
