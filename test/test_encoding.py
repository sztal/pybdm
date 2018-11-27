"""Unit tests for encoding/decoding functions."""
import pytest
import numpy as np
from bdm.encoding import encode_sequence, decode_sequence
from bdm.encoding import encode_array, decode_array
from bdm.encoding import array_from_string, string_from_array


@pytest.mark.parametrize('x,shape,expected', [
    ('', None, np.array([])),
    ('1010', None, np.array([1, 0, 1, 0])),
    ('1-0', None, np.array([[1], [0]])),
    ('0000-1000-0101', None, np.array([[0,0,0,0], [1,0,0,0], [0,1,0,1]])),
    ('12-34-56-78', (2, 2, 2), np.array([[[1,2],[3,4]], [[5,6],[7,8]]]))
])
def test_array_from_string(x, shape, expected):
    output = array_from_string(x, shape=shape)
    assert np.array_equal(output, expected)

@pytest.mark.parametrize('arr,expected', [
    (np.array([]), ''),
    (np.array([[0,0,0,0], [1,0,0,0], [0,1,0,1]]), '0000-1000-0101')
])
def test_string_from_array(arr, expected):
    output = string_from_array(arr)
    assert output == expected

@pytest.mark.parametrize('seq,base,expected', [
    (np.array([]), 2, 0),
    (np.array([1, 0, 0, 1]), 2, 9),
    (np.array([2, 0, 0, 1]), 3, 55)
])
def test_encode_sequence(seq, base, expected):
    output = encode_sequence(seq, base=base)
    assert output == expected

@pytest.mark.parametrize('code,base,min_length,expected', [
    (0, 2, None, np.array([])),
    (9, 2, None, np.array([1, 0, 0, 1])),
    (28, 3, None, np.array([1, 0, 0, 1])),
    (28, 4, 5, np.array([0, 0, 1, 3, 0]))
])
def test_decode_sequence(code, base, min_length, expected):
    output = decode_sequence(code, base=base, min_length=min_length)
    assert np.array_equal(output, expected)

@pytest.mark.parametrize('x,base,expected', [
    (np.array([]), 7, 0),
    (np.array([1, 1, 0]), 2, 6),
    (np.array([0, 0, 1, 0, 1, 0]), 4, 68)
])
def test_encode_array(x, base, expected):
    output = encode_array(x, base=base)
    assert output == expected

@pytest.mark.parametrize('code,shape,base,expected', [
    (17, (3, 3), 2, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])),
    (7, (2, 2), 3, np.array([[0, 0], [2, 1]]))
])
def test_decode_array(code, shape, base, expected):
    output = decode_array(code, shape, base=base)
    assert np.array_equal(output, expected)
