"""Automatic test of the *transpose conjecture*.

It checks whether CTM values are transpose invariant.
This test is slow.
"""
# pylint: disable=protected-access
import pytest
from pytest import approx
import numpy as np
from bdm.encoding import decode_array


@pytest.mark.slow
def test_transpose_conjecture(bdm_d2):
    for i in range(2**16):
        arr = decode_array(i, (4, 4))
        assert bdm_d2.bdm(arr) == approx(bdm_d2.bdm(arr.T))

@pytest.mark.slow
def test_complement_conjecture_d1(bdm_d1):
    for i in range(2**12):
        arr = decode_array(i, (12, ))
        c_arr = np.where(arr == 1, 0, 1)
        assert bdm_d1.bdm(arr) == approx(bdm_d1.bdm(c_arr))

@pytest.mark.slow
def test_complement_conjecture_d2(bdm_d2):
    for i in range(2**16):
        arr = decode_array(i, (4, 4))
        c_arr = np.where(arr == 1, 0, 1)
        assert bdm_d2.bdm(arr) == approx(bdm_d2.bdm(c_arr))

def test_flip_complemente_1d(bdm_d1):

    def flip(s):
        return ''.join('1' if x == '0' else '0' for x in s)

    ctm = bdm_d1._ctm
    for i in range(1, 13):
        shp = (i, )
        dct = ctm[shp]
        for s in dct.keys():
            fs = flip(s)
            assert dct[s] == dct[fs]
