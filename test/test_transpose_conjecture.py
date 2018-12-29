"""Automatic test of the *transpose conjecture*.

It checks whether CTM values are transpose invariant.
This test is slow.
"""
import pytest
from pytest import approx
from bdm.encoding import decode_array


@pytest.mark.slow
def test_transpose_conjecture(bdm_d2):
    for i in range(2**16):
        arr = decode_array(i, (4, 4))
        assert bdm_d2.bdm(arr) == approx(bdm_d2.bdm(arr.T))
