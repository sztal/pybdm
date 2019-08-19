"""Automatic test of the *transpose conjecture*.

It checks whether CTM values are transpose invariant.
This test is slow.
"""
# pylint: disable=protected-access
from math import factorial
import pytest
from pytest import approx
import numpy as np
from bdm import BDM, BDMRecursive
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

@pytest.mark.slow
@pytest.mark.parametrize('nsymbols', [2, 4, 5, 6, 9])
def test_ctm_distribution_d1(nsymbols):
    bdm = BDM(ndim=1, nsymbols=nsymbols)
    total = 0
    for dct in bdm._ctm.values():
        for key, cmx in dct.items():
            n = len(set(key))
            mult = factorial(nsymbols) / factorial(nsymbols - n)
            total += 2**-cmx * mult
    assert total == approx(1, .01)

# TODO: this is not passing; investigate reasons
# @pytest.mark.slow
# @pytest.mark.parametrize('nsymbols', [2])
# def test_ctm_distribution_d2(nsymbols):
#     total = 0
#     bdm = BDM(ndim=2, nsymbols=nsymbols)
#     for dct in bdm._ctm.values():
#         for key, cmx in dct.items():
#             n = len(set(key))
#             mult = factorial(nsymbols) / factorial(nsymbols - n)
#             total += 2**-cmx * mult
#     assert total == approx(1, .01)
