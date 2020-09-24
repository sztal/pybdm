"""Tests for `bdm` module."""
import os
import pytest
from pytest import approx
import numpy as np
from joblib import Parallel, delayed
from pybdm.bdm import BDM
from pybdm.decompose import block_decompose


# Helpers ---------------------------------------------------------------------

def array_from_string(s, shape=None, dtype=int):
    # pylint: disable=redefined-outer-name
    arr = np.array(list(s), dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr

# -----------------------------------------------------------------------------


s0 = '0'*24
s1 = '0'*12+'1'*12
s2 = '0'*6+'1'*12+'0'*18+'1'*12

bdm1_test_input = [(array_from_string(s0, (24,)), 26.610413747641715),
                   (array_from_string(s1, (24,)), 51.22082749528343),
                   (array_from_string(s2, (48,)), 114.06151272200972)]

_dirpath = os.path.join(os.path.split(__file__)[0], 'data')
# Get test input data and expected values
bdm2_test_input = []
with open(os.path.join(_dirpath, 'bdm-b2-d4x4-test-input.tsv'), 'r') as stream:
    for line in stream:
        string, bdm = line.strip().split("\t")
        bdm = float(bdm.strip())
        arr = np.array([
            array_from_string(x, (len(x),)) for x in
            string.strip().split('-')
        ])
        bdm2_test_input.append((arr, bdm))


ent1_test_input = [(array_from_string(s0, (24,)), 0.0),
                   (array_from_string(s1, (24,)), 1.0),
                   (array_from_string(s2, (48,)), 2.0)]

ent2_test_input = []
with open(os.path.join(_dirpath, 'ent-b2-d4x4-test-input.tsv'), 'r') as stream:
    for line in stream:
        string, ent2 = line.strip().split(",")
        ent2 = float(ent2.strip())
        arr = np.array([
            array_from_string(x, (len(x),)) for x in
            string.strip().split('-')
        ])
        ent2_test_input.append((arr, ent2))


class TestBDM:

    @pytest.mark.parametrize('ndim', (1, 2, 3))
    @pytest.mark.parametrize('nsymbols', (2, 10))
    @pytest.mark.parametrize('nstates', (4, 5, 6))
    @pytest.mark.parametrize('shape', [(4, 4), (4, 3)])
    @pytest.mark.parametrize('min_length', (2, 12))
    def test_bdm_init(self, ndim, nsymbols, nstates, shape, min_length):
        # pylint: disable=unused-variable
        try:
            bdm1 = BDM(
                ndim=ndim,
                nsymbols=nsymbols,
                partition='ignore'
            )
            bdm1 = BDM(
                ndim=ndim,
                nsymbols=nsymbols,
                partition='recursive',
                min_length=min_length,
            )
        except LookupError:
            assert (
                (ndim > 2) or
                (nstates not in (4, 5)) or
                (nsymbols not in (2, 4, 5, 6, 9)) or
                (nsymbols == 2 and nstates != 5) or
                (nsymbols > 2 and nstates != 4)
            )
        except AttributeError:
            assert len(set(shape)) != 1

    @pytest.mark.parametrize('X,expected', bdm1_test_input)
    def test_bdm_d1(self, bdm_d1, X, expected):
        output = bdm_d1.bdm(X)
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', [
        ([0], 4.964344),
        ([0,0,0], 11.66997),
        ([0,1,2,3,4,5,6,7,8,8,8,5], 49.712),
        ([2,1,0,3,4,5,6,7,8,8,8,5], 49.712),
        ([2,1,0,3,4,5,6,7,8,8,8,5,4,2,4], 49.712 + 11.90539),
        ([4,1,2,1,5,4,0,5,1,8,4,2], 53.897870350),
        ([4,1,2,1,5,4,0,5,1,8,4,2,0,1,2,3,4,
          5,6,7,8,8,8,5,0,1,2,3,4,5,6,7,8,8,8,5], 104.60987415)
    ])
    def test_bdm_d1_b9(self, bdm_d1_b9, X, expected):
        X = np.array(X)
        output = bdm_d1_b9.bdm(X)
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', bdm2_test_input)
    def test_bdm_d2(self, bdm_d2, X, expected):
        output = bdm_d2.bdm(X)
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', ent1_test_input)
    @pytest.mark.parametrize('rv', (True, False))
    def test_ent_d1(self, bdm_d1, X, expected, rv):
        output = bdm_d1.ent(X, rv=rv)
        if not rv:
            counter = bdm_d1.decompose_and_count(X)
            N = sum([ sum(c.values()) for c in counter.values() ])
            expected *= N
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', ent2_test_input)
    @pytest.mark.parametrize('rv', (True, False))
    def test_ent_d2(self, bdm_d2, X, expected, rv):
        output = bdm_d2.ent(X, rv=rv)
        if not rv:
            counter = bdm_d2.decompose_and_count(X)
            N = sum([ sum(c.values()) for c in counter.values() ])
            expected *= N
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', [
        (np.ones((30,), dtype=int), 0),
        (np.array([0,1,0,0,0,1,1,0,0,0,1,0,0,0,1], dtype=int), 0.648665654727082),
        (np.array([
            0,0,0,1,1,1,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,1,1,0,1,
            0,0,0,1,1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,0,0,1,0,
            0,1,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,
            1,0,0,0,1,0,1,0,0,1,1,0,1,0,0,0,0,0,1,0,1,1,1,0,
            0,0,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,
            0,0,1,1,0,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,1,
            0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,0,
            0,1,1,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1
        ], dtype=int), 0.9012807791010917)
    ])
    def test_nbdm_d1(self, bdm_d1, X, expected):
        output = bdm_d1.nbdm(X)
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', [
        (np.ones((20, 20), dtype=int), 0),
        (np.array([[0,0,1,0],[1,0,0,1],[0,0,1,1],[1,0,1,0]], dtype=int), 0.6139131118181638)
    ])
    def test_nbdm_d2(self, bdm_d2, X, expected):
        output = bdm_d2.nbdm(X)
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', [
        (np.ones((30,), dtype=int), 0),
        (np.array([0 for i in range(12)]+[1 for i in range(12)], dtype=int), 1)
    ])
    def test_nent_d1(self, bdm_d1, X, expected):
        output = bdm_d1.nent(X)
        assert output == approx(expected)

    @pytest.mark.parametrize('X,expected', [
        (np.ones((20, 20), dtype=int), 0),
        (np.vstack((np.ones((4, 4), dtype=int), np.zeros((4, 4), dtype=int))), 1)
    ])
    def test_nent_d2(self, bdm_d2, X, expected):
        output = bdm_d2.nent(X)
        assert output == approx(expected)

    @pytest.mark.slow
    def test_bdm_parallel(self, bdm_d2):
        X = np.ones((5000, 5000), dtype=int)
        expected = bdm_d2.bdm(X)
        counters = Parallel(n_jobs=2) \
            (delayed(bdm_d2.decompose_and_count)(d)
             for d in block_decompose(X, (1000, 1000)))
        output = bdm_d2.calc_bdm(*counters)
        assert output == approx(expected)
