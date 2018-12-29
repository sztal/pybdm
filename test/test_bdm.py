"""Tests for `bdm` module."""
# pylint: disable=W0621
import os
import pytest
import numpy as np
from pytest import approx
from joblib import Parallel, delayed
from bdm.encoding import array_from_string
from bdm.utils import slice_dataset

s0 = '0'*24
s1 = '0'*12+'1'*12
s2 = '0'*6+'1'*12+'0'*18+'1'*12

bdm1_test_input = [(array_from_string(s0), 26.610413747641715),
                   (array_from_string(s1), 51.22082749528343),
                   (array_from_string(s2), 114.06151272200972)]

_dirpath = os.path.join(os.path.split(__file__)[0], 'data')
# Get test input data and expected values
bdm2_test_input = []
with open(os.path.join(_dirpath, 'bdm-b2-d4x4-test-input.tsv'), 'r') as stream:
    for line in stream:
        string, bdm = line.strip().split("\t")
        bdm = float(bdm.strip())
        arr = array_from_string(string.strip())
        bdm2_test_input.append((arr, bdm))


ent1_test_input = [(array_from_string(s0), 0.0),
                   (array_from_string(s1), 1.0),
                   (array_from_string(s2), 2.0)]

ent2_test_input = []
with open(os.path.join(_dirpath, 'ent-b2-d4x4-test-input.tsv'), 'r') as stream:
    for line in stream:
        string, ent2 = line.strip().split(",")
        ent2 = float(ent2.strip())
        arr = array_from_string(string.strip())
        ent2_test_input.append((arr, ent2))


class TestBDM:

    @pytest.mark.parametrize('x,expected', bdm1_test_input)
    def test_complexity_d1(self, bdm_d1, x, expected):
        output = bdm_d1.bdm(x)
        assert output == approx(expected)

    @pytest.mark.parametrize('x,expected', bdm2_test_input)
    def test_complexity_d2(self, bdm_d2, x, expected):
        output = bdm_d2.bdm(x)
        assert output == approx(expected)

    @pytest.mark.parametrize('x,expected', ent1_test_input)
    def test_entropy_d1(self, bdm_d1, x, expected):
        output = bdm_d1.entropy(x)
        assert output == approx(expected)

    @pytest.mark.parametrize('x,expected', ent2_test_input)
    def test_entropy_d2(self, bdm_d2, x, expected):
        output = bdm_d2.entropy(x)
        assert output == approx(expected)

    @pytest.mark.slow
    def test_bdm_parallel(self, bdm_d2):
        X = np.ones((500, 500), dtype=int)
        expected = bdm_d2.bdm(X)
        counters = Parallel(n_jobs=2) \
            (delayed(bdm_d2.count_and_lookup)(d)
             for d in slice_dataset(X, (100, 100)))
        output = bdm_d2.compute_bdm(*counters)
        assert output == approx(expected)
