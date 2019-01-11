"""Tests for the `algorithms` module."""
# pylint: disable=W0212,W0621
import pytest
import numpy as np
from bdm.base import BDMBase
from bdm.algorithms import PerturbationExperiment


@pytest.fixture(scope='function')
def perturbation(bdm_d2):
    X = np.ones((25, 25), dtype=int)
    return PerturbationExperiment(X, bdm_d2)

@pytest.fixture(scope='function')
def perturbation_overlap():
    X = np.ones((25, 25), dtype=int)
    bdm = BDMBase(ndim=2, shift=1)
    return PerturbationExperiment(X, bdm)


class TestPerturbationExperiment:

    @pytest.mark.parametrize('idx,expected', [
        ((0, 0), [(slice(0, 4), slice(0, 4))]),
        ((10, 4), [(slice(8, 12), slice(4, 8))]),
        ((2, 2), [(slice(0, 4), slice(0, 4))]),
        ((8, 3), [(slice(8, 12), slice(0, 4))]),
        ((21, 16), [(slice(20, 24), slice(16, 20))])
    ])
    def test_reduce_idx(self, perturbation, idx, expected):
        output = [ x for x in perturbation._idx_to_slices(idx) ]
        assert output == expected

    @pytest.mark.parametrize('idx,expected', [
        ((0, 0), [(slice(0, 4), slice(0, 4))]),
        ((0, 1), [(slice(0, 4), slice(0, 4)), (slice(0, 4), slice(1, 5))]),
        ((11, 15), [
            (slice(8, 12), slice(12, 16)), (slice(8, 12), slice(13, 17)),
            (slice(8, 12), slice(14, 18)), (slice(8, 12), slice(15, 19)),
            (slice(9, 13), slice(12, 16)), (slice(9, 13), slice(13, 17)),
            (slice(9, 13), slice(14, 18)), (slice(9, 13), slice(15, 19)),
            (slice(10, 14), slice(12, 16)), (slice(10, 14), slice(13, 17)),
            (slice(10, 14), slice(14, 18)), (slice(10, 14), slice(15, 19)),
            (slice(11, 15), slice(12, 16)), (slice(11, 15), slice(13, 17)),
            (slice(11, 15), slice(14, 18)), (slice(11, 15), slice(15, 19))
        ])
    ])
    def test_reduce_idx_recursive(self, perturbation_overlap, idx,  expected):
        output = [ x for x in perturbation_overlap._idx_to_slices(idx) ]
        assert output == expected

    @pytest.mark.parametrize('idx,value,keep_changes,expected', [
        ((0, 0), None, False, 0)
    ])
    def test_perturb(self, idx, value, keep_changes, expected):
        pass
