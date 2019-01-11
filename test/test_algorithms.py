"""Tests for the `algorithms` module."""
# pylint: disable=W0212,W0621
import pytest
from pytest import approx
import numpy as np
from bdm.base import BDMBase
from bdm.algorithms import PerturbationExperiment


@pytest.fixture(scope='function')
def perturbation(bdm_d2):
    np.random.seed(1001)
    X = np.random.randint(0, 2, (25, 25), dtype=int)
    return PerturbationExperiment(X, bdm_d2)

@pytest.fixture(scope='function')
def perturbation_overlap():
    np.random.seed(1001)
    X = np.random.randint(0, 2, (25, 25), dtype=int)
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
    def test_idx_to_parts(self, perturbation, idx, expected):
        expected = [ perturbation.X[s] for s in expected ]
        output = [ x for x in perturbation._idx_to_parts(idx) ]
        assert len(output) == len(expected)
        for o, e in zip(output, expected):
            assert np.array_equal(o, e)

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
    def test_idx_to_parts_overlap(self, perturbation_overlap, idx,  expected):
        perturbation = perturbation_overlap
        expected = [ perturbation.X[s] for s in expected ]
        output = [ x for x in perturbation._idx_to_parts(idx) ]
        assert len(output) == len(expected)
        for o, e in zip(output, expected):
            assert np.array_equal(o, e)

    @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
    @pytest.mark.parametrize('value', [1, 0, None])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb(self, perturbation, idx, value, keep_changes):
        X0 = perturbation.X.copy()
        X1 = X0.copy()
        if value is not None:
            X1[idx] = value
        else:
            X1[idx] = 0 if X0[idx] == 1 else 1
        bdm0 = perturbation.bdm.bdm(X0)
        bdm1 = perturbation.bdm.bdm(X1)
        expected = bdm1 - bdm0
        output = perturbation.perturb(idx, value=value, keep_changes=keep_changes)
        assert output == approx(expected)
        if not keep_changes:
            assert np.array_equal(X0, perturbation.X)

    @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
    @pytest.mark.parametrize('value', [1, 0, None])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_overlap(self, perturbation_overlap, idx, value, keep_changes):
        perturbation = perturbation_overlap
        X0 = perturbation.X.copy()
        X1 = X0.copy()
        if value is not None:
            X1[idx] = value
        else:
            X1[idx] = 0 if X0[idx] == 1 else 1
        bdm0 = perturbation.bdm.bdm(X0)
        bdm1 = perturbation.bdm.bdm(X1)
        expected = bdm1 - bdm0
        output = perturbation.perturb(idx, value=value, keep_changes=keep_changes)
        assert output == approx(expected)
        if not keep_changes:
            assert np.array_equal(X0, perturbation.X)
