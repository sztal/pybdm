"""Tests for the `algorithms` module."""
# pylint: disable=W0212,W0621
# pylint: disable=R0914
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

@pytest.fixture(scope='function')
def perturbation_d1(bdm_d1):
    np.random.seed(999)
    X = np.random.randint(0, 2, (100, ), dtype=int)
    return PerturbationExperiment(X, bdm_d1)

@pytest.fixture(scope='function')
def perturbation_d1_overlap():
    np.random.seed(99)
    X = np.random.randint(0, 2, (100, ), dtype=int)
    bdm = BDMBase(ndim=1, shift=1)
    return PerturbationExperiment(X, bdm)

@pytest.fixture(scope='function')
def perturbation_ent(bdm_d2):
    np.random.seed(1001)
    X = np.random.randint(0, 2, (25, 25), dtype=int)
    return PerturbationExperiment(X, bdm_d2, metric='entropy')

@pytest.fixture(scope='function')
def perturbation_ent_overlap():
    np.random.seed(1001)
    X = np.random.randint(0, 2, (25, 25), dtype=int)
    bdm = BDMBase(ndim=2, shift=1)
    return PerturbationExperiment(X, bdm, metric='entropy')

@pytest.fixture(scope='function')
def perturbation_d1_ent(bdm_d1):
    np.random.seed(999)
    X = np.random.randint(0, 2, (100, ), dtype=int)
    return PerturbationExperiment(X, bdm_d1, metric='entropy')

@pytest.fixture(scope='function')
def perturbation_d1_ent_overlap():
    np.random.seed(99)
    X = np.random.randint(0, 2, (100, ), dtype=int)
    bdm = BDMBase(ndim=1, shift=1)
    return PerturbationExperiment(X, bdm, metric='entropy')


@pytest.mark.slow
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

    def _assert_perturb(self, perturbation, idx, value, keep_changes, metric='bdm'):
        X0 = perturbation.X.copy()
        X1 = X0.copy()
        C0 = perturbation._counter.copy()
        if value >= 0:
            X1[idx] = value
        else:
            X1[idx] = 0 if X0[idx] == 1 else 1
        C1 = perturbation.bdm.count_and_lookup(X1)
        if metric == 'bdm':
            x0 = perturbation.bdm.bdm(X0)
            x1 = perturbation.bdm.bdm(X1)
        else:
            x0 = perturbation.bdm.entropy(X0)
            x1 = perturbation.bdm.entropy(X1)
        expected = x1 - x0
        output = perturbation.perturb(idx, value=value, keep_changes=keep_changes)
        assert output == approx(expected)
        if keep_changes and abs(expected) > 0:
            assert (X0 != X1).sum() == 1
            assert X0[idx] != X1[idx]
            assert perturbation._counter == C1
            assert perturbation._counter != C0
        elif not keep_changes:
            assert np.array_equal(X0, perturbation.X)
            assert perturbation._counter == C0

    @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb(self, perturbation, idx, value, keep_changes):
        self._assert_perturb(
            perturbation, idx, value, keep_changes, metric='bdm'
        )

    @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_overlap(self, perturbation_overlap, idx, value, keep_changes):
        self._assert_perturb(
            perturbation_overlap, idx, value, keep_changes, metric='bdm'
        )

    @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_d1(self, perturbation_d1, idx, value, keep_changes):
        self._assert_perturb(
            perturbation_d1, idx, value, keep_changes, metric='bdm'
        )

    @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_d1_overlap(self, perturbation_d1_overlap, idx, value, keep_changes):
        self._assert_perturb(
            perturbation_d1_overlap, idx, value, keep_changes, metric='bdm'
        )

    @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_ent(self, perturbation_ent, idx, value, keep_changes):
        self._assert_perturb(
            perturbation_ent, idx, value, keep_changes, metric='entropy'
        )

    @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_ent_overlap(self, perturbation_ent_overlap, idx,
                                 value, keep_changes):
        self._assert_perturb(
            perturbation_ent_overlap, idx, value, keep_changes, metric='entropy'
        )

    @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_d1_ent(self, perturbation_d1_ent, idx, value, keep_changes):
        self._assert_perturb(
            perturbation_d1_ent, idx, value, keep_changes, metric='entropy'
        )

    @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
    @pytest.mark.parametrize('value', [1, 0, -1])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_perturb_d1_ent_overlap(self, perturbation_d1_ent_overlap, idx,
                                    value, keep_changes):
        self._assert_perturb(
            perturbation_d1_ent_overlap, idx, value, keep_changes, metric='entropy'
        )

    def _assert_run(self, perturbation, changes, keep_changes):
        X0 = perturbation.X.copy()
        output = perturbation.run(changes, keep_changes=keep_changes)
        if changes is None:
            N_changes = np.multiply.reduce(X0.shape)
        else:
            N_changes = changes.shape[0]
        assert output.shape[0] == N_changes
        if keep_changes:
            assert (X0 != perturbation.X).sum() == N_changes
            if changes is not None:
                for row in changes:
                    idx = tuple(row[:-1])
                    assert X0[idx] != perturbation.X[idx]
        else:
            assert np.array_equal(X0, perturbation.X)

    @pytest.mark.parametrize('changes', [
        np.array([[0, 0, -1], [0, 5, -1], [10, 15, -1]]),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run(self, perturbation, changes, keep_changes):
        self._assert_run(perturbation, changes, keep_changes)

    @pytest.mark.parametrize('changes', [
        np.array([[0, 0, -1], [0, 5, -1], [10, 15, -1]]),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run_overlap(self, perturbation_overlap, changes, keep_changes):
        self._assert_run(perturbation_overlap, changes, keep_changes)

    @pytest.mark.parametrize('changes', [
        np.array([[0, -1], [5, -1], [15, -1]]),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run_d1(self, perturbation_d1, changes, keep_changes):
        self._assert_run(perturbation_d1, changes, keep_changes)

    @pytest.mark.parametrize('changes', [
        np.array([[0, -1], [5, -1], [15, -1]]),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run_d1_overlap(self, perturbation_d1_overlap, changes, keep_changes):
        self._assert_run(perturbation_d1_overlap, changes, keep_changes)

    @pytest.mark.parametrize('changes', [
        np.array([[0, 0, -1], [0, 5, -1], [10, 15, -1]]),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run_ent(self, perturbation_ent, changes, keep_changes):
        self._assert_run(perturbation_ent, changes, keep_changes)

    @pytest.mark.parametrize('changes', [
        np.array([[0, 0, -1], [0, 5, -1], [10, 15, -1]]),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run_ent_overlap(self, perturbation_ent_overlap, changes, keep_changes):
        self._assert_run(perturbation_ent_overlap, changes, keep_changes)

    @pytest.mark.parametrize('changes', [
        (np.array([[0, -1], [5, -1], [15, -1]])),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run_ent_d1(self, perturbation_d1_ent, changes, keep_changes):
        self._assert_run(perturbation_d1_ent, changes, keep_changes)

    @pytest.mark.parametrize('changes', [
        (np.array([[0, -1], [5, -1], [15, -1]])),
        None
    ])
    @pytest.mark.parametrize('keep_changes', [True, False])
    def test_run_ent_d1_overlap(self, perturbation_d1_ent_overlap, changes, keep_changes):
        self._assert_run(perturbation_d1_ent_overlap, changes, keep_changes)
