"""Tests for the `algorithms` module."""
# pylint: disable=redefined-outer-name,protected-access
from collections.abc import Mapping
from random import choice
import pytest
from pytest import approx
import numpy as np
from pybdm.bdm import BDM
from pybdm.perturbation import Perturbation, PerturbationStep


PS = PerturbationStep


@pytest.fixture(scope='function')
def perturbation_d2(bdm_d2):
    np.random.seed(1001)
    X = np.random.randint(0, 2, (25, 25), dtype=int)
    return Perturbation(bdm_d2, X)

@pytest.fixture(scope='function')
def perturbation_d1(bdm_d1):
    np.random.seed(999)
    X = np.random.randint(0, 2, (100, ), dtype=int)
    return Perturbation(bdm_d1, X)

@pytest.fixture(scope='function')
def perturbation_d1_b9(bdm_d1_b9):
    np.random.seed(10101)
    X = np.random.randint(0, 9, (100,), dtype=int)
    return Perturbation(bdm_d1_b9, X)

@pytest.fixture(scope='function')
def perturbation_ent(bdm_d2):
    np.random.seed(1001)
    X = np.random.randint(0, 2, (25, 25), dtype=int)
    return Perturbation(bdm_d2, X, metric='ent')

@pytest.fixture(scope='function')
def perturbation_d1_ent(bdm_d1):
    np.random.seed(999)
    X = np.random.randint(0, 2, (100, ), dtype=int)
    return Perturbation(bdm_d1, X, metric='ent')


class TestStep:

    @pytest.mark.parametrize('idx,expected', [
        ((1, 1), np.array([[1, 1]])),
        ([(1, 1), (2, 1)], np.array([[1,1],[2,1]])),
        (((1, 1), (0, 1)), np.array([[1, 1],[0,1]])),
        (np.array([[0,1],[0,0]]), np.array([[0,1],[0,0]])),
        (lambda x: x != 0, np.argwhere),
        (None, lambda x: np.argwhere(np.ones_like(x, dtype=bool)))
    ])
    def test_idx(self, idx, expected, perturbation_d2):
        step = PerturbationStep(idx)
        step.bind(perturbation_d2)
        output = step.idx
        if callable(expected):
            expected = expected(step.ctx.X)
        for o, e in zip(output, expected):
            if isinstance(o, np.ndarray):
                assert np.array_equal(o, e)
            else:
                assert o == e

    @pytest.mark.parametrize('newvals', [
        None, 1, { 0: 2, 1: 77 }, lambda x: 0 if x == 1 else x,
        np.ones((625,), dtype=int)
    ])
    def test_newvals(self, newvals, perturbation_d2):
        step = PerturbationStep(newvals=newvals)
        step.bind(perturbation_d2)
        output = step.newvals
        if np.isscalar(newvals) or isinstance(newvals, np.ndarray):
            expected = newvals
        elif isinstance(newvals, Mapping):
            expected = np.array([
                newvals[v] for v in step.ctx.X[step.vidx]
            ])
        elif callable(newvals):
            expected = np.array([
                newvals(v) for v in step.ctx.X[step.vidx]
            ])
        elif newvals is None:
            expected = np.where(step.ctx.X[step.vidx] == 0, 1, 0)
        if isinstance(output, np.ndarray):
            assert np.array_equal(output, expected)
        else:
            assert output == expected

    @pytest.mark.parametrize('step,expected', [
        (PS((0,1)), [np.array([[0,1]])]),
        (PS([(0,1),(0,0)]), [np.array([[0,1]]), np.array([[0,0]])]),
        (PS(np.array([[0,1],[0,0]])), [np.array([[0,1]]),np.array([[0,0]])])
    ])
    def test_to_sequence(self, step, expected, perturbation_d2):
        step.bind(perturbation_d2)
        output = [ s.idx for s in step.to_sequence() ]
        assert len(output) == len(expected)
        assert all(np.array_equal(o, e) for o, e in zip(output, expected))

@pytest.mark.slow
class TestPerturbationExperiment:

    def _assert_step(self, step, keep_changes, batch, perturbation):
        X0 = perturbation.X.copy()
        cmx0 = perturbation.bdm.bdm(X0)
        assert cmx0 == perturbation._cmx
        cmx1 = perturbation.make_step(step, keep_changes=keep_changes,
                                      batch=batch)
        assert cmx1 != cmx0
        if keep_changes:
            assert cmx1 == perturbation._cmx
        else:
            assert perturbation._cmx == cmx0
            assert np.array_equal(perturbation.X, X0)

    @pytest.mark.parametrize('step', [
        PS((0, 0)),
        PS(np.argwhere, batch=True)
    ])
    @pytest.mark.parametrize('keep_changes', [False, True])
    @pytest.mark.parametrize('batch', [False, True])
    def test_make_step_d2(self, step, keep_changes, batch, perturbation_d2):
        self._assert_step(step, keep_changes, batch, perturbation_d2)

    @pytest.mark.parametrize('step', [
        (np.array([0, 6, 13]),),
        { 'idx': np.argwhere, 'batch': True }
    ])
    @pytest.mark.parametrize('keep_changes', [False, True])
    @pytest.mark.parametrize('batch', [False, True])
    def test_make_step_d1_b2(self, step, keep_changes, batch, perturbation_d1):
        self._assert_step(step, keep_changes, batch, perturbation_d1)

    @pytest.mark.parametrize('step', [
        (np.array([1, 14, 26, 44, 4]),),
        PS(np.argwhere, batch=True)
    ])
    @pytest.mark.parametrize('keep_changes', [False, True])
    @pytest.mark.parametrize('batch', [False, True])
    def test_make_step_d1_b9(self, step, keep_changes, batch, perturbation_d1_b9):
        self._assert_step(step, keep_changes, batch, perturbation_d1_b9)

#     @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb(self, perturbation, idx, value, keep_changes):
#         self._assert_perturb(
#             perturbation, idx, value, keep_changes, metric='bdm'
#         )

#     @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_overlap(self, perturbation_overlap, idx, value, keep_changes):
#         self._assert_perturb(
#             perturbation_overlap, idx, value, keep_changes, metric='bdm'
#         )

#     @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_d1(self, perturbation_d1, idx, value, keep_changes):
#         self._assert_perturb(
#             perturbation_d1, idx, value, keep_changes, metric='bdm'
#         )

#     @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_d1_overlap(self, perturbation_d1_overlap, idx, value, keep_changes):
#         self._assert_perturb(
#             perturbation_d1_overlap, idx, value, keep_changes, metric='bdm'
#         )

#     @pytest.mark.parametrize('idx', [(0,), (1,), (55,), (99,)])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_d1_b9(self, perturbation_d1_b9, idx, value, keep_changes):
#         self._assert_perturb(
#             perturbation_d1_b9, idx, value, keep_changes, metric='bdm'
#         )

#     @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_ent(self, perturbation_ent, idx, value, keep_changes):
#         self._assert_perturb(
#             perturbation_ent, idx, value, keep_changes, metric='ent'
#         )

#     @pytest.mark.parametrize('idx', [(0, 0), (1, 0), (10, 15), (24, 24)])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_ent_overlap(self, perturbation_ent_overlap, idx,
#                                  value, keep_changes):
#         self._assert_perturb(
#             perturbation_ent_overlap, idx, value, keep_changes, metric='ent'
#         )

#     @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_d1_ent(self, perturbation_d1_ent, idx, value, keep_changes):
#         self._assert_perturb(
#             perturbation_d1_ent, idx, value, keep_changes, metric='ent'
#         )

#     @pytest.mark.parametrize('idx', [(0, ), (1, ), (55, ), (99, )])
#     @pytest.mark.parametrize('value', [1, 0, -1])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_perturb_d1_ent_overlap(self, perturbation_d1_ent_overlap, idx,
#                                     value, keep_changes):
#         self._assert_perturb(
#             perturbation_d1_ent_overlap, idx, value, keep_changes, metric='ent'
#         )

#     def _assert_run(self, perturbation, idx, values, keep_changes):
#         X0 = perturbation.X.copy()
#         output = perturbation.run(idx, values, keep_changes=keep_changes)
#         if idx is None:
#             N_changes = prod(X0.shape)
#         else:
#             N_changes = idx.shape[0]
#         assert output.shape[0] == N_changes
#         if keep_changes:
#             assert (X0 != perturbation.X).sum() == N_changes
#             if idx is not None:
#                 if idx.ndim == 1:
#                     idx = np.expand_dims(idx, 1)
#                 for row in idx:
#                     _idx = tuple(row)
#                     assert X0[_idx] != perturbation.X[_idx]
#         else:
#             assert np.array_equal(X0, perturbation.X)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([[0, 0], [0, 5], [10, 15]], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([[0, 1], [10, 10]], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run(self, perturbation, idx, values, keep_changes):
#         self._assert_run(perturbation, idx, values, keep_changes)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([[0, 0], [0, 5], [10, 15]], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([[0, 1], [10, 10]], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run_overlap(self, perturbation_overlap, idx, values, keep_changes):
#         self._assert_run(perturbation_overlap, idx, values, keep_changes)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([0, 5, 15], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([1, 10], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run_d1(self, perturbation_d1, idx, values, keep_changes):
#         self._assert_run(perturbation_d1, idx, values, keep_changes)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([0, 5, 15], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([1, 10], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run_d1_overlap(self, perturbation_d1_overlap, idx, values, keep_changes):
#         self._assert_run(perturbation_d1_overlap, idx, values, keep_changes)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([[0, 0], [0, 5], [10, 15]], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([[0, 1], [10, 10]], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run_ent(self, perturbation_ent, idx, values, keep_changes):
#         self._assert_run(perturbation_ent, idx, values, keep_changes)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([[0, 0], [0, 5], [10, 15]], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([[0, 1], [10, 10]], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run_ent_overlap(self, perturbation_ent_overlap, idx, values, keep_changes):
#         self._assert_run(perturbation_ent_overlap, idx, values, keep_changes)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([0, 5, 15], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([1, 10], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run_ent_d1(self, perturbation_d1_ent, idx, values, keep_changes):
#         self._assert_run(perturbation_d1_ent, idx, values, keep_changes)

#     @pytest.mark.parametrize('idx,values', [
#         (np.array([0, 5, 15], dtype=int), np.array([-1, -1, -1], dtype=int)),
#         (None, None),
#         (np.array([1, 10], dtype=int), None)
#     ])
#     @pytest.mark.parametrize('keep_changes', [True, False])
#     def test_run_ent_d1_overlap(self, perturbation_d1_ent_overlap, idx, values, keep_changes):
#         self._assert_run(perturbation_d1_ent_overlap, idx, values, keep_changes)
