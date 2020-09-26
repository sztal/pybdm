"""Tests for the `algorithms` module."""
# pylint: disable=redefined-outer-name,protected-access
from collections.abc import Mapping
import pytest
from pytest import approx
import numpy as np
from pybdm.perturbation import Perturbation, PerturbationStep
from pybdm.utils import random_idx


PS = PerturbationStep


@pytest.fixture(scope='function')
def perturbation_d2(bdm_d2):
    np.random.seed(1001)
    X = np.random.randint(0, 2, (100, 100), dtype=int)
    return Perturbation(bdm_d2, X)

@pytest.fixture(scope='function')
def perturbation_d2_rec(bdm_d2_rec):
    np.random.seed(1001)
    X = np.random.randint(0, 2, (100, 100), dtype=int)
    return Perturbation(bdm_d2_rec, X)

@pytest.fixture(scope='function')
def perturbation_d1(bdm_d1):
    np.random.seed(999)
    X = np.random.randint(0, 2, (500, ), dtype=int)
    return Perturbation(bdm_d1, X)

@pytest.fixture(scope='function')
def perturbation_d1_b9(bdm_d1_b9):
    np.random.seed(10101)
    X = np.random.randint(0, 9, (500,), dtype=int)
    return Perturbation(bdm_d1_b9, X)


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
class TestPerturbation:

    def _assert_step(self, step, keep_changes, batch, metric, perturbation):
        # pylint: disable=unused-variable,too-many-locals
        # pylint: disable=too-many-statements
        P = perturbation
        X0 = P.X.copy()
        P.batch = batch
        P.metric = metric
        P.X = X0.copy()
        bdm = perturbation.bdm
        steps = list(P.prepare_steps(step))

        step.bind(P)
        main = step
        idx = np.unique(main.idx, axis=0)

        true0 = bdm.cmx(X0, metric=metric)
        assert P._cmx == approx(true0)

        # Batch scenario
        if batch:
            assert len(steps) == 1
            S = steps[0]
            vidx = S.vidx
            vals = S.newvals
            X1 = X0.copy()
            X1[vidx] = vals
            cmx_step = P.make_step(S, keep_changes=keep_changes)
            cmx_data = bdm.cmx(P.X, metric=metric)
            true1 = bdm.cmx(X1, metric=metric)
            assert cmx_step == approx(true1)
            if keep_changes:
                assert (X1 == P.X).all()
                changed = np.argwhere(P.X != X0)
                assert np.array_equal(changed, idx)
                assert cmx_step == approx(cmx_data)
            else:
                assert (P.X == X0).all()
                assert cmx_data == approx(true0)
        # Sequential scenario
        else:
            X1 = X0.copy()
            for i, S in enumerate(steps):
                _idx = S.idx
                vidx = S.vidx
                vals = S.newvals
                X2 = X1.copy()
                X2[vidx] = vals
                cmx_step = P.make_step(S, keep_changes=keep_changes)
                cmx_data = bdm.cmx(P.X, metric=metric)
                true1 = bdm.cmx(X2, metric=metric)
                assert cmx_step == approx(true1)
                if keep_changes:
                    changed = np.argwhere(P.X != X1)
                    assert np.array_equal(changed, _idx)
                    assert (P.X == X2).all()
                    assert cmx_step == approx(cmx_data)
                    assert (P.X != X0).sum() == i + 1
                    X1 = X2
                else:
                    assert cmx_data == approx(true0)
                    assert (P.X == X0).all()
                    assert (X2 != X0).sum() == 1

    @pytest.mark.parametrize('step', [
        PS(lambda x: random_idx(x, n=1)),
        PS(lambda x: random_idx(x, n=5)),
        PS(lambda x: random_idx(x, n=10)),
        PS(lambda x: random_idx(x, n=50)),
        PS(lambda x: random_idx(x, n=100)),
        PS(lambda x: random_idx(x, n=250))
    ])
    @pytest.mark.parametrize('keep_changes', [False, True])
    @pytest.mark.parametrize('batch', [False, True])
    @pytest.mark.parametrize('metric', ['ent', 'bdm'])
    def test_make_step_d2(self, step, keep_changes, batch, metric,
                          perturbation_d2):
        self._assert_step(step, keep_changes, batch, metric, perturbation_d2)

    @pytest.mark.parametrize('step', [
        PS(lambda x: random_idx(x, n=1)),
        PS(lambda x: random_idx(x, n=5)),
        PS(lambda x: random_idx(x, n=10)),
        PS(lambda x: random_idx(x, n=50)),
        PS(lambda x: random_idx(x, n=100)),
        PS(lambda x: random_idx(x, n=250))
    ])
    @pytest.mark.parametrize('keep_changes', [False, True])
    @pytest.mark.parametrize('batch', [False, True])
    @pytest.mark.parametrize('metric', ['ent', 'bdm'])
    def test_make_step_d2_rec(self, step, keep_changes, batch, metric,
                              perturbation_d2_rec):
        self._assert_step(step, keep_changes, batch, metric, perturbation_d2_rec)

    @pytest.mark.parametrize('step', [
        PS(lambda x: random_idx(x, n=1)),
        PS(lambda x: random_idx(x, n=5)),
        PS(lambda x: random_idx(x, n=10)),
        PS(lambda x: random_idx(x, n=50)),
        PS(lambda x: random_idx(x, n=100))
    ])
    @pytest.mark.parametrize('keep_changes', [False, True])
    @pytest.mark.parametrize('batch', [False, True])
    @pytest.mark.parametrize('metric', ['ent', 'bdm'])
    def test_make_step_d1(self, step, keep_changes, batch, metric,
                          perturbation_d1):
        self._assert_step(step, keep_changes, batch, metric, perturbation_d1)

    @pytest.mark.parametrize('step', [
        PS(lambda x: random_idx(x, n=1)),
        PS(lambda x: random_idx(x, n=5)),
        PS(lambda x: random_idx(x, n=10)),
        PS(lambda x: random_idx(x, n=50)),
        PS(lambda x: random_idx(x, n=100))
    ])
    @pytest.mark.parametrize('keep_changes', [False, True])
    @pytest.mark.parametrize('batch', [False, True])
    @pytest.mark.parametrize('metric', ['ent', 'bdm'])
    def test_make_step_d1_b9(self, step, keep_changes, batch, metric,
                             perturbation_d1_b9):
        self._assert_step(step, keep_changes, batch, metric, perturbation_d1_b9)
