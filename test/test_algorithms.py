"""Tests for the `algorithms` module."""
# pylint: disable=W0621
import pytest
import numpy as np
from bdm.algorithms import PerturbationExperiment


@pytest.fixture(scope='function')
def perturbation(bdm_d2):
    data = np.ones((25, 25), dtype=int)
    return PerturbationExperiment(bdm_d2, data)


class TestPerturbationExperiment:

    @pytest.mark.parametrize('i,idx', [
        (624, (24, 24)), (0, (0, 0)),
        (77, (3, 2)), (17, (0, 17))
    ])
    def test_idx_converters(self, perturbation, i, idx):
        out_idx = perturbation.num_to_idx(i)
        out_i = perturbation.idx_to_num(idx)
        assert out_idx == idx
        assert out_i == i
