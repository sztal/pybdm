"""Unit tests for BDM stage functions."""
# pylint: disable=E1101,W0621,W0212
from collections import Counter
import pytest
import numpy as np
from bdm import BDM
from bdm.stages import partition, lookup, aggregate, compute_bdm
from bdm.stages import partition_ignore, partition_shrink
from bdm.encoding import decode_string as dec
from bdm.utils import get_ctm_dataset


@pytest.fixture(scope='session')
def ctmbin2d():
    """CTM reference dataset for 2D binary matrices."""
    return get_ctm_dataset('CTM-B2-D4x4')

@pytest.mark.parametrize('x,shape,shift,reduced_idx,expected', [
    (np.ones((2, 2)), (2, 2), 0, None, [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (4, 4), 0, None, [
        np.ones((4, 4)), np.ones((4, 1)), np.ones((1, 4)), np.ones((1, 1))
    ]),
    (np.array([[1,2,3],[4,5,6],[7,8,9]]), (2, 2), 1, None, [
        np.array([[1,2],[4,5]]), np.array([[2,3],[5,6]]),
        np.array([[4,5],[7,8]]), np.array([[5,6],[8,9]])
    ]),
    (np.array([[1,2,3],[4,5,6],[7,8,9]]), (2, 2), 1, (1, 2), [
        np.array([[2,3],[5,6]]), np.array([[4,5],[7,8]])
    ])
])
def test_partition(x, shape, shift, reduced_idx, expected):
    output = [ p for p in partition(x, shape, shift, reduced_idx) ]
    assert len(output) == len(expected)
    assert all([ np.array_equal(o, e) for o, e in zip(output, expected) ])

@pytest.mark.parametrize('x,shape,expected', [
    (np.ones((2, 2)), (2, 2), [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (4, 4), [ np.ones((4, 4)) ]),
    (np.ones((3, 3)), (2, 2), [ np.ones((2, 2)) ])
])
def test_partition_ignore(x, shape, expected):
    output = [ p for p in partition_ignore(x, shape) ]
    assert len(output) == len(expected)
    assert all([ np.array_equal(o, e) for o, e in zip(output, expected) ])

@pytest.mark.parametrize('x,shape,min_length,expected', [
    (np.ones((6, 6)), (4, 4), 2, [
        np.ones((4, 4)), np.ones((2, 2)), np.ones((2, 2)),
        np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))
    ]),
    (np.ones((6, 6)), (4, 4), 3, [ np.ones((4, 4)) ]),
    (np.ones((20, )), (12, ), 6, [ np.ones((12, )), np.ones((8, ))])
])
def test_partition_shrink(x, shape, min_length, expected):
    output = [ p for p in partition_shrink(x, shape, min_length=min_length) ]
    assert len(output) == len(expected)
    assert all([ np.array_equal(o, e) for o, e in zip(output, expected) ])

@pytest.mark.parametrize('parts,expected', [
    ([ np.ones((4, 4)).astype(int) ], [ (dec(65535, (4, 4)), 22.006706292292176) ]),
])
def test_lookup(parts, ctmbin2d, expected):
    output = [ x for x in lookup(parts, ctmbin2d) ]
    assert output == expected

@pytest.mark.parametrize('ctms,expected', [
    ([ (65535, 22.0067) for _ in range(4) ], Counter([
        (65535, 22.0067), (65535, 22.0067), (65535, 22.0067), (65535, 22.0067)
    ]))
])
def test_aggregate(ctms, expected):
    output = aggregate(ctms)
    assert output == expected


class TestPipeline:

    @classmethod
    def setup_class(cls):
        cls.bdm1 = BDM(ndim=1)
        cls.bdm2 = BDM(ndim=2)

    @pytest.mark.parametrize('data,shape,exp_ctms,exp_bdm', [
        (np.ones((30, ), dtype=int), (12, ), [
            ('111111111111', 1.95207842085224e-08),
            ('111111111111', 1.95207842085224e-08),
        ], 1.000000019520784)
    ])
    def test_pipeline_1d_ignore(self, data, shape, exp_ctms, exp_bdm):
        parts = [ p for p in partition_ignore(data, shape) ]
        ctms = [ x for x in lookup(parts, self.bdm1._ctm) ]
        counter = aggregate(ctms)
        bdm = compute_bdm(counter)
        assert ctms == exp_ctms
        assert bdm == exp_bdm
