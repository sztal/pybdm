"""Unit tests for BDM stage functions."""
# pylint: disable=E1101,W0621,W0212
from collections import Counter
import pytest
from pytest import approx
import numpy as np
from bdm.base import BDMBase, BDMIgnore, BDMRecursive
from bdm.encoding import decode_string as dec


@pytest.fixture(scope='session')
def bdm_d2_base():
    return BDMBase(ndim=2, shift=0)

@pytest.mark.parametrize('x,shift,shape,reduced_idx,expected', [
    (np.ones((2, 2)), 0, (2, 2), None, [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), 0, (4, 4), None, [
        np.ones((4, 4)), np.ones((4, 1)), np.ones((1, 4)), np.ones((1, 1))
    ]),
    (np.array([[1,2,3], [4,5,6], [7,8,9]]), 1, (2,2), None, [
        np.array([[1,2],[4,5]]), np.array([[2,3],[5,6]]),
        np.array([[4,5],[7,8]]), np.array([[5,6],[8,9]])
    ])
])
def test_partition(x, shift, shape, reduced_idx, expected):
    bdm = BDMBase(ndim=2, shift=shift, shape=shape)
    output = [ p for p in bdm.partition(x, shape, reduced_idx) ]
    assert len(output) == len(expected)
    assert all([ np.array_equal(o, e) for o, e in zip(output, expected) ])

@pytest.mark.parametrize('x,shape,expected', [
    (np.ones((2, 2)), (2, 2), [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (4, 4), [ np.ones((4, 4)) ]),
    (np.ones((3, 3)), (2, 2), [ np.ones((2, 2)) ])
])
def test_partition_ignore(x, shape, expected):
    bdm = BDMIgnore(ndim=2, shape=shape)
    output = [ p for p in bdm.partition(x, shape) ]
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
    bdm = BDMRecursive(ndim=len(shape), shape=shape, min_length=min_length)
    output = [ p for p in bdm.partition(x, shape=shape, min_length=min_length) ]
    assert len(output) == len(expected)
    assert all([ np.array_equal(o, e) for o, e in zip(output, expected) ])

@pytest.mark.parametrize('parts,expected', [
    ( [ np.ones((4, 4)).astype(int) ],
      [ (dec(65535, (4, 4)), 22.006706292292176) ] ),
])
def test_lookup(parts, bdm_d2_base, expected):
    output = [ x for x in bdm_d2_base.lookup(parts) ]
    for o, e in zip(output, expected):
        assert o[0] == e[0]
        assert o[1] == approx(e[1])

@pytest.mark.parametrize('ctms,expected', [
    ([ (65535, 22.0067) for _ in range(4) ], Counter([
        (65535, 22.0067), (65535, 22.0067), (65535, 22.0067), (65535, 22.0067)
    ]))
])
def test_aggregate(ctms, bdm_d2_base, expected):
    output = bdm_d2_base.aggregate(ctms)
    assert output == expected
