"""Unit tests for BDM partition algorithms."""
import pytest
import numpy as np
from bdm.partitions import PartitionIgnore, PartitionCorrelated, PartitionRecursive


def _test_partition(partition, X, expected):
    output = [ p for p in partition.partition(X) ]
    assert len(output) == len(expected)
    assert all(np.array_equal(o, e) for o, e in zip(output, expected))


@pytest.mark.parametrize('X,shape,expected', [
    (np.ones((2, 2)), (2, 2), [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (2, 2), [
        np.ones((2, 2)), np.ones((2, 2)),
        np.ones((2, 2)), np.ones((2, 2))
    ]),
    (np.ones((12,)), (6,), [ np.ones((6,)), np.ones((6,)) ])
])
def test_partition_ignore(X, shape, expected):
    partition = PartitionIgnore(shape=shape)
    _test_partition(partition, X, expected)

@pytest.mark.parametrize('X,shape,shift,expected', [
    (np.ones((2, 2)), (2, 2), 1, [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (3, 3), 1, [ np.ones((3, 3)) for _ in range(9) ]),
    (np.ones((5, 5)), (3, 3), 2, [ np.ones((3, 3)) for _ in range(4) ])
])
def test_partition_correlated(X, shape, shift, expected):
    partition = PartitionCorrelated(shape=shape, shift=shift)
    _test_partition(partition, X, expected)

@pytest.mark.parametrize('X,shape,min_length,expected', [
    (np.ones((2, 2)), (2, 2), 2, [ np.ones((2, 2)) ]),
    (np.ones((5, 5)), (2, 2), 2, [ np.ones((2, 2)) for _ in range(4) ]),
    (np.ones((5, 5)), (3, 3), 2, [
        np.ones((3, 3)), np.ones((2, 2)),
        np.ones((2, 2)), np.ones((2, 2))
    ])
])
def test_partition_recursive(X, shape, min_length, expected):
    partition = PartitionRecursive(shape=shape, min_length=min_length)
    _test_partition(partition, X, expected)
