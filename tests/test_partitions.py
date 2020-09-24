"""Unit tests for BDM partition algorithms."""
import pytest
import numpy as np
from pybdm.partitions import PartitionIgnore, PartitionRecursive
from pybdm.partitions import get_partition


def _test_decompose(partition, X, expected):
    output = list(partition.decompose(X))
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
    _test_decompose(partition, X, expected)

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
    _test_decompose(partition, X, expected)

@pytest.mark.parametrize('name,alias,expected', [
    ('PartitionIgnore', 'ignore', PartitionIgnore),
    ('PartitionRecursive', 'recursive', PartitionRecursive)
])
def test_get_partition(name, alias, expected):
    out1 = get_partition(name)
    out2 = get_partition(alias)
    assert out1 is out2 is expected
