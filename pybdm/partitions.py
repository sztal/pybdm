"""Partition algorithm classes.

Partition algorithms are used during the decomposition stage of BDM
(see :doc:`theory` and :py:mod:`pybdm.bdm`), in which datasets are sliced
into blocks of appropriate sizes.

Decomposition can be done in multiple ways that handles boundaries differently.
This is why partition algorithms have to be properly configured,
so it is well-specified what approach exactly is to be used.

Currently only two partition algorithms ('ignore' and 'recursive')
are implemented. They are most natural and allow straightforward
definition of normalized bdm (see :doc:`theory`).
"""
# pylint: disable=unused-argument
from collections import Counter
from importlib import import_module
from .decompose import block_decompose


__all__ = [
    'PartitionIgnore',
    'PartitionRecursive'
]


def get_partition(name_or_alias):
    """Get partition class by name or alias."""
    mod = import_module(__name__)
    for name in __all__:
        part = getattr(mod, name)
        if name_or_alias in (name, part.alias):
            return part
    raise NameError("unknown partition '{}'".format(name_or_alias))


class _Partition:
    """Partition algorithm base class.

    Attributes
    ----------
    shape : tuple
        Blocks' shape.
    """
    alias = 'base'

    def __init__(self, shape):
        """Initialization method."""
        self.shape = shape

    def __repr__(self):
        return "<{cn}({p})>".format(
            cn=self.__class__.__name__,
            p=", ".join(str(k)+"="+str(v) for k, v in self._get_params().items())
        )

    def _get_params(self):
        return { 'shape': self.shape }

    def decompose(self, X):
        """Decompose a dataset into blocks.

        Parameters
        ----------
        X : array_like
            Dataset of arbitrary dimensionality represented as a *Numpy* array.

        Yields
        ------
        array_like
            Dataset blocks.
        """
        yield from block_decompose(X, shape=self.shape)

    def block_census(self, X):
        """Calculate block shape census of a dataset.

        Parameters
        ----------
        X : array_like
            Dataset of arbitrary dimensionality represented as a *Numpy* array.
        """
        return Counter(map(lambda x: x.shape, self.decompose(X)))


class PartitionIgnore(_Partition):
    """Partition with the 'ignore' boundary condition.

    Attributes
    ----------
    shape : tuple
        Part shape.
    alias : str
        Equal to ``'ignore'``.

    Notes
    -----
    See :doc:`theory` for a detailed description.
    """
    alias = 'ignore'

    def decompose(self, X):
        """Decompose with the 'ignore' boundary.

        .. automethod:: _Partition.decompose
        """
        for block in super().decompose(X):
            if block.shape == self.shape:
                yield block


class PartitionRecursive(_Partition):
    """Partition with the 'recursive' boundary condition.

    Attributes
    ----------
    shape : tuple
        Part shape.
    min_length : int
        Minimum parts' length. Non-negative.
        In case of multidimensional objects it specifies minimum
        length of any single dimension.
    alias : str
        Equal to ``'recursive'``.

    Notes
    -----
    See :doc:`theory` for a detailed description.
    """
    alias = 'recursive'

    def __init__(self, shape, min_length=2):
        """Initialization method."""
        super().__init__(shape=shape)
        self.min_length = min_length

    def _get_params(self):
        return { **super()._get_params(), 'min_length': self.min_length }

    def _decompose(self, X, shape):
        for part in block_decompose(X, shape=shape):
            if part.shape == shape:
                yield part
            else:
                min_dim_length = min(part.shape)
                if min_dim_length < self.min_length:
                    continue
                shrinked_shape = tuple(min_dim_length for _ in range(len(shape)))
                yield from self._decompose(part, shrinked_shape)

    def decompose(self, X):
        """Decompose with the 'recursive' boundary.

        .. automethod:: _Partition.decompose
        """
        yield from self._decompose(X, shape=self.shape)
