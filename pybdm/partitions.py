"""Partition algorithm classes.

Partition algorithms are used during the decomposition stage of BDM
(see :doc:`theory` and :py:mod:`pybdm.bdm`), in which datasets are sliced
into blocks of appropriate sizes.

Decomposition can be done in multiple ways that handles boundaries differently.
This is why partition algorithms have to be properly configured,
so it is well-specified what approach exactly is to be used.
"""
# pylint: disable=unused-argument
from .utils import decompose_dataset, iter_part_shapes


class _Partition:
    """Partition algorithm base class.

    Attributes
    ----------
    shape : tuple
        Blocks' shape.
    """
    name = 'none'

    def __init__(self, shape):
        """Initialization method."""
        self.shape = shape

    def __repr__(self):
        cn = self.__class__.__name__
        return "<{}({})>".format(cn, ", ".join(self.params))

    @property
    def params(self):
        return [ "shape={}".format(self.shape) ]

    def decompose(self, X):
        """Decompose a dataset into blocks.

        Parameters
        ----------
        x : array_like
            Dataset of arbitrary dimensionality represented as a *Numpy* array.

        Yields
        ------
        array_like
            Dataset blocks.
        """
        cn = self.__class__.__name__
        raise NotImplementedError("'{}' is not meant for a direct use".format(cn))

    def _iter_shapes(self, X):
        yield from iter_part_shapes(X, shape=self.shape, shift=0)


class PartitionIgnore(_Partition):
    """Partition with the 'ignore' boundary condition.

    Attributes
    ----------
    shape : tuple
        Part shape.

    Notes
    -----
    See :doc:`theory` for a detailed description.
    """
    name = 'ignore'

    def decompose(self, X):
        """Decompose with the 'ignore' boundary.

        .. automethod:: _Partition.decompose
        """
        for part in decompose_dataset(X, shape=self.shape, shift=0):
            if part.shape == self.shape:
                yield part

    def _iter_shapes(self, X):
        for shape in super()._iter_shapes(X):
            if shape == self.shape:
                yield shape

class PartitionCorrelated(PartitionIgnore):
    """Partition with the 'correlated' boundary condition.

    Attributes
    ----------
    shape : tuple
        Part shape.
    shift : int (positive)
        Shift parameter for the sliding window.

    Notes
    -----
    See :doc:`theory` for a detailed description.

    Raises
    ------
    AttributeError
        If `shift` is not positive.
    """
    name = 'correlated'

    def __init__(self, shape, shift=1):
        """Initialization method."""
        super().__init__(shape=shape)
        if shift < 1:
            raise AttributeError("'shift' has to be a positive integer")
        self.shift = shift

    @property
    def params(self):
        return super().params + [ "shift={}".format(self.shift) ]

    def decompose(self, X):
        """Decompose with the 'correlated' boundary.

        .. automethod:: _Partition.decompose
        """
        for part in decompose_dataset(X, shape=self.shape, shift=self.shift):
            if part.shape == self.shape:
                yield part

    def _iter_shapes(self, X):
        shapes = iter_part_shapes(X, shape=self.shape, shift=self.shift)
        for shape in shapes:
            if shape == self.shape:
                yield shape


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

    Notes
    -----
    See :doc:`theory` for a detailed description.
    """
    name = 'recursive'

    def __init__(self, shape, min_length=2):
        """Initialization method."""
        super().__init__(shape=shape)
        self.min_length = min_length

    @property
    def params(self):
        return super().params + [ "min_length={}".format(self.min_length) ]

    def _decompose(self, X, shape):
        for part in decompose_dataset(X, shape=shape, shift=0):
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
