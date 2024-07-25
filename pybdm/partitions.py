"""Partition algorithm classes.

Partition algorithms are used during the decomposition stage of BDM
(see :doc:`theory` and :py:mod:`pybdm.bdm`), in which datasets are sliced
into blocks of appropriate sizes.

Decomposition can be done in multiple ways that handles boundaries differently.
This is why partition algorithms have to be properly configured,
so it is well-specified what approach exactly is to be used.
"""
# pylint: disable=unused-argument
import numpy as np
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
        return f"<{cn}({', '.join(self.params)})>"

    @property
    def params(self):
        return [ "shape={self.shape}" ]

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
        raise NotImplementedError(f"'{cn}' is not meant for a direct use")

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
        return super().params + [ f"shift={self.shift}" ]

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
        return super().params + [ f"min_length={self.min_length}" ]

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

class PartitionPeriodic(PartitionIgnore):
    """Partition with the 'periodic' boundary condition.
    Attributes
    ----------
    shape : tuple
        Part shape.
    Notes
    -----
    See :doc:`theory` for a detailed description.
    """

    name = 'periodic'

    def _extend_2d_dataset(self, X):

        row_multiplier = 0
        col_multipler = 0

        dataset_row_size = X.shape[0]
        dataset_col_size = X.shape[1]

        block_row_size = self.shape[0]
        block_col_size = self.shape[1]

        row_idx = block_row_size - (dataset_row_size % block_row_size)
        row_idx = row_idx % block_row_size

        col_idx = block_col_size - (dataset_col_size % block_col_size)
        col_idx = col_idx % block_col_size

        if dataset_row_size < row_idx:
            row_multiplier = row_idx // dataset_row_size
            row_idx = dataset_row_size

        if dataset_col_size < col_idx:
            col_multipler = col_idx // dataset_col_size
            col_idx = col_idx % dataset_col_size

        periodic_cols = np.hstack((
            np.tile(X, col_multipler),
            X[:,:col_idx]
        ))
        periodic_rows = np.hstack((
            X[:row_idx, :],
            np.tile(X[:row_idx, :], row_multiplier),
            X[:row_idx, :col_idx]
        ))

        extended_dataset = np.hstack((X, periodic_cols))
        extended_dataset = np.vstack((extended_dataset, periodic_rows))

        return extended_dataset

    def _extend_1d_dataset(self, X):

        multiplier = 0
        dataset_size = X.shape[0]
        block_size = self.shape[0]

        row_idx = block_size - (dataset_size % block_size)
        row_idx = row_idx % block_size

        if dataset_size < row_idx:
            multiplier = row_idx // dataset_size
            row_idx = row_idx % dataset_size

        extended_dataset = np.hstack((
            X, np.tile(X, multiplier), X[:row_idx]
        ))

        return extended_dataset

    def decompose(self, X):
        """Decompose with the 'periodic' boundary but using an extended matrix.
        .. automethod:: _Partition.decompose
        """

        if X.ndim == 1:
            extended_X = self._extend_1d_dataset(X)
        else:
            extended_X = self._extend_2d_dataset(X)

        for part in decompose_dataset(extended_X, shape=self.shape, shift=0):
            if part.shape == self.shape:
                yield part
