"""Partition algorithm classes."""
# pylint: disable=unused-argument
from bdm.utils import slice_dataset


class _Partition:
    """Partition algorithm base class."""
    boundary = 'none'

    def __init__(self, shape):
        """Initialization method."""
        self.shape = shape

    def partition(self, X):
        """Partition method.

        Parameters
        ----------
        x : array_like
            Dataset of arbitrary dimensionality represented as a *Numpy* array.
        """
        cn = self.__class__.__name__
        raise NotImplementedError("'{}' is not meant for a direct use".format(cn))


class PartitionIgnore(_Partition):
    """Partition with the 'ignore' boundary condition.

    Attributes
    ----------
    shape : tuple
        Part shape.
    """
    boundary = 'ignore'

    def partition(self, X):
        """Partition 'ignore' method.

        .. automethod:: _Partition.partition
        """
        for part in slice_dataset(X, shape=self.shape, shift=0):
            if part.shape == self.shape:
                yield part

class PartitionCorrelated(_Partition):
    """Partition with the 'correlated' boundary condition.

    Attributes
    ----------
    shape : tuple
        Part shape.
    shift : int (positive)
        Shift parameter for the sliding window.

    Raises
    ------
    AttributeError
        If `shift` is not positive.
    """
    boundary = 'correlated'

    def __init__(self, shape, shift=1):
        """Initialization method."""
        super().__init__(shape=shape)
        if shift < 1:
            raise AttributeError("'shift' has to be a positive integer")
        self.shift = shift

    def partition(self, X):
        """Partition 'correlated' method.

        .. automethod:: _Partition.partition
        """
        for part in slice_dataset(X, shape=self.shape, shift=self.shift):
            if part.shape == self.shape:
                yield part

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
    """
    boundary = 'recursive'

    def __init__(self, shape, min_length=2):
        """Initialization method."""
        super().__init__(shape=shape)
        self.min_length = min_length

    def _partition(self, X, shape):
        for part in slice_dataset(X, shape=shape, shift=0):
            if part.shape == shape:
                yield part
            else:
                min_dim_length = min(part.shape)
                if min_dim_length < self.min_length:
                    continue
                shrinked_shape = tuple(min_dim_length for _ in range(len(shape)))
                yield from self._partition(part, shrinked_shape)

    def partition(self, X):
        """Partition 'recursive' method.

        .. automethod:: _Partition.partition
        """
        yield from self._partition(X, shape=self.shape)
