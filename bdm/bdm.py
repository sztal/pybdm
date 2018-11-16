"""Block Decomposition Method

`BDM` class provides a top-level interface for configuring an instance
of a block decomposition method as well as running actual computations
approximating algorithmic complexity of given datasets.

Configuration step is necessary for specifying dimensionality of allowed
datasets as well as boundary conditions for block decomposition etc.
"""
from .boundary import leftover


class BDM:
    """Block decomposition method depends on the type data
    (binary sequences or matrices) as well as boundary conditions.

    Block decomposition method is implemented using the *split-apply-combine*
    pipeline approach. First a dataset is partitioned into parts with dimensions
    appropriate for a selected data dimensionality and corresponding
    reference lookup table of CTM value. Then CTM values for all parts
    are extracted. Finally CTM values are aggregated to a single
    approximation of complexity for the entire dataset.
    This stepwise approach makes the implementation modular,
    so every step can be customized during the configuration of a `BDM` object
    or by subclassing.

    Attributes
    ----------
    dtype : {'sequence', 'matrix'}
        Expected type of input datasets.
    boundary : callable
        Callable object that implements a proper boundary condition.
    split : callable or None
        Splitting function. Use the default method if ``None``.
    apply : callable or None
        Apply function. Standard *CTM* value lookup if ``None``.
    combine : callable or None
        Combine function. Use the default method if ``None``.
    """

    def __init__(self, dtype, boundary=leftover,
                 partition=None, apply=None, combine=None):
        """Initialization method."""
        self.dtype = dtype
        self.boundary = boundary
        self._partition = partition
        self._apply = apply
        self._combine = combine

    def split(self, x):
        """Default partition method."""
        pass

    def apply(self, x):
        """Default apply method."""
        pass

    def combine(self, x):
        """Default combine method."""
        pass

    def complexity(self, x):
        """Approximate complexity of a dataset.

        Parameters
        ----------
        x : (N, k) array_like
            Dataset representax as a :py:class:`numpy.ndarray`.
        """
        parts = self.split(x)
        parts = self.apply(parts)
        cmx = self.combine(parts)
        return cmx
