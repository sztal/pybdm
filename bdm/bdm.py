"""Block Decomposition Method

`BDM` class provides a top-level interface for configuring an instance
of a block decomposition method as well as running actual computations
approximating algorithmic complexity of given datasets.

Configuration step is necessary for specifying dimensionality of allowed
datasets, encoding of reference CTM data as well as
boundary conditions for block decomposition etc. This is why BDM
is implemented in object-oriented fashion, an instance can be first configured
properly and then it exposes a public method :py:meth:`bdm.BDM.complexity`
for computing approximated complexity via BDM.
"""
from .stages import partition_ignore, lookup, aggregate, compute_bdm
from .utils import get_ctm_dataset


_ndim_to_shape = {
    1: (12, ),
    2: (4, 4)
}
_ndim_to_ctm = {
    1: 'CTM-B2-D12',
    2: 'CTM-B2-D4x4'
}


class BDM:
    """Block decomposition method interface.

    Block decomposition method is dependent on a reference CTM dataset
    with precomputed algorithmic complexity for small objects of a given
    dimensionality approximated with the *Coding Theorem Method* (CTM).

    Block decomposition method is implemented using the *split-apply-combine*
    pipeline approach. First a dataset is partitioned into parts with dimensions
    appropriate for a selected data dimensionality and corresponding
    reference lookup table of CTM value. Then CTM values for all parts
    are extracted. Finally CTM values are aggregated to a single
    approximation of complexity for the entire dataset.
    This stepwise approach makes the implementation modular,
    so every step can be customized during the configuration of a `BDM` object
    or by subclassing.

    Notes
    -----
    Currently CTM reference datasets are computed only for binary sequences
    of length up to 12 and binary 4-by-4 binary matrices.

    Attributes
    ----------
    ndim : int
        Number of dimensions of target dataset objects. Positive integer.
    ctm_width : int or None
        Width of the sliding window and CTM records.
        Set automatically based on `ndim` if ``None``.
        It is specified with a single integer, since multidimensional
        objects have to have all sides of equal lengths.
    ctm_dname : str or None
        Name of a reference CTM dataset.
        Set automatically based on `ndim` if ``None``.
        For now it is mean only for inspection purposes
        (this attribute should not be set and changed).
    partition_func : callable
        Partition stage method.
        In this stage the input dataset is partitioned into parts compatible
        with the selected reference CTM dataset and boundary conditions are applied.
        It has to return an **iterable** object yielding dataset parts.
        In most cases partition functions should wrap around
        :py:func:`bdm.stages.partition`.
        In all cases a partition functions has to consume only 2 parameters,
        dataset and a tuple specifying parts' shape using the *Numpy* array convention.
        The second argument is supplied automatically in :py:meth:`bdm.BDM.complexity`
        based on the object configuration.
    lookup_func : callable
        Lookup stage method.
        In this stage dataset parts are converted to string keys and their
        CTM values are looked up.
        The functions has to return an **iterable** yielding 2-tuples
        with string keys and CTM values.
        In most cases there should be no need for redefining
        the standard :py:func:`bdm.stages.lookup` function.
    aggregate_func : callable
        Aggregate stage method.
        In this stage results for parts are properly aggregated into a final
        BDM values. The standard function is :py:func:`bdm.stages.aggregate`.
    """
    def __init__(self, ndim, ctm_width=None, ctm_dname=None,
                 partition_func=partition_ignore,
                 lookup_func=lookup, aggregate_func=aggregate):
        """Initialization method."""
        self.ndim = ndim
        self.ctm_shape = _ndim_to_shape[ndim] if ctm_width is None \
            else tuple([ ctm_width for _ in range(ndim) ])
        self.ctm_dname = _ndim_to_ctm[ndim] if ctm_dname is None else ctm_dname
        self._ctm = get_ctm_dataset(self.ctm_dname)
        self.partition = partition_func
        self.lookup = lookup_func
        self.aggregate = aggregate_func

    def count_and_lookup(self, x):
        """Count parts and assign complexity values.

        Parameters
        ----------
        x : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.

        Returns
        -------
        collections.Counter
            Lookup table mapping 2-tuples with string keys and CTM values
            to numbers of occurences.

        Examples
        --------
        >>> import numpy as np
        >>> bdm = BDM(ndim=1)
        >>> bdm.count_and_lookup(np.ones((12, ), dtype=int))
        Counter({('111111111111', 1.95207842085224e-08): 1})
        """
        parts = self.partition(x, self.ctm_shape)
        ctms = self.lookup(parts, self._ctm)
        counter = self.aggregate(ctms)
        return counter

    def bdm(self, x, raise_if_zero=True):
        """Approximate complexity of a dataset.

        Parameters
        ----------
        x : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.
        raise_if_zero: bool
            Should error be raised if BDM value is zero.
            Zero value indicates that a dataset could have incorrect dimensions.

        Returns
        -------
        float
            Approximate algorithmic complexity.

        Raises
        ------
        ValueError
            If computed BDM value is 0 and `raise_if_zero` is ``True``.

        Examples
        --------
        >>> import numpy as np
        >>> bdm = BDM(ndim=2)
        >>> bdm.bdm(np.ones((12, 12), dtype=int))
        25.176631293734488
        """
        counter = self.count_and_lookup(x)
        cmx = compute_bdm(counter)
        if raise_if_zero and cmx == 0:
            raise ValueError("Computed BDM is 0, dataset may have incorrect dimensions")
        return cmx
