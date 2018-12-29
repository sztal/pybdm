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
# pylint: disable=W0221
from collections import Counter
from functools import reduce
import numpy as np
from .utils import get_ctm_dataset, get_reduced_idx, get_reduced_shape
from .encoding import string_from_array


class BDMBase:
    """Block decomposition method interface base class.

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

    Stage methods
    -------------
    The *split-apply-combine* approach is implemented through
    stage methods. Stage methods contain the main logic
    of the *Block Decomposition Method* and its different flavours
    depending on boundary conditions etc.
    In the first stage an input dataset is partitioned into parts of shape
    appropriate for the selected CTM dataset (split stage).
    Next, approximate complexity for parts based on the *Coding Theorem Method*
    is looked up in the reference (apply stage) dataset.
    Finally, values for individual parts are aggregated into a final BDM value
    (combine stage). The aggregate stage is divided into two substages
    in order to allow easy parallel and/or distributed computations.
    Hence, there is a method that counts unique slices,
    and a method that aggregates a counter object(s) into a final BDM value.

    The general principle is that specialized *BDM* classes with specific
    boundary conditions should extend the `BDMBase` class
    (or other specialized ``BDM`` class) and implement boundary
    conditions by extending stage methods. All configuration parameters
    for stage methods should be stored within an instance to ensure
    consistent behaviour of advanced algorithms operating on BDM objects
    (such as perturbation algorithms).

    Notes
    -----
    Currently CTM reference datasets are computed only for binary sequences
    of length up to 12 and binary 4-by-4 binary matrices.

    `BDMBase` should not be used for actual computations.
    It is meant to serve as a base class for extending
    and implementing particular boundary conditions.

    Attributes
    ----------
    ndim : int
        Number of dimensions of target dataset objects. Positive integer.
    shift : {0, 1}
        Shift value for the partition algorithm.
        If ``0`` then datasets are sliced into non-overlapping parts.
        If ``1`` then datasets are sliced into overlapping parts.
    shape : tuple
        Shape of slices.
    ctmname : str
        Name of the CTM dataset.
    """
    _ndim_to_shape = {
        1: (12, ),
        2: (4, 4)
    }
    _ndim_to_ctm = {
        1: 'CTM-B2-D12B',
        2: 'CTM-B2-D4x4'
    }
    boundary_condition = 'none'

    def __init__(self, ndim, shift, shape=None, ctmname=None):
        """Initialization method.

        Raises
        ------
        AttributeError
            If parts' `shape` is not equal in each dimension.
        """
        if shift not in (0, 1):
            raise AttributeError("'shift' supports only values of `0` and `1`")
        self.ndim = ndim
        self.shift = shift
        self.shape = shape if shape else self._ndim_to_shape[ndim]
        if not self.shape or any([ x != self.shape[0] for x in self.shape ]):
            raise AttributeError("'shape' has to be equal in each dimension")
        self.ctmname = ctmname if ctmname else self._ndim_to_ctm[ndim]
        self._ctm = get_ctm_dataset(self.ctmname)
        self._sep = '-'

    def partition(self, x, shape=None, indexes=None):
        """Standard partition stage function.

        Parameters
        ----------
        x : array_like
            Dataset of arbitrary dimensionality represented as a *Numpy* array.
        shape : tuple
            Dataset parts' shape.
            Use `shape` defined on the object if ``None``.
        indexes : iterable or None
            Reduced dataset 1D indexes to iterate over.
            Useful when running partition in parallel.
            Iterate over all parts if ``None``.

        Yields
        ------
        array_like
            Dataset parts.

        Raises
        ------
        AttributeError
            If parts' `shape` and dataset's shape have different numbers of axes.

        Acknowledgments
        ---------------
        Special thanks go to Paweł Weroński for the help with the design of
        the non-recursive *partition* algorithm.

        Examples
        --------
        >>> bdm = BDMBase(ndim=2, shift=0, shape=(2, 2))
        >>> [ x for x in bdm.partition(np.ones((3, 3), dtype=int)) ]
        [array([[1, 1],
               [1, 1]]), array([[1],
               [1]]), array([[1, 1]]), array([[1]])]
        """
        if not shape:
            shape = self.shape
        if len(x.shape) != len(self.shape):
            raise AttributeError(
                "dataset and slice shapes does not have the same number of axes"
            )
        r_shape = get_reduced_shape(x, shape, length_only=False)
        n_parts = int(np.multiply.reduce(r_shape))
        indexes = indexes if indexes else range(n_parts)
        width = shape[0]
        slice_shift = self.shift if self.shift > 0 else width
        for i in indexes:
            r_idx = get_reduced_idx(i, r_shape)
            if self.shift <= 0:
                idx = tuple(slice(k*width, k*width + slice_shift) for k in r_idx)
            else:
                idx = tuple(slice(k, k + width) for k in r_idx)
            yield x[idx]

    def lookup(self, parts):
        """Lookup CTM values for parts in a reference dataset.

        Parameters
        ----------
        parts : sequence
            Ordered sequence of dataset parts.
        ctm : dict
            Reference CTM dataset.

        Yields
        ------
        tuple
            2-tuple with string representation of a dataset part and its CTM value.

        Raises
        ------
        KeyError
            If key of an object can not be found in the reference CTM lookup table.

        Examples
        --------
        >>> bdm = BDMBase(ndim=1, shift=0)
        >>> data = np.ones((12, ), dtype=int)
        >>> parts = bdm.partition(data, (12, ))
        >>> [ x for x in bdm.lookup(parts) ]
        [('111111111111', 25.610413747641715)]
        """
        for part in parts:
            key = string_from_array(part)
            try:
                cmx = self._ctm[key]
            except KeyError:
                raise KeyError(f"CTM dataset does not contain object '{key}'")
            yield key, cmx

    def aggregate(self, ctms):
        """Combine CTM of parts into BDM value.

        Parameters
        ----------
        ctms : sequence of 2-tuples
            Ordered 1D sequence of string keys and CTM values.

        Returns
        -------
        float
            BDM value.

        Examples
        --------
        >>> bdm = BDMBase(ndim=1, shift=0)
        >>> data = np.ones((24, ), dtype=int)
        >>> parts = bdm.partition(data, (12, ))
        >>> ctms = bdm.lookup(parts)
        >>> bdm.aggregate(ctms)
        Counter({('111111111111', 25.610413747641715): 2})
        """
        counter = Counter(ctms)
        return counter

    def count_and_lookup(self, x, **kwds):
        """Count parts and assign complexity values.

        Parameters
        ----------
        x : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.
        kwds :
            Optional keyword arguments passed to the partition method.

        Returns
        -------
        collections.Counter
            Lookup table mapping 2-tuples with string keys and CTM values
            to numbers of occurences.

        Examples
        --------
        >>> import numpy as np
        >>> bdm = BDMBase(ndim=1, shift=0)
        >>> bdm.count_and_lookup(np.ones((12, ), dtype=int))
        Counter({('111111111111', 25.610413747641715): 1})
        """
        parts = self.partition(x, **kwds)
        ctms = self.lookup(parts)
        counter = self.aggregate(ctms)
        return counter

    def compute_bdm(self, *counters):
        """Compute BDM approximation.

        Parameters
        ----------
        *counters :
            Counter objects grouping object keys and occurences.

        Returns
        -------
        float
            Approximate algorithmic complexity.

        Examples
        --------
        >>> from collections import Counter
        >>> bdm = BDMBase(ndim=1, shift=0)
        >>> c1 = Counter([('111111111111', 1.95207842085224e-08)])
        >>> c2 = Counter([('111111111111', 1.95207842085224e-08)])
        >>> bdm.compute_bdm(c1, c2)
        1.000000019520784
        """
        counter = reduce(lambda x, y: x+y, counters)
        bdm = 0
        for key, n in counter.items():
            _, ctm = key
            bdm += ctm + np.log2(n)
        return bdm

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
        >>> bdm = BDMBase(ndim=2, shift=0)
        >>> bdm.bdm(np.ones((12, 12), dtype=int))
        25.176631293734488
        """
        counter = self.count_and_lookup(x)
        cmx = self.compute_bdm(counter)
        if raise_if_zero and cmx == 0:
            raise ValueError("Computed BDM is 0, dataset may have incorrect dimensions")
        return cmx

    def compute_entropy(self, *counters):
        """Compute block entropy from counter.

        Parameters
        ----------
        *counters :
            Counter objects grouping object keys and occurences.

        Returns
        -------
        float
            Block entropy in base 2.

        Examples
        --------
        >>> from collections import Counter
        >>> bdm = BDMBase(ndim=1, shift=0)
        >>> c1 = Counter([('111111111111', 1.95207842085224e-08)])
        >>> c2 = Counter([('000000000000', 1.95207842085224e-08)])
        >>> bdm.compute_entropy(c1, c2)
        1.0
        """
        counter = reduce(lambda x, y: x+y, counters)
        ncounts = sum(counter.values())
        ent = 0
        for n in counter.values():
            p = n/ncounts
            ent -= p*np.log2(p)
        return ent

    def entropy(self, x):
        """Block entropy of a dataset.

        Parameters
        ----------
        x : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.

        Returns
        -------
        float
            Block entropy in base 2.

        Examples
        --------
        >>> import numpy as np
        >>> bdm = BDMBase(ndim=2, shift=0)
        >>> bdm.entropy(np.ones((12, 12), dtype=int))
        0.0
        """
        counter = self.count_and_lookup(x)
        return self.compute_entropy(counter)


class BDMIgnore(BDMBase):
    """Block decomposition method with ignore boundary condition.

    See Also
    --------
    For detailed documentation see :py:class:`bdm.bdm.BDMBase`.
    """
    boundary_condition = 'ignore'

    def __init__(self, ndim, shape=None, ctmname=None):
        """Initialization method."""
        super().__init__(ndim, shift=0, shape=shape, ctmname=ctmname)

    def partition(self, x, shape=None, indexes=None):
        """Partition with ignore leftovers boundary condition.

        See Also
        --------
        For detailed documentation see :py:meth:`bdm.bdm.BDMBase.partition`.

        Examples
        --------
        >>> bdm = BDMIgnore(ndim=1, shape=(2, 2))
        >>> [ x for x in bdm.partition(np.ones((3, 3), dtype=int)) ]
        [array([[1, 1],
               [1, 1]])]
        """
        if not shape:
            shape = self.shape
        for part in super().partition(x, shape=shape, indexes=indexes):
            if part.shape == shape:
                yield part


class BDMShrink(BDMBase):
    """Block decomposition method with shrink boundary condition.

    Attributes
    ----------
    min_length : int
        Minimum parts' length. Non-negative.
        In case of multidimensional objects it specifies minimum
        length of any single dimension.

    See Also
    --------
    For detailed documentation see :py:meth:`bdm.bdm.BDMBase.partition`.
    """
    boundary_condition = 'shrink'

    def __init__(self, ndim, min_length, shape=None, ctmname=None):
        """Initialization method."""
        super().__init__(ndim, shift=0, shape=shape, ctmname=ctmname)
        self.min_length = min_length

    def partition(self, x, shape=None, indexes=None, min_length=None):
        """Partition algorithm with a shrinking parts' size.

        See Also
        --------
        For detailed documentation see :py:meth:`bdm.bdm.BDMBase.partition`.

        Examples
        --------
        >>> bdm = BDMShrink(ndim=1, shape=(6, ), min_length=4)
        >>> [ p for p in bdm.partition(np.ones(10, )) ]
        [array([1., 1., 1., 1., 1., 1.]), array([1., 1., 1., 1.])]
        """
        if not shape:
            shape = self.shape
        if min_length is None:
            min_length = self.min_length
        for part in super().partition(x, shape=shape, indexes=indexes):
            if part.shape == shape:
                yield part
            else:
                min_dim_length = min(part.shape)
                if min_dim_length < min_length:
                    continue
                shrinked_shape = tuple(min_dim_length for _ in range(len(shape)))
                yield from super().partition(part, shape=shrinked_shape, indexes=None)
