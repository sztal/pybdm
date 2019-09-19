# -*- coding: UTF-8 -*-
"""Block Decomposition Method

`BDM` class provides a top-level interface for configuring an instance
of a block decomposition method as well as running actual computations
approximating algorithmic complexity of given datasets.

Configuration step is necessary for specifying dimensionality of allowed
datasets, encoding of reference CTM data as well as
boundary conditions for block decomposition etc. This is why BDM
is implemented in object-oriented fashion, an instance can be first configured
properly and then it exposes a public method :py:meth:`bdm.BDM.bdm`
for computing approximated complexity via BDM.
"""
# pylint: disable=W0221
import warnings
from math import factorial, log2
from collections import Counter, defaultdict
from functools import reduce
from itertools import cycle, repeat, chain
import numpy as np
from .utils import get_ctm_dataset, slice_dataset, iter_part_shapes
from .encoding import string_from_array, normalize_key
from .exceptions import BDMRuntimeWarning
from .exceptions import CTMDatasetNotFoundError, BDMConfigurationError


class BDMBase:
    """Block decomposition method.

    Block decomposition method is dependent on a reference CTM dataset
    with precomputed algorithmic complexity for small objects of a given
    dimensionality approximated with the *Coding Theorem Method* (CTM).

    Attributes
    ----------
    ndim : int
        Number of dimensions of target dataset objects. Positive integer.
    shift : {0, 1}
        Shift value for the partition algorithm.
        If ``0`` then datasets are sliced into non-overlapping parts.
        If ``1`` then datasets are sliced into overlapping parts.
    shape : tuple
        Shape of slices.  If ``None`` then biggest shape supported
        by the selected CTM dataset is used.
    nsymbols : int
        Number of symbols in the alphabet.
    ctmname : str
        Name of the CTM dataset. If ``None`` then a CTM dataset is selected
        automatically based on `ndim` and `nsymbols`.
    warn_if_missing_ctm : bool
        Should ``BDMRuntimeWarning`` be sent in case there is missing CTM value.
        Some CTM values may be missing for larger alphabets as it is
        computationally infeasible to explore entire parts space.
        Missing CTM values are imputed with mean CTM complexities
        over all parts of a given shape.


    Overview
    --------
    Block decomposition method is implemented using the *split-apply-combine*
    pipeline approach. First a dataset is partitioned into parts with dimensions
    appropriate for a selected data dimensionality and corresponding
    reference lookup table of CTM value. Then CTM values for all parts
    are extracted. Finally CTM values are aggregated to a single
    approximation of complexity for the entire dataset.
    This stepwise approach makes the implementation modular,
    so every step can be customized during the configuration of a `BDM` object
    or by subclassing.

    **Stage methods**

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
    Currently CTM values are computed only for 1D strings of length up to 12
    elements based on alphabets with 2, 4, 5, 6 and 9 symbols as well as
    for symmetric binary matrices of size up to 4-by-4.

    `BDMBase` should not be used for actual computations.
    It is meant to serve as a base class for extending
    and implementing particular boundary conditions.

    Raises
    ------
    AttributeError
        If 'shift' is not ``0`` or ``1``.
    AttributeError
        If parts' `shape` is not equal in each dimension.
    CTMDatasetNotFoundError
        If there is no CTM dataset for a combination of `ndim` and `nsymbols`
        or a given `ctmname`.
    """
    _ndim_to_ctm = {
        # 1D datasets
        (1, 2): 'CTM-B2-D12',
        (1, 4): 'CTM-B4-D12',
        (1, 5): 'CTM-B5-D12',
        (1, 6): 'CTM-B6-D12',
        (1, 9): 'CTM-B9-D12',
        # 2D datasets
        (2, 2): 'CTM-B2-D4x4',
    }
    boundary_condition = 'none'

    def __init__(self, ndim, shift, shape=None, nsymbols=2, ctmname=None,
                 warn_if_missing_ctm=True):
        """Initialization method."""
        if shift not in (0, 1):
            raise BDMConfigurationError("'shift' supports only values of `0` and `1`")
        self.ndim = ndim
        self.shift = shift
        try:
            self.ctmname = ctmname if ctmname else self._ndim_to_ctm[(ndim, nsymbols)]
        except KeyError:
            msg = "no CTM dataset for 'ndim={}' and 'nsymbols={}'".format(
                ndim, nsymbols
            )
            raise CTMDatasetNotFoundError(msg)
        try:
            nsymbols, _shape = self.ctmname.split('-')[-2:]
        except ValueError:
            msg = "incorrect 'ctmname'; it should be in format " + \
                "'name-b<nsymbols>-d<shape>'"
            raise BDMConfigurationError(msg)
        self.nsymbols = int(nsymbols[1:])
        if shape is None:
            self.shape = tuple(int(x) for x in _shape[1:].split('x'))
        elif any([ x != shape[0] for x in shape ]):
            raise BDMConfigurationError("'shape' has to be equal in each dimension")
        else:
            self.shape = shape
        ctm, ctm_missing = get_ctm_dataset(self.ctmname)
        self._ctm = ctm
        self._ctm_missing = ctm_missing
        self.warn_if_missing_ctm = warn_if_missing_ctm

    def partition(self, X, shape=None):
        """Standard partition stage function.

        Parameters
        ----------
        x : array_like
            Dataset of arbitrary dimensionality represented as a *Numpy* array.
        shape : tuple
            Dataset parts' shape.
            Use `shape` defined on the object if ``None``.
            This argument should not be usually used.
            It is meant to be used only in implementations
            of specialized recursive partition algorithms.

        Yields
        ------
        array_like
            Dataset parts.

        Raises
        ------
        AttributeError
            If parts' `shape` and dataset's shape have different numbers of axes.


        **Acknowledgments**

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
        yield from slice_dataset(X, shape=shape, shift=self.shift)

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

        Warns
        -----
        BDMRuntimeWarning
            If ``warn_if_missing_ctm=True`` and there is no precomputed CTM
            value for a part during the lookup stage.

        Examples
        --------
        >>> bdm = BDMBase(ndim=1, shift=0)
        >>> data = np.ones((12, ), dtype=int)
        >>> parts = bdm.partition(data, (12, ))
        >>> [ x for x in bdm.lookup(parts) ] # doctest: +FLOAT_CMP
        [('111111111111', 25.610413747641715)]
        """
        for part in parts:
            sh = part.shape
            key = string_from_array(part)
            key_n = normalize_key(key)
            try:
                cmx = self._ctm[sh][key_n]
            except KeyError:
                cmx = self._ctm_missing[sh]
                if self.warn_if_missing_ctm:
                    msg = "CTM dataset does not contain object '{}' of shape {}".format(
                        key, sh
                    )
                    warnings.warn(msg, BDMRuntimeWarning, stacklevel=2)
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
        >>> bdm.aggregate(ctms) # doctest: +FLOAT_CMP
        Counter({('111111111111', 25.610413747641715): 2})
        """
        counter = Counter(ctms)
        return counter

    def lookup_and_count(self, X):
        """Count parts and assign complexity values.

        Parameters
        ----------
        X : array_like
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
        >>> bdm = BDMBase(ndim=1, shift=0)
        >>> bdm.lookup_and_count(np.ones((12, ), dtype=int)) # doctest: +FLOAT_CMP
        Counter({('111111111111', 25.610413747641715): 1})
        """
        parts = self.partition(X)
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
        >>> bdm.compute_bdm(c1, c2) # doctest: +FLOAT_CMP
        1.000000019520784
        """
        counter = reduce(lambda x, y: x+y, counters)
        bdm = 0
        for key, n in counter.items():
            _, ctm = key
            bdm += ctm + log2(n)
        return bdm

    def bdm(self, X, normalize=False, raise_if_zero=True, check_data=True):
        """Approximate complexity of a dataset.

        Parameters
        ----------
        X : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.
        normalize : bool
            Should BDM be normalized to be in the [0, 1] range.
        raise_if_zero : bool
            Should error be raised if BDM value is zero.
            Zero value indicates that a dataset could have incorrect dimensions.
        check_data : bool
            Should data format be checked.
            May be disabled to gain some speed when calling multiple times.

        Returns
        -------
        float
            Approximate algorithmic complexity.

        Raises
        ------
        TypeError
            If `X` is not an integer array and `check_data=True`.
        ValueError
            If `X` has more than `nsymbols` unique values
            and `check_data=True`.
        ValueError
            If `X` has symbols outside of the ``0`` to `nsymbols-1` range
            and `check_data=True`.
        ValueError
            If computed BDM value is 0 and `raise_if_zero` is ``True``.

        Examples
        --------
        >>> import numpy as np
        >>> bdm = BDMBase(ndim=2, shift=0)
        >>> bdm.bdm(np.ones((12, 12), dtype=int)) # doctest: +FLOAT_CMP
        25.176631293734488
        """
        if check_data:
            self._check_data(X)
        if normalize and self.shift > 0:
            raise NotImplementedError(
                "normalized 'bdm' not implemented for positive 'shift'"
            )
        counter = self.lookup_and_count(X)
        cmx = self.compute_bdm(counter)
        if raise_if_zero and cmx == 0:
            raise ValueError("Computed BDM is 0, dataset may have incorrect dimensions")
        if normalize:
            min_cmx = self._get_min_bdm(X)
            max_cmx = self._get_max_bdm(X)
            cmx = (cmx - min_cmx) / (max_cmx - min_cmx)
        return cmx

    def compute_ent(self, *counters):
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
        >>> bdm.compute_ent(c1, c2) # doctest: +FLOAT_CMP
        1.0
        """
        counter = reduce(lambda x, y: x+y, counters)
        ncounts = sum(counter.values())
        ent = 0
        for n in counter.values():
            p = n/ncounts
            ent -= p*np.log2(p)
        return ent

    def ent(self, X, normalize=False, check_data=True):
        """Block entropy of a dataset.

        Parameters
        ----------
        X : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.
        normalize : bool
            Should entropy be normalized to be in the [0, 1] range.
        check_data : bool
            Should data format be checked.
            May be disabled to gain some speed when calling multiple times.

        Returns
        -------
        float
            Block entropy in base 2.

        Raises
        ------
        TypeError
            If `X` is not an integer array and `check_data=True`.
        ValueError
            If `X` has more than `nsymbols` unique values
            and `check_data=True`.
        ValueError
            If `X` has symbols outside of the ``0`` to `nsymbols-1` range
            and `check_data=True`.

        Examples
        --------
        >>> import numpy as np
        >>> bdm = BDMBase(ndim=2, shift=0)
        >>> bdm.ent(np.ones((12, 12), dtype=int)) # doctest: +FLOAT_CMP
        0.0
        """
        if check_data:
            self._check_data(X)
        if normalize and self.shift > 0:
            raise NotImplementedError(
                "normalized 'ent' not implemented for positive 'shift'"
            )
        counter = self.lookup_and_count(X)
        ent = self.compute_ent(counter)
        if normalize:
            min_ent = self._get_min_ent(X)
            max_ent = self._get_max_ent(X)
            ent = (ent - min_ent) / (max_ent - min_ent)
        return ent

    def _check_data(self, X):
        """Check if data is correctly formatted.

        Symbols have to mapped to integers from ``0`` to `self.nsymbols-1`.
        """
        if not issubclass(X.dtype.type, np.integer):
            raise TypeError("'X' has to be an integer array")
        symbols = np.unique(X)
        if symbols.size > self.nsymbols:
            raise ValueError("'X' has more than {} unique symbols".format(
                self.nsymbols
            ))
        valid_symbols = np.array([ _ for _ in range(self.nsymbols) ])
        bad_symbols = symbols[~np.isin(symbols, valid_symbols)]
        if bad_symbols.size > 0:
            raise ValueError("'X' contains symbols outside of [0, {}]: {}".format(
                str(self.nsymbols-1),
                ", ".join(str(s) for s in bad_symbols)
            ))

    def _cycle_parts(self, shape):
        """Cycle over all possible dataset parts sorted by complexity."""

        def rep(part):
            key, cmx = part
            n = len(set(key))
            k = factorial(self.nsymbols) / factorial(self.nsymbols - n)
            return repeat((key, cmx), int(k))

        parts = chain.from_iterable(map(rep, self._ctm[shape].items()))
        return cycle(enumerate(parts))

    def _get_counter_dct(self, X):
        cycle_dct = {}
        counter_dct = defaultdict(Counter)
        for shape, n in self._iter_shapes(X):
            if shape not in cycle_dct:
                cycle_dct[shape] = self._cycle_parts(shape)
            for _ in range(n):
                idx, kv = next(cycle_dct[shape])
                _, cmx = kv
                counter_dct[shape].update(((idx, cmx),))
        return counter_dct

    def _iter_shapes(self, X):
        shapes = Counter(iter_part_shapes(X, shape=self.shape, shift=self.shift))
        yield from shapes.items()

    def _get_max_bdm(self, X):
        counter_dct = self._get_counter_dct(X)
        bdm = 0
        for dct in counter_dct.values():
            for c, n in dct.items():
                _, cmx = c
                bdm += cmx + log2(n)
        return bdm

    def _get_min_bdm(self, X):
        bdm = 0
        for shape, n in self._iter_shapes(X):
            cmx = next(reversed(self._ctm[shape].values()))
            bdm += cmx + log2(n)
        return bdm

    def _get_max_ent(self, X):
        counter_dct = self._get_counter_dct(X)
        parts_count = Counter()
        for dct in counter_dct.values():
            parts_count.update(idx for idx, _ in dct)
        return self.compute_ent(parts_count)

    def _get_min_ent(self, X):
        shapes = Counter(shape for shape, _ in self._iter_shapes(X))
        ncounts = sum(shapes.values())
        ent = 0
        for n in shapes.values():
            p = n/ncounts
            ent -= p*log2(p)
        return ent


class BDMIgnore(BDMBase):
    """Block decomposition method with ignore boundary condition.

    See Also
    --------
    BDMBase : base BDM class
    """
    boundary_condition = 'ignore'

    def __init__(self, ndim, shape=None, nsymbols=2, ctmname=None,
                 warn_if_missing_ctm=True):
        """Initialization method."""
        super().__init__(ndim, shift=0, shape=shape, ctmname=ctmname,
                         nsymbols=nsymbols, warn_if_missing_ctm=warn_if_missing_ctm)

    def partition(self, X, shape=None):
        """Partition with ignore leftovers boundary condition.

        .. automethod:: BDMBase.partition

        Examples
        --------
        >>> bdm = BDMIgnore(ndim=1, shape=(2, 2))
        >>> [ x for x in bdm.partition(np.ones((3, 3), dtype=int)) ]
        [array([[1, 1],
               [1, 1]])]
        """
        if not shape:
            shape = self.shape
        for part in super().partition(X, shape=shape):
            if part.shape == shape:
                yield part

    def _iter_shapes(self, X):
        for shape, n in super()._iter_shapes(X):
            if shape == self.shape:
                yield shape, n


class BDMRecursive(BDMBase):
    """Block decomposition method with recursive boundary condition.

    Attributes
    ----------
    min_length : int
        Minimum parts' length. Non-negative.
        In case of multidimensional objects it specifies minimum
        length of any single dimension.

    See Also
    --------
    BDMBase : base BDM class
    """
    boundary_condition = 'recursive'

    def __init__(self, ndim, min_length, shape=None, nsymbols=2, ctmname=None,
                 warn_if_missing_ctm=True):
        """Initialization method."""
        super().__init__(ndim, shift=0, shape=shape, ctmname=ctmname,
                         nsymbols=nsymbols, warn_if_missing_ctm=warn_if_missing_ctm)
        self.min_length = min_length

    def partition(self, X, shape=None):
        """Partition algorithm with a shrinking parts' size.

        .. automethod:: BDMBase.partition

        Examples
        --------
        >>> bdm = BDMRecursive(ndim=1, shape=(6, ), min_length=4)
        >>> [ p for p in bdm.partition(np.ones(10, )) ]
        [array([1., 1., 1., 1., 1., 1.]), array([1., 1., 1., 1.])]
        """
        if not shape:
            shape = self.shape
        for part in super().partition(X, shape=shape):
            if part.shape == shape:
                yield part
            else:
                min_dim_length = min(part.shape)
                if min_dim_length < self.min_length:
                    continue
                shrinked_shape = tuple(min_dim_length for _ in range(len(shape)))
                yield from super().partition(part, shape=shrinked_shape)
