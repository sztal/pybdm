# -*- coding: UTF-8 -*-
"""Block Decomposition Method

`BDM` class provides a top-level interface for configuring an instance
of a block decomposition method as well as running actual computations
approximating algorithmic complexity of given datasets.

Configuration step is necessary for specifying dimensionality of allowed
datasets, reference CTM data as well as
boundary conditions for block decomposition etc. This is why BDM
is implemented in an object-oriented fashion, so an instance can be first
configured properly and then it exposes a public method :py:meth:`BDM.bdm`
for computing approximated complexity via BDM.
"""
from math import log2, prod
from collections import defaultdict, Counter
from functools import reduce
import numpy as np
from .options import Options
from .partitions import get_partition
from .ctm import CTMStore, INT_DTYPE
from .encoding import encode_sequences, normalize_sequences
from .utils import chunked_buckets
from .blocks import BlockCounter


class BDM:
    """Block decomposition method.

    Attributes
    ----------
    ndim : int
        Number of dimensions of target dataset objects. Positive integer.
    nsymbols : int
        Number of symbols in the alphabet.
    nstates : int
        nstates : int, positive
        Number of states of explored Turing machines.
        The more states the better approximation.
    shape : tuple of int
        Block shape. Has to be equal in all dimensions.
    partition : pybdm.partitions.Partition
        Partition algorithm class object.
        The class is called with the `shape` attribute determined
        automatically if not passed and other attributes passed via ``**kwds``.
    options : pybdm.options.Options
        Options table. See :py:mod:`pybdm.options` for details.
        Instance-level options defined through this attribute
        override global options.

    Notes
    -----
    Block decomposition method in **PyBDM** is implemented in an object-oriented
    fashion. This design choice was dictated by the fact that BDM can not be
    applied willy-nilly to any dataset, but has to be configured for a particular
    type of data (e.g. binary matrices). Hence, it is more convenient to first
    configure and instatiate a particular instance of BDM and the apply
    it freely to data instead of passing a lot of arguments at every call.

    BDM has also natural structure corresponding to the so-called
    *split-apply-combine* strategy in data analysis.
    First, a large dataset it decomposed into smaller blocks for which
    precomputed CTM values can be efficiently looked up.
    Then CTM values for slices are aggregated in a theory-informed way
    into a global approximation of complexity of the full dataset.
    Thus, BDM computations naturally decomposes into four stages:

    #. **Partition (decomposition) stage.** First a dataset is decomposed
       into blocks. This is done by the :py:meth:`decompose` method.
       The method itself is dependent on the `partition` attribute which points
       to a :py:mod:`pybdm.partitions.Partition` object,
       which implements and configures a particular variant
       of the decomposition algorithm. Detailed description of the available
       algorithms can be found in :doc:`theory`.
    #. **Lookup stage.** At this stage CTM values for blocks are looked up.
       This is when the CTM reference dataset is used.
       It is implemented in the :py:meth`lookup` method.
    #. **Count stage.** Unique dataset blocks are counted and arranged in
       an efficient data structure together with their CTM values.
    #. **Aggregate stage.** Final BDM value is computed based on block
       counter data structure.

    See also
    --------
    pybdm.ctm : available CTM datasets
    pybdm.partitions : available partition and boundary condition classes
    pybdm.options : general `pybdm` options.
    """
    def __init__(self, ndim, nsymbols=2, nstates=None, shape=None,
                 partition='ignore', options=None, **kwds):
        """Initialization method.

        Parameters
        ----------
        **kwds :
            Other keyword arguments passed to a partition algorithm class.

        Raises
        ------
        AttributeError
            If block `shape` is not equal in each dimension.
        LookupError
            If there is no CTM dataset for a combination of `ndim`
            and `nsymbols`.
        """
        ctm = CTMStore.from_params(
            ndim=ndim,
            nsymbols=nsymbols,
            nstates=nstates
        )
        if shape is None:
            shape = next(iter(sorted(ctm.data, key=sum, reverse=True)))
        if any(x != shape[0] for x in shape):
            raise AttributeError("'shape' has to be equal in each dimension")

        if isinstance(partition, str):
            partition = get_partition(partition)
        if callable(partition):
            partition = partition(shape=shape, **kwds)

        self.ctm = ctm
        self.partition = partition
        self.options = Options(**(options or {}))

    def __repr__(self):
        return "<{cn}(ndim={d}, nsymbols={b}, nstates={s}) with {p}>".format(
            cn=self.__class__.__name__,
            d=self.ndim,
            b=self.nsymbols,
            s=self.nstates,
            p=str(self.partition)[1:-1]
        )

    # Properties --------------------------------------------------------------

    @property
    def ndim(self):
        return self.ctm.ndim

    @property
    def nsymbols(self):
        return self.ctm.nsymbols

    @property
    def nstates(self):
        return self.ctm.nstates

    @property
    def shape(self):
        return self.partition.shape

    # -------------------------------------------------------------------------

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

        Raises
        ------
        AttributeError
            If blocks' `shape` and dataset's shape have different numbers of axes.


        **Acknowledgments**

        Special thanks go to Paweł Weroński for the help with the design of
        the general *partition* algorithm.

        Examples
        --------
        >>> bdm = BDM(ndim=2, shape=(2, 2))
        >>> [ x for x in bdm.decompose(np.ones((4, 3), dtype=int)) ]
        [array([[1, 1],
               [1, 1]]), array([[1, 1],
               [1, 1]])]
        """
        yield from self.partition.decompose(X)

    def count_blocks(self, blocks, buffer_size=None):
        """Count unique blocks and map them to standard and normalized
        integer codes.

        Parameters
        ----------
        blocks : iterable of array_like
            Data blocks.
        buffer_size : int
            Size of block chunks to use to speed up computations.
            Non-positive values mean all blocks will be processed
            at once (not recommended for large datasets).
            Use global/instance option value (`'count_buffer_size'`)
            when ``None``.

        Returns
        -------
        dict
            Maps block shapes to :py:class:`collections.Counter` objects
            counting unique block codes.
        """
        if buffer_size is None:
            buffer_size = self.options.bdm_buffer_size
        blocks = chunked_buckets(blocks, n=buffer_size, key=lambda x: x.shape)
        dct = defaultdict(Counter)

        for shape, chunk in blocks:
            arr = np.empty((len(chunk), prod(shape)), dtype=INT_DTYPE)
            for i, a in enumerate(chunk):
                arr[i] = a.flatten()
            codes = encode_sequences(arr, base=self.nsymbols)
            codes_n = encode_sequences(
                normalize_sequences(arr, base=self.nsymbols),
                base=self.nsymbols
            )
            dct[shape] += Counter(zip(codes_n, codes))
        return BlockCounter(dct)

    def decompose_and_count(self, X, **kwds):
        """Decompose a dataset and count blocks.
        ``**kwds`` are passed to :py:meth:`count_blocks`.

        See also
        --------
        decompose : object decomposition
        count_blocks : block counting
        """
        blocks = self.decompose(X)
        return self.count_blocks(blocks, **kwds)

    def get_freq(self, counter):
        """Get array for block frequencies."""
        return np.hstack([
            np.array(list(counter[shape].values()), dtype=INT_DTYPE)
            for shape in counter
        ])

    def get_cmx(self, counter):
        """Get array of CTM values."""
        return np.hstack([
            self.ctm.get(shape, map(lambda x: x[0], counter[shape]))
            for shape in counter
        ])

    def lookup(self, *counters):
        """Lookup CTM values.

        Parameters
        ----------
        *counters : Counter
            Counter objects mapping codes to frequences.

        Returns
        -------
        1D array_like
            Frequencies of unique blocks.
        1D array_like
            CTM complexity values for unique blocks.
        """
        counter = reduce(lambda x, y: x + y, counters)
        freq = self.get_freq(counter)
        cmx = self.get_cmx(counter)
        return freq, cmx

    def calc_bdm(self, *counters):
        """Estimate Kolmogorov complexity based on the BDM formula.

        Parameters
        ----------
        counters : dict-like
            Counters dictionaries as returned by
            :py:meth:`count_blocks`.

        Returns
        -------
        float
            Estimated algorithmic complexity.

        Notes
        -----
        Detailed description can be found in :doc:`theory`.

        Examples
        --------
        >>> from collections import Counter
        >>> bdm = BDM(ndim=1)
        >>> c1 = Counter([('111111111111', 1.95207842085224e-08)])
        >>> c2 = Counter([('111111111111', 1.95207842085224e-08)])
        >>> bdm.compute_bdm(c1, c2) # doctest: +FLOAT_CMP
        1.000000019520784
        """
        freq, cmx = self.lookup(*counters)
        return np.log2(freq).sum() + cmx.sum()

    def bdm(self, X, normalized=False, check_data=None, **kwds):
        """Approximate complexity of a dataset with BDM.

        Parameters
        ----------
        X : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.
        normalized : bool
            Should BDM be normalized to be in the [0, 1] range.
        check_data : bool
            Should data format be checked.
            May be disabled to gain some speed when calling multiple times.
            Use `options` attribute if ``None``.
        **kwds :
            Passed to :py:meth:`count_blocks`.

        Returns
        -------
        float
            Estimated algorithmic complexity.

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

        Notes
        -----
        Detailed description can be found in :doc:`theory`.

        Examples
        --------
        >>> import numpy as np
        >>> bdm = BDM(ndim=2)
        >>> bdm.bdm(np.ones((12, 12), dtype=int)) # doctest: +FLOAT_CMP
        25.176631293734488
        """
        self.check_data(X, check_data)

        counter = self.decompose_and_count(X, **kwds)
        cmx = self.calc_bdm(counter)

        if self.options.bdm_if_zero == 'raise' and cmx == 0:
            raise ValueError("Computed BDM is 0, dataset may have incorrect dimensions")

        if normalized:
            min_cmx = self._get_min_bdm(counter)
            max_cmx = self._get_max_bdm(counter)
            cmx = (cmx - min_cmx) / (max_cmx - min_cmx)

        return cmx

    def nbdm(self, X, **kwds):
        """Alias for normalized BDM

        Other arguments are passed as keywords.

        See also
        --------
        bdm : BDM method
        """
        return self.bdm(X, normalized=True, **kwds)

    def calc_ent(self, *counters, rv=False):
        """Calculate block entropy from a counter obejct.

        Parameters
        ----------
        counter : dict-like
            Counters dictionary as returned by
            :py:meth:`count_blocks`.
        rv : bool
            Calculate entropy of a single random variable
            yielding a random block.

        Returns
        -------
        float
            Block entropy in base 2.
            It is equal to the entropy of the frequency distribution
            of blocks multipled by the number of blocks.

        Examples
        --------
        >>> from collections import Counter
        >>> bdm = BDM(ndim=1)
        >>> c1 = Counter([('111111111111', 1.95207842085224e-08)])
        >>> c2 = Counter([('000000000000', 1.95207842085224e-08)])
        >>> bdm.compute_ent(c1, c2) # doctest: +FLOAT_CMP
        1.0
        """
        counter = reduce(lambda x, y: x + y, counters)
        freq = self.get_freq(counter)
        n_blocks = freq.sum()
        p = freq / n_blocks
        p = p[p > 0]
        ent = -(p*np.log2(p)).sum()
        if not rv:
            ent *= n_blocks
        return ent

    def ent(self, X, normalized=False, check_data=None, rv=False, **kwds):
        """Block entropy of a dataset.

        Parameters
        ----------
        X : array_like
            Dataset representation as a :py:class:`numpy.ndarray`.
            Number of axes must agree with the `ndim` attribute.
        normalized : bool
            Should entropy be normalized to be in the [0, 1] range.
        check_data : bool
            Should data format be checked.
            May be disabled to gain some speed when calling multiple times.
            Use `options` attribute if ``None``.
        rv : bool
            Calculate entropy of a single random variable
            yielding a random block.
        **kwds :
            Passed to :py:meth:`count_blocks`.

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
        >>> bdm = BDM(ndim=2)
        >>> bdm.ent(np.ones((12, 12), dtype=int)) # doctest: +FLOAT_CMP
        0.0
        """
        self.check_data(X, check_data)

        counter = self.decompose_and_count(X, **kwds)
        ent = self.calc_ent(counter, rv=rv or normalized)

        if normalized:
            min_ent = self._get_min_ent(counter)
            max_ent = self._get_max_ent(counter)
            ent = (ent - min_ent) / (max_ent - min_ent)

        return ent

    def nent(self, X, **kwds):
        """Alias for normalized block entropy.

        Other arguments are passed as keywords.

        See also
        --------
        ent : block entropy method
        """
        return self.ent(X, normalized=True, **kwds)

    def check_data(self, X, check=None):
        """Check if data is correctly formatted.

        Symbols have to mapped to integers from ``0`` to `self.nsymbols-1`.
        """
        if check is None:
            check = self.options.bdm_check_data
        if not check:
            return
        if not issubclass(X.dtype.type, np.integer):
            raise TypeError("'X' has to be an integer array")
        symbols = np.unique(X)
        if symbols.size > self.nsymbols:
            raise ValueError("'X' has more than {} unique symbols".format(
                self.nsymbols
            ))
        valid_symbols = np.arange(self.nsymbols)
        bad_symbols = symbols[~np.isin(symbols, valid_symbols)]
        if bad_symbols.size > 0:
            raise ValueError("'X' contains symbols outside of [0, {}]: {}".format(
                str(self.nsymbols-1),
                ", ".join(str(s) for s in bad_symbols)
            ))

    def _get_min_bdm(self, counter):
        bdm = 0
        for shape, v in counter.items():
            cmx = self.ctm[shape][0]
            freq = log2(sum(v.values()))
            bdm += cmx + freq
        return bdm

    def _get_max_bdm(self, counter):
        bdm = 0
        for shape, v in counter.items():
            n_blocks = sum(v.values())
            n_uniq = self.ctm.info[shape]['blocks_total']
            n_enum = self.ctm.info[shape]['blocks_enum']
            n_miss = min(n_blocks, n_uniq - n_enum)
            n_cmx = min(n_blocks - n_miss, n_uniq)
            div = min(n_blocks, n_uniq)
            freq = div * log2(n_blocks / div)
            cmx = n_miss * self.ctm.missing[shape] + \
                self.ctm[shape].values[:n_cmx].sum()
            bdm += cmx + freq
        return bdm

    def _get_min_ent(self, counter):
        freq = np.array([ len(v) for v in counter.values() ])
        p = freq / freq.sum()
        ent = -(p * np.log2(p)).sum()
        return ent

    def _get_max_ent(self, counter):
        n_blocks = np.hstack([
            np.array(list(x.values()))
            for x in counter.values()
        ]).sum()
        n_uniq = sum(
            self.ctm.info[shape]['blocks_total']
            for shape in counter
        )
        ent = log2(min(n_blocks, n_uniq))
        return ent
