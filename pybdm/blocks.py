"""Decomposition functions.

Decomposition functions serve two interrelated purposes.
The first one is mapping between raw and block data representations,
where a block representation is a view of an array in terms of
blocks of a given shape it can be decomposed to.

The second one is decomposing arrays by iterating over blocks
of given shapes.
"""
import math
from collections import Counter
from itertools import product
from functools import singledispatch
import numpy as np


def get_block_shape(X, shape):
    """Get block shape of an array.

    Parameters
    ----------
    X : array_like
        Array of arbitrary dimensions.
    shape : tuple of int
        Block shape.

    Raises
    ------
    ValueError
        If `len(shape)` is not equal to `X.ndim`.

    Returns
    -------
    tuple
        Block shape tuple.

    Examples
    --------
    >>> X = np.array([1, 2, 3, 4])
    >>> get_block_shape(X, shape=(2,))
    (2,)
    >>> X = np.array([1, 2, 3])
    >>> get_block_shape(X, shape=(3,))
    (2,)
    """
    if len(shape) != X.ndim:
        raise ValueError("block 'shape' and 'X.ndim' are not conformable")
    return tuple(math.ceil(x / y) for x, y in zip(X.shape, shape))

def get_block_slice(idx, shape):
    """Get slice indices for the raw data array from block ids and block shape.

    Parameters
    ----------
    idx : tuple of int
        Block index.
    shape : tuple of int
        Data shape in block representation.

    Returns
    -------
    tuple of slice
        Slice indices of block elements in raw data array.

    Raises
    ------
    ValueError
        If `len(idx)` is not equal to `len(shape)`.

    Examples
    --------
    >>> get_block_slice((2, 2), shape=(4,4))
    (slice(8, 12, None), slice(8, 12, None))
    """
    if len(idx) != len(shape):
        raise ValueError("'idx' and 'shape' are not conformable")
    return tuple(slice(x*y, (x+1)*y) for x, y in zip(idx, shape))

def get_block(X, idx, shape):
    """Get dataset block.

    Parameters
    ----------
    X : array_like
        Array of arbitrary dimensions.
    idx : tuple of int
        Block index.
    shape : tuple of int
        Data shape in block representation.

    Returns
    -------
    array_like
        Dataset block.
    """
    return X[get_block_slice(idx, shape)]

@singledispatch
def get_block_idx(idx, shape, unique=True):
    """Get block index from raw index.

    Multiple indexes can be passed at once as `numpy.ndarray` object
    (with individual indices in rows).

    Parameters
    ----------
    idx : tuple of int or ndarray
        Raw index or array with raw indexes in rows.
    shape : tuple of int
        Data shape in block representation.
    unique : bool
        Should only unique block indexes be returned
        in case multiple raw indexes are used.
        Ignored when `idx` is a single `tuple`.

    Returns
    -------
    tuple of int
        Block index.

    Raises
    ------
    ValueError
        If `len(idx)` is not equal to `len(shape)`.

    Examples
    --------
    >>> get_block_idx((6,), (4,))
    (1,)
    >>> get_block_idx((6,10), (4,4))
    (1, 2)
    """
    # pylint: disable=unused-argument
    if len(idx) != len(shape):
        raise ValueError("'idx' and 'shape' are not conformable")
    return tuple(x // y for x, y in zip(idx, shape))

@get_block_idx.register(np.ndarray)
def _(idx, shape, unique=True):
    if isinstance(shape, tuple):
        shape = np.array(shape)
    idx = idx.squeeze()
    if idx.ndim == 1:
        if shape.size == 1:
            idx = idx.reshape((-1, 1), order='C')
        else:
            idx = idx.reshape((1, -1), order='C')
    if idx.ndim != 2 or idx.shape[1] != shape.size:
        raise ValueError("'idx' and 'shape' are not conformable")
    block_idx = idx // shape
    if unique:
        block_idx = np.unique(block_idx, axis=0)
    return block_idx

def iter_block_slices(X, shape):
    """Iterate over block slices.

    Parameters
    ----------
    X : array_like
        Array of arbitrary dimensions.
    shape : tuple of int
        Block shape.

    Raises
    ------
    ValueError
        If `len(shape)` is not equal to `X.ndim`.

    Yields
    ------
    slice
        Block slices.

    Examples
    --------
    >>> X = np.arange(16).reshape(8, 2)
    >>> [ s for s in iter_block_slices(X, (4,4)) ]
    [(slice(0, 4, None), slice(0, 4, None)),
     (slice(4, 8, None), slice(0, 4, None))]
    """
    block_shape = get_block_shape(X, shape)
    for idx in product(*(range(x) for x in block_shape)):
        yield get_block_slice(idx, shape)

def block_decompose(X, shape):
    """Block decompose a dataset.

    Parameters
    ----------
    X : array_like
        Array of arbitrary dimensions.
    shape : tuple of int
        Block shape.

    Raises
    ------
    ValueError
        If `len(shape)` is not equal to `X.ndim`.

    Yields
    ------
    array_like
        Dataset blocks.

    Examples
    --------
    >>> X = np.arange(8).reshape(8, 2)
    >>> [ b for b in block_decompose(X, (2, 2)) ]
    [array([[0, 1],
            [2, 3]]),
     array([[4, 5],
            [6, 7]])]
    """
    for idx in iter_block_slices(X, shape):
        yield X[idx]


class BlockCounter:
    """Block counter.

    `BlockCounter` is a wrapper around a map from block shapes
    to block :py:class:`collections.Counter` objects with methods
    for adding, subtracting and crossing different counters.
    This is useful in particular in the context of perturbation analyses.

    Attributes
    ----------
    counters : mapping
        A mapping from shape tuples to
        :py:class:`collections.Counter` objects.
    """
    def __init__(self, counters):
        self.counters = dict(counters)

    def __repr__(self):
        return "{cn}({counters})".format(
            cn=self.__class__.__name__,
            counters=self.counters
        )

    def __iter__(self):
        return self.counters.__iter__()

    def __getitem__(self, key):
        return self.counters[key]

    def __setitem__(self, key, value):
        self.counters[key]= value

    def __delitem__(self, key):
        del self.counters[key]

    def __add__(self, other):
        """Add other block counter object."""
        out = self.copy()
        for shape, counter in other.items():
            out[shape].update(counter)
        return out._clean()

    def __sub__(self, other):
        """Subtract other block counter object."""
        out = self.copy()
        for shape, counter in other.items():
            out[shape].subtract(counter)
        return out._clean()

    def copy(self):
        return self.__class__({
            k: v.copy() for k, v in self.counters.items()
        })

    def get(self, key, fallback=None):
        return self.counters.get(key, fallback)

    def items(self):
        return self.counters.items()

    def values(self):
        return self.counters.values()

    def update(self, other):
        """Update `self` with respect to `other`.

        Non-positive counts can be present in `other`,
        but are dropped after updating from `self`.
        """
        for shape in other:
            self[shape] += other[shape]

    # -------------------------------------------------------------------------

    def keydiff(self, other):
        """Get set-difference of keys between two block counters.

        Returns
        -------
        dict
            From shapes to :py:class:`numpy.ndarrays` with
            normalized block codes.
        """
        dct = {
            shape: np.array([
                k for k in counter if k not in other.get(shape, {})
            ]) for shape, counter in self.items()
        }
        return { k: v for k, v in dct.items() if v.size > 0 }

    def _clean(self):
        for shape in self:
            self[shape] = Counter({
                k: v for k, v in self[shape].items() if v != 0
            })
        self.counters = { k: v for k, v in self.items() if v }
        return self

    # -------------------------------------------------------------------------
