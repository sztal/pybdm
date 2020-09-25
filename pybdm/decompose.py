"""Decomposition functions.

Decomposition functions serve two interrelated purposes.
The first one is mapping between raw and block data representations,
where a block representation is a view of an array in terms of
blocks of a given shape it can be decomposed to.

The second one is decomposing arrays by iterating over blocks
of given shapes.
"""
import math
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
