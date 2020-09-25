"""Encoding and decoding of arrays with fixed number of unique symbols.

A convenient feature of arrays with fixed number of unique symbols
is that they can always be uniquely represented with single integer
codes and the set of all possible arrays of a given size spans all integers
from ``0`` to ``n``. This allows mapping CTM values for all parts up to
a given size to simple *Numpy* arrays with integer indices
which in turn allows implementation of both memory and computation time
efficient algorithms for CTM lookup and BDM calculations.

Integer encoding can be also used for easy generation of objects
of fixed dimensionality as each such object using a fixed,
finite alphabet of symbols can be uniquely mapped to an integer code.

Before encoding arrays are flattened in row-major (C-style) order.
"""
from math import prod
from functools import singledispatch
import numpy as np


def encode_sequences(arr, base=2):
    """Convert discrete sequence(s) to integer codes.

    Parameters
    ----------
    arr : (N, k) array_like
        2D array with sequences stacked in rows.
        1D arrays are interpreted as a single sequence.
    base : int
        Encoding base.

    Returns
    -------
    (N,) array_like
        Array of integer codes.

    Raises
    ------
    ValueError
        If `base` is not an integer equal or grater than ``2``.
    AttributeError
        If `arr` is not a 2D or 1D array.

    Examples
    --------
    >>> X = np.array([[1,0,2,0], [0,1,2,3]])
    >>> encode_sequences(X, base=4)
    array([72, 27])
    """
    if not isinstance(base, (int, np.integer)) or base < 2:
        raise ValueError("'base' has to be an integer equal or greater than '2'")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1, order='C')
    if arr.ndim != 2:
        raise AttributeError("'arr' has to be 2D array")

    mul = base ** np.flip(np.arange(arr.shape[-1]))
    return (arr * mul).sum(1)

def normalize_sequences(arr, base=2):
    """Normalize sequences with `base` unique symbols.

    Normalized sequence is a sequence in which first unique symbol
    is always ``0``, second is ``1`` and so on.

    Parameters
    ----------
    arr : (N, k) array_like
        2D array with sequences stacked in rows.
        1D arrays are interpreted as a single sequence.
    base : int
        Encoding base (number of unique symbols).

    Returns
    -------
    (N, k) array_like
        2D array of normalized sequences stacked in rows.

    Raises
    ------
    ValueError
        If `base` is not an integer equal or grater than ``2``.
    AttributeError
        If `arr` is not a 2D or 1D array.

    Examples
    --------
    >>> X = np.array([1, 0, 2, 0], [0, 1, 2, 3])
    >>> normalize_sequences(X, base=4)
    array([[0, 1, 2, 1],
           [0, 1, 2, 3]])
    """
    if not isinstance(base, (int, np.integer)) or base < 2:
        raise ValueError("'base' has to be an integer equal or greater than '2'")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise AttributeError("'arr' has to be 2D array")

    ridx = np.arange(arr.shape[0])
    arr = -arr - 1

    for n in range(base):
        syms = arr[(ridx, np.argmax(arr < 0, axis=1))]
        arr[(arr == syms[:, None]) & (arr < 0)] = n

    return arr

@singledispatch
def decode_sequences(codes, shape, base, dtype=int):
    """Decode sequence(s) from sequence code(s).

    Parameters
    ----------
    codes : int or 1D non-negative integer ndarray
        Non-negative integers.
    shape : tuple of int
        Block shape for sequences.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.
    dtype : numpy.dtype
        Data type of the resulting array of sequences.

    Returns
    -------
    array_like
        2D integer array.
        Each row gives a sequence representing a single
        block (flattened).

    Raises
    ------
    ValueError
        If optional 'shape' is not conformable with decoded sequence.

    Examples
    --------
    >>> decode_sequences(4, shape=(3,), base=2)
    array([1, 0, 0])
    >>> decode_sequences(447, shape=(8,), base=4)
    array([0, 0, 0, 1, 2, 3, 3, 3])
    >>> import numpy as np
    >>> codes = np.array([4, 447])
    >>> decode_sequences(codes, shape=(6,), base=4)
    array([[0, 0, 0, 0, 1, 0]
           [0, 1, 2, 3, 3, 3]])
    """
    width = prod(shape)
    arr = np.zeros((width,), dtype=dtype)

    for i in range(width):
        div = base**(width - i - 1)
        mult, codes = divmod(codes, div)
        arr[i] = mult

    return arr

@decode_sequences.register(np.ndarray)
def _(codes, shape, base, dtype=int):
    if codes.ndim != 1 or not issubclass(codes.dtype.type, np.integer):
        raise AttributeError("'codes' has to be 1D integer array")

    width = prod(shape)
    arr = np.zeros((codes.shape[0], width), dtype=dtype)

    for i in range(width):
        div = base**(width - i - 1)
        mult, codes = np.divmod(codes, div)
        arr[:, i] = mult

    return arr
