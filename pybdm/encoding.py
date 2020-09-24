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
import math
from collections import deque
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

def decode_sequence(code, shape=None, base=2, dtype=None):
    """Decode sequence from a sequence code.

    Parameters
    ----------
    code : int
        Non-negative integer.
    shape : tuple of int, optional
        Shape tuple the sequence should be conformable with.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.
    dtype : numpy.dtype, optional
        Data type of the resulting array.

    Returns
    -------
    array_like
        1D *Numpy* array.

    Raises
    ------
    ValueError
        If optional 'shape' is not conformable with decoded sequence.

    Examples
    --------
    >>> decode_sequence(4)
    array([1, 0, 0])
    >>> decode_sequence(447, shape=(8,), base=4)
    array([0, 0, 0, 1, 2, 3, 3, 3])
    """
    if dtype is not None and not issubclass(dtype, np.integer):
        raise TypeError("'dtype' has to be an integer class")

    bits = deque()
    while code > 0:
        code, rest = divmod(code, base)
        bits.appendleft(rest)
    if shape is not None:
        size = math.prod(shape)
        pad = size - len(bits)
        if pad > 0:
            for _ in range(pad):
                bits.appendleft(0)
        elif pad < 0:
            raise ValueError("sequence is not conformable with 'shape'")
    return np.array(bits, dtype=dtype)

def decode_array(code, shape, base=2, **kwds):
    """Decode array of integer-symbols from a sequence code.

    Parameters
    ----------
    code : int
        Non-negative integer.
    shape : tuple of ints
        Expected array shape.
    base : int
        Encoding base.
    **kwds :
        Passed to :func:`decode_sequence`.

    Returns
    -------
    array_like
        *Numpy* array.

    Raises
    ------
    TypeError
        If `dtype` is not an integer type.
    ValueError
        If size of the decoded sequence is greater by the size implied
        by `shape`.
    """
    seq = decode_sequence(code, shape=shape, base=base, **kwds)
    # pylint: disable=invalid-unary-operand-type
    return seq.reshape(shape, order='C')
