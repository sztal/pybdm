"""Encoding and decoding of arrays with fixed number of unique symbols.

While computing BDM dataset parts have to be encoded into simple hashable objects
such as strings or integers for efficient lookup of CTM values from reference
datasets.

In case of CTM dataset containing objects with several different dimensionalities
string keys have to be used and this representation is used by
:py:mod:`bdm.stages` functions at the moment.

Integer encoding can be used for easy generation of objects
of fixed dimensionality as each such object using a fixed,
finite alphabet of symbols can be uniquely mapped to an integer code.
"""
from collections import deque
import numpy as np


def array_from_string(x, shape, cast_to=int):
    """Make array from string code.

    Parameters
    ----------
    x : str
        String code.
    shape : tuple
        Desired shape of the output array.
    cast_to : type or None
        Cast array to given type. No casting if ``None``.
        Defaults to integer type.

    Returns
    -------
    array_like
        Array encoded in the string code.

    Examples
    --------
    >>> array_from_string('1010', shape=(4,))
    array([1, 0, 1, 0])
    >>> array_from_string('1000', shape=(2, 2))
    array([[1, 0],
           [0, 0]])
    """
    arr = np.array(list(x))
    if arr.ndim == 0:
        arr = arr.reshape((1, ))
    if cast_to:
        arr = arr.astype(cast_to)
    return arr.reshape(shape)

def string_from_array(arr):
    """Encode an array as a string code.

    Parameters
    ----------
    arr : (N, k) array_like
        *Numpy* array.

    Returns
    -------
    str
        String code of an array.

    Examples
    --------
    >>> string_from_array(np.array([1, 0, 0]))
    '100'
    >>> string_from_array(np.array([[1,0], [3,4]]))
    '1034'
    """
    x = np.apply_along_axis(''.join, arr.ndim - 1, arr.astype(str))
    x = ''.join(np.ravel(x))
    return x

def encode_sequence(seq, base=2):
    """Encode sequence of integer-symbols.

    Parameters
    ----------
    seq : (N, ) array_like
        Sequence of integer symbols represented as 1D *Numpy* array.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.

    Returns
    -------
    int
        Integer code of a sequence.

    Raises
    ------
    AttributeError
        If `seq` is not 1D.
    TypeError
        If `seq` is not of integer type.
    ValueError
        If `seq` contain values which are negative or beyond the size
        of the alphabet (encoding base).

    Examples
    --------
    >>> encode_sequence(np.array([1, 0, 0]))
    4
    """
    if seq.size == 0:
        return 0
    if seq.ndim != 1:
        raise AttributeError("'seq' has to be a 1D array")
    if seq.dtype != np.int:
        raise TypeError("'seq' has to be of integer dtype")
    if not (seq >= 0).all():
        raise ValueError("'seq' has to consist of non-negative integers")
    proper_values = np.arange(base)
    if not np.isin(seq, proper_values).all():
        raise ValueError(f"There are symbol codes greater than {base-1}")
    code = 0
    for i, x in enumerate(reversed(seq)):
        if x > 0:
            code += x * base**i
    return code

def decode_sequence(code, base=2, min_length=None):
    """Decode sequence from a sequence code.

    Parameters
    ----------
    code : int
        Non-negative integer.
    base : int
        Encoding base.
        Should be equal to the number of unique symbols in the alphabet.
    min_length : int or None
        Minimal number of represented bits.
        Use shortest representation if ``None``.

    Returns
    -------
    array_like
        1D *Numpy* array.

    Examples
    --------
    >>> decode_sequence(4)
    array([1, 0, 0])
    """
    bits = deque()
    while code > 0:
        code, rest = divmod(code, base)
        bits.appendleft(rest)
    n = len(bits)
    if min_length and n < min_length:
        for _ in range(min_length - n):
            bits.appendleft(0)
    return np.array(bits)

def encode_array(x, base=2, **kwds):
    """Encode array of integer-symbols.

    Parameters
    ----------
    x : (N, k) array_like
        Array of integer symbols.
    base : int
        Encoding base.
    **kwds :
        Keyword arguments passed to :py:func:`numpy.ravel`.

    Returns
    -------
    int
        Integer code of an array.
    """
    seq = np.ravel(x, **kwds)
    return encode_sequence(seq, base=base)

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
        Keyword arguments passed to :py:func:`numpy.reshape`.

    Returns
    -------
    array_like
        *Numpy* array.
    """
    length = np.multiply.reduce(shape)
    seq = decode_sequence(code, base=base, min_length=length)
    if seq.size > length:
        raise ValueError(f"{code} does not encode array of shape {shape}")
    arr = seq.reshape(shape, **kwds)
    return arr

def normalize_array(X):
    """Normalize array so symbols are consecutively mapped to 0, 1, 2, ...

    Parameters
    ----------
    X : array_like
        *Numpy* array of arbitrary dimensions.

    Returns
    -------
    array_like
        *Numpy* array of the same dimensions with mapped symbols.

    Examples
    --------
    >>> X = np.array([1, 2, 3], dtype=int)
    >>> normalize_array(X)
    array([0, 1, 2])
    >>> X = np.array([[1,2],[2,1]], dtype=int)
    >>> normalize_array(X)
    array([[0, 1],
           [1, 0]])
    """
    shp = X.shape
    ndim = X.ndim
    dct = {}
    counter = 0
    X = X.copy()
    if ndim > 1:
        X = X.ravel()
    for idx, x in np.ndenumerate(X):
        if x not in dct:
            dct[x] = counter
            counter += 1
        X[idx] = dct[x]
    if ndim > 1:
        X = X.reshape(shp)
    return X

def normalize_key(key):
    """Normalize part key so symbols are consecutively mapped to 0, 1, 2, ...

    Parameters
    ----------
    key : str
        Part key as returned by :py:func:`string_from_array`.

    Returns
    -------
    str
        Normalized key with mapped symbols.

    Examples
    --------
    >>> normalize_key('123')
    '012'
    >>> normalize_key('40524')
    '01230'
    """
    dct = {}
    counter = 0

    def _normalize(s):
        nonlocal dct
        nonlocal counter
        if s not in dct:
            dct[s] = counter
            counter += 1
        return str(dct[s])

    return ''.join(_normalize(s) for s in key)
