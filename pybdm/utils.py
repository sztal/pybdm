"""Utility functions."""
# pylint: disable=no-name-in-module,anomalous-backslash-in-string
from collections import defaultdict
import numpy as np
from scipy.special import binom, gammaln


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
    U, idx = np.unique(X, return_index=True)
    U = U[np.argsort(idx)]
    N = np.empty_like(X)
    for i, uniq in enumerate(U):
        N = np.where(X == uniq, i, N)
    return N.astype(int)

def chunked_buckets(iterable, n, key):
    """Chunked buckets generator.

    It is an iterable in which items are grouped by a key function,
    accumulated up to a given size in groups and yielded only after.

    Parameters
    ----------
    iterable : iterable
        Items to iterate over.
    n : int
        Batch size. Accumulate all items at once if non-positive.
    key : callable
        Key function used for grouping items.

    Yields
    ------
    key, items
        2-tuple with key value and accumulated items stored in a list.
    """
    dct = defaultdict(list)

    for item in iterable:
        k = key(item)
        dct[k].append(item)
        if 0 < n <= len(dct[k]):
            yield k, dct.pop(k)

    yield from dct.items()

def n_distinct(arr, axis=None):
    """Count number of distinct elements in an array.

    Parameters
    ----------
    arr : array_like
        An array with arbitrary number of dimensions.
    axis : int, optional
        Axis to count elements by.
        Count over all elements when ``None``.

    Returns
    -------
    integer array or int
        Counts of distinct values.

    Examples
    --------
    >>> A = np.array([[0, 1], [2, 2]])
    >>> n_distinct(A, axis=None)
    3
    >>> n_distinct(A, axis=0)
    array([2, 2])
    >>> n_distinct(A, axis=1)
    array([2, 1])
    """
    arr = np.sort(arr, axis=axis)
    return (np.diff(arr, axis=axis or 0) != 0).sum(axis=axis) + 1

def S2(n, k):
    """Stirling numbers of the second kind.

    They give a number of different splits of a set of ``n`` elements
    into ``k`` subsets. The explicit formula is used for efficient
    calculations:

    .. math:: \frac{1}{k!}\sum_{i=0}^k(-1)^{k-i}\binom{k,i}i^n

    Parameters
    ----------
    n : int, positive
        Number of elements in the set.
    k : int, positive
        Number of subsets.

    Raises
    ------
    ValueError
        If `k` is greater than `n`.

    Notes
    -----
    This function is supposed to be used with care.
    For larger values it is likely to crash due to integer overflow.
    """
    if k > n:
        raise ValueError("'k' cannot be greater than 'n'")
    i = np.arange(k+1)

    mul = np.exp(-gammaln(k+1))
    run = np.sum((-1)**(k-i)*binom(k, i)*i**n)

    return round(mul*run)

def count_ctm_classes(n, k):
    """Count CTM equivalence classes.

    It is equal to:

    .. math:: \sum_{i=1}^k S(n, i)

    where :math:`S(n, i)` are Stirling numbers of the second kind.

    Parameters
    ----------
    n : int, positive
        Number of elements in a block.
    k : int, positive
        Number of symbols.
    """
    k = min(n, k)
    return sum(S2(n, i) for i in range(1, k+1))
