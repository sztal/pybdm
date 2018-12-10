"""Block decomposition stage functions.

Stage functions contain the main logic of the *Block Decomposition Method*
and its different flavours depending on boundary conditions etc.
The implementation follows the *split-apply-combine* approach.
In the first stage an input dataset is partitioned into parts of shape
appropriate for the selected CTM dataset (split stage).
Next, approximate complexity for parts based on the *Coding Theorem Method*
is looked up in the reference (apply stage) dataset.
Finally, values for individual parts are aggregated into a final BDM value
(combine stage).

The partition (split) stage is implemented by the family of ``partition_*`` functions.
The lookup (apply) stage is implemented by the family of ``lookup_*`` functions.
The aggregate (combine) stage is implented by the family of ``aggregate_*`` functions.

The general principle is that specific functions should in most cases
be wrappers around the core family functions, which accordingly are:

* :py:func:`bdm.stages.partition`
* :py:func:`bdm.stages.lookup`
* :py:func:`bdm.stages.aggregate`
"""
from collections import Counter
from functools import reduce
import numpy as np
from .encoding import string_from_array
from .utils import get_reduced_shape, get_reduced_idx


def partition(x, shape, shift=0, reduced_idx=None):
    """Standard partition stage function.

    Parameters
    ----------
    x : array_like
        Dataset of arbitrary dimensionality represented as a *Numpy* array.
    shape : tuple
        Dataset parts' shape.
    shift : int
        Shift of the sliding window.
        In general, if positive, should not be greater than ``1``.
        Shift by partition shape if not positive.
    reduced_idx : iterable or None
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
        If parts' `shape` is not equal in each dimension.
        If parts' `shape` and dataset's shape have different numbers of axes.

    Acknowledgments
    ---------------
    Special thanks go to Paweł Weroński for the help with the design of
    the non-recursive *partition* algorithm.

    Examples
    --------
    >>> [ x for x in partition(np.ones((3, 3), dtype=int), shape=(2, 2)) ]
    [array([[1, 1],
           [1, 1]]), array([[1],
           [1]]), array([[1, 1]]), array([[1]])]
    """
    r_shape = get_reduced_shape(x, shape, length_only=False)
    n_parts = int(np.multiply.reduce(r_shape))
    reduced_idx = reduced_idx if reduced_idx else range(n_parts)
    width = shape[0]
    _shift = shift if shift > 0 else width
    for i in reduced_idx:
        r_idx = get_reduced_idx(i, r_shape)
        if shift <= 0:
            idx = tuple(slice(k*width, k*width + _shift) for k in r_idx)
        else:
            idx = tuple(slice(k, k + width) for k in r_idx)
        yield x[idx]

def partition_ignore(x, shape, reduced_idx=None):
    """Partition with ignore leftovers boundary condition.

    In this variant parts that can not be further sliced and fitted into
    the desired `shape` are simply omitted.

    Parameters
    ----------
    x : array_like
        Dataset of arbitrary dimensionality represented by a *Numpy* array.
    shape : tuple
        Shape of parts.
    reduced_idx : iterable or None
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
        If parts' `shape` is not equal in each dimension.
        If parts' `shape` and dataset's shape are not conformable.

    Examples
    --------
    >>> [ x for x in partition_ignore(np.ones((3, 3), dtype=int), shape=(2, 2)) ]
    [array([[1, 1],
           [1, 1]])]
    """
    for part in partition(x, shape, shift=0, reduced_idx=reduced_idx):
        if part.shape == shape:
            yield part

def partition_shrink(x, shape, min_length=2, reduced_idx=None):
    """Partition stage function with a shrinking parts' size.

    Parameters
    ----------
    x : array_like
        Dataset of arbitrary dimensionality represented as a *Numpy* array.
    shape : tuple
        Dataset parts' shape.
    min_dim_length : int
        Minimum parts' length.
        In case of multidimensional objects it specifies minimum
        length of any single dimension.
    reduced_idx : iterable or None
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
        If parts' `shape` is not equal in each dimension.
        If parts' `shape` and dataset's shape have different numbers of axes.

    Examples
    --------
    >>> [ p for p in partition_shrink(np.ones(10, ), (6, ), min_length=4) ]
    [array([1., 1., 1., 1., 1., 1.]), array([1., 1., 1., 1.])]
    """
    for part in partition(x, shape, shift=0, reduced_idx=reduced_idx):
        if part.shape == shape:
            yield part
        else:
            min_dim_length = min(part.shape)
            if min_dim_length < min_length:
                continue
            shrinked_shape = tuple(min_dim_length for _ in range(len(shape)))
            yield from partition(part, shrinked_shape, shift=0, reduced_idx=None)


def lookup(parts, ctm, sep='-'):
    """Lookup CTM values for parts in a reference dataset.

    Parameters
    ----------
    parts : sequence
        Ordered sequence of dataset parts.
    ctm : dict
        Reference CTM dataset.
    sep : str
        Sequence separator in string codes.

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
    >>> from bdm import BDM
    >>> bdm = BDM(ndim=1)
    >>> data = np.ones((16, ), dtype=int)
    >>> parts = partition_ignore(data, (12, ))
    >>> [ x for x in lookup(parts, bdm._ctm) ]
    [('111111111111', 1.95207842085224e-08)]
    """
    for part in parts:
        key = string_from_array(part)
        try:
            if sep in key:
                cmx = ctm[key]
            else:
                cmx = ctm.get(key, ctm[key.lstrip('0')])
        except KeyError:
            raise KeyError(f"CTM dataset does not contain object '{key}'")
        yield key, cmx

def aggregate(ctms):
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
    >>> from bdm import BDM
    >>> bdm = BDM(ndim=1)
    >>> data = np.ones((30, ), dtype=int)
    >>> parts = partition_ignore(data, (12, ))
    >>> ctms = lookup(parts, bdm._ctm)
    >>> aggregate(ctms)
    Counter({('111111111111', 1.95207842085224e-08): 2})
    """
    counter = Counter()
    for key, ctm in ctms:
        counter.update([ (key, ctm) ])
    return counter

def compute_bdm(*counters):
    """Compute BDM approximation.

    Parameters
    ----------
    *counters :
        Counter objects grouping object keys and occurences.

    Returns
    -------
    float
        Approximate algorithmic complexity.
    """
    counter = reduce(lambda x, y: x+y, counters)
    bdm = 0
    for key, n in counter.items():
        _, ctm = key
        bdm += ctm + np.log2(n)
    return bdm
