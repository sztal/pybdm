"""Utility functions."""
import pickle
from functools import lru_cache
from pkg_resources import resource_stream
import numpy as np
from .ctmdata import CTM_DATASETS as _ctm_datasets, __name__ as _ctmdata_path


def get_reduced_shape(X, shape, shift=0, size_only=True):
    """Get shape of a reduced dataset.

    The notion of reduced dataset shape is central for the core partition algorithm.
    A reduced representation of dataset shape is a tuple of integers
    indicating how many units along each dimension would a dataset have
    if it was sliced into pieces of a given shape and each slice would
    be considered an entry in a ``N``-dimensional array
    (where ``N`` is the number of axes in the original dataset).

    More on reduced shapes
    ----------------------

    Reduced representation makes it ease to achieve several important goals:

    #. Compute the total number of slices. This is needed to represent
       ``N`` nested for-loops going over all axes of a dataset as a single
       flat for-loop. Loop flattenning allows the partition algorithm
       to operate over arrays with arbitrary numbers of axes.
       It also makes it easily paralelizable.
    #. Position in the reduced representation can be easily
       back-transformed to positions in the original dataset.
       This makes it possible to loop over a reduced representation
       while slicing out pieces of the original dataset.

    Here is an example of a reduced shape representation::

        # Consider a 2D array and a slice shape (2, 2)
        x x x x
        x x x x
        x x x x
        x x x x
        # Then the array is reduced to:
        x x
        x x
        # So the reduced shape is (2, 2) instead of (4, 4)

    Parameters
    ----------
    X : array_like
        Dataset of arbitrary dimensionality represented as a *Numpy* array.
    shape : tuple
        Shape of the dataset's parts. Has to be symmetric.
    shift : int
        Shift of the sliding window.
        In general, if positive, should not be greater than ``1``.
        Shift by partition shape if not positive.
    size_only : bool
        Should only the 1D length of the reduced dataset be returned.
        1D length is the total number of dataset parts.

    Returns
    -------
    tuple
        Shape tuple if ``size_only=False``.
    int
        Number of parts if ``size_only=True``.

    Raises
    ------
    AttributeError
        If parts' `shape` is not equal in each dimension.
        If parts' `shape` and dataset's shape have different numbers of axes.

    Examples
    --------
    >>> x = np.ones((5, 5))
    >>> get_reduced_shape(x, (2, 2), size_only=False)
    (3, 3)
    >>> get_reduced_shape(x, (2, 2), size_only=True)
    9
    """
    if len(set(shape)) != 1:
        raise AttributeError(f"Partition shape is not symmetric {shape}")
    if len(shape) != X.ndim:
        X = X.squeeze()
        if len(shape) != X.ndim:
            raise AttributeError("Dataset and parts have different numbers of axes")
    if shift <= 0:
        r_shape = tuple(int(np.ceil(x / p)) for x, p in zip(X.shape, shape))
    else:
        r_shape = tuple(int(x-p+1) for x, p in zip(X.shape, shape))
    if size_only:
        return int(np.multiply.reduce(r_shape))
    return r_shape

def get_reduced_idx(i, shape):
    """Get index of a part in a reduced representation from a part's number.

    See Also
    --------
    :py:func:`bdm.utils.get_reduced_shape`

    Parameters
    ----------
    i : int
        Part number.
    shape : tuple
        Shape of a reduced dataset.

    Returns
    -------
    tuple
        Index of a part in a reduced dataset.

    Examples
    --------
    >>> get_reduced_idx(5, (2, 2, 2))
    (1, 0, 1)
    >>> get_reduced_idx(2, (1, 4))
    (0, 2)
    """
    if i >= int(np.multiply.reduce(shape)):
        raise IndexError("'i' is beyond the provided shape")
    elif i < 0:
        raise IndexError("'i' has to be non-zero")
    K = len(shape)
    r_idx = tuple(
        (i % int(np.multiply.reduce(shape[k:K]))) //
        int(np.multiply.reduce(shape[(k+1):K]))
        for k in range(K)
    )
    return r_idx

def slice_dataset(X, shape, shift=0):
    """Slice a dataset into *n* pieces.

    Slicing is done in a way that ensures that only pieces
    on boundaries of the sliced dataset can have leftovers
    in regard to a specified shape.
    This is very important for proper computing of BDM in the context
    of parallel processing.

    Parameters
    ----------
    X : array_like
        Daataset represented as a *Numpy* array.
    shape : tuple
        Slice shape.
    shift : int
        Shift value for slicing.
        Nonoverlaping slicing if non-positive.

    Yields
    ------
    array_like
        Dataset slices.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((5, 3), dtype=int)
    >>> [ x for x in slice_dataset(X, (3, 3)) ]
    [array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]]), array([[1, 1, 1],
           [1, 1, 1]])]
    """
    if len(shape) != X.ndim:
        raise ArithmeticError(
            "dataset and slice shape does not have the same number of axes"
        )
    r_shape = get_reduced_shape(X, shape, size_only=False)
    n_parts = int(np.multiply.reduce(r_shape))
    width = shape[0]
    slice_shift = shift if shift > 0 else width
    for i in range(n_parts):
        r_idx = get_reduced_idx(i, r_shape)
        if shift <= 0:
            idx = tuple(slice(k*width, k*width + slice_shift) for k in r_idx)
        else:
            idx = tuple(slice(k, k + width) for k in r_idx)
        yield X[idx]

def list_ctm_datasets():
    """Get a list of available precomputed CTM datasets.

    Examples
    --------
    >>> list_ctm_datasets()
    ['CTM-B2-D12', 'CTM-B2-D4x4']
    """
    return [ x for x in sorted(_ctm_datasets.keys()) ]

@lru_cache(maxsize=2**int(np.ceil(np.log2(len(_ctm_datasets)))))
def get_ctm_dataset(name):
    """Get CTM dataset by name.

    This function uses a global cache, so each CTM dataset
    is loaded to the memory only once.

    Parameters
    ----------
    name : str
        Name of a dataset.

    Returns
    -------
    dict
        CTM lookup table.

    Raises
    ------
    ValueError
        If non-existent CTM dataset is requested.
    """
    if name not in _ctm_datasets:
        raise ValueError(f"There is no {name} CTM dataset")
    with resource_stream(_ctmdata_path, _ctm_datasets[name]) as stream:
        return pickle.load(stream)
