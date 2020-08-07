"""Utility functions."""
import gzip
import pickle
from collections import OrderedDict
from itertools import product
from functools import lru_cache
from pkg_resources import resource_stream
import numpy as np
from .ctmdata import CTM_DATASETS as _ctm_datasets, __name__ as _ctmdata_path


def prod(seq):
    # pylint: disable=anomalous-backslash-in-string
    """Product of a sequence of numbers.

    Parameters
    ----------
    seq : sequence
        A sequence of numbers.

    Returns
    -------
    float or int
        Product of numbers.

    Notes
    -----
    This is defined as:

    .. math::

        \prod_{i=1}^n x_i
    """
    mult = 1
    for x in seq:
        mult *= x
    return mult

def iter_slices(X, shape, shift=0):
    """Iter over slice indices of a dataset.

    Slicing is done in a way that ensures that only pieces
    on boundaries of the sliced dataset can have leftovers
    with respect to a specified shape.

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
    slice
        Slice indices.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((5, 3), dtype=int)
    >>> [ x for x in iter_slices(X, (3, 3)) ]
    [(slice(0, 3, None), slice(0, 3, None)), (slice(3, 5, None), slice(0, 3, None))]
    """
    if len(set(shape)) != 1:
        raise AttributeError("Partition shape is not symmetric {}".format(shape))
    if len(shape) != X.ndim:
        raise AttributeError(
            "dataset and slice shape does not have the same number of axes"
        )

    if shift <= 0:
        shift = shape[0]
        data_shape = X.shape
    else:
        data_shape = tuple(max(x - s + 1, 0) for x, s in zip(X.shape, shape))

    start_idx = product(*(range(0, k, shift) for k in data_shape))
    for start in start_idx:
        yield tuple(
            slice(s, min(s + w, t)) for s, w, t in zip(start, shape, X.shape)
        )

def iter_part_shapes(X, shape, shift=0):
    """Iterate over part shapes induced by slicing.

    Parameters
    ----------
    X : array_like
        Dataset represented as a *Numpy* array.
    shape : tuple
        Slice shape.
    shift : int
        Shift value for slicing.
        Nonoverlaping slicing if non-positive.

    Yields
    ------
    tuple
        Part shapes.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((5, 3), dtype=int)
    >>> [ x for x in iter_part_shapes(X, (3, 3)) ]
    [(3, 3), (2, 3)]
    """
    for idx in iter_slices(X, shape=shape, shift=shift):
        part = tuple(s.stop - s.start for s in idx)
        yield part

def decompose_dataset(X, shape, shift=0):
    """Decompose a dataset into blocks.

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
        Dataset blocks.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((5, 3), dtype=int)
    >>> [ x for x in decompose_dataset(X, (3, 3)) ]
    [array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]]), array([[1, 1, 1],
           [1, 1, 1]])]
    """
    for idx in iter_slices(X, shape=shape, shift=shift):
        yield X[idx]

def list_ctm_datasets():
    """Get a list of available precomputed CTM datasets.

    Examples
    --------
    >>> list_ctm_datasets()
    ['CTM-B2-D12', 'CTM-B2-D4x4', 'CTM-B4-D12', 'CTM-B5-D12', 'CTM-B6-D12', 'CTM-B9-D12']
    """
    return list(sorted(_ctm_datasets.keys()))

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
        raise ValueError("There is no {} CTM dataset".format(name))
    with resource_stream(_ctmdata_path, _ctm_datasets[name]) as stream:
        dct = pickle.loads(gzip.decompress(stream.read()))
    for key in dct:
        o = dct[key]
        dct[key] = OrderedDict(sorted(o.items(), key=lambda x: x[1], reverse=True))
    missing = {}
    for sh, cmx in dct.items():
        missing[sh] = np.max(list(cmx.values())) + 1
    return dct, missing
