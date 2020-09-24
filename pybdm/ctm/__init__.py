"""CTM datasets and lookup table objects.

CTM datasets store precomputed algorithmic complexity values for
simple objects estimated based on *Coding Theorem Method*.
(see :doc:`theory`).

All datasets' names use the following naming scheme: ``CTM-S{X}-B{Y}-D{Z}``,
where ``X``, ``Y`` and ``Z`` stand for numbers of states, symbols
and dimensions.

Datasets
--------
:``CTM-S5-B2-D1``:
    Binary strings of length from 1 to 12.
    Based on an enumeration of TMs with 5 states.
:``CTM-S4-B4-D1``:
    4-symbols strings of length from 1 to 12.
    Based on an enumeration of TMs with 4 states.
:``CTM-S4-B5-D1``:
    5-symbols strings of length from 1 to 12.
    Based on an enumeration of TMs with 4 states.
:``CTM-S4-B6-D1``:
    6-symbols strings of length from 1 to 12.
    Based on an enumeration of TMs with 4 states.
:``CTM-S4-B9-D1``:
    9-symbols strings of length from 1 to 12.
    Based on an enumeration of TMs with 4 states.
:``CTM-S5-B2-D2``:
    Square binary matrices of width from 1 to 4.
    Based on an enumeration of 2D TMs (Turmities)
    with 5 states.
"""
import os
import gzip
import pickle
from math import ceil, log2, prod
from collections import OrderedDict, ChainMap
from types import MappingProxyType
from functools import lru_cache
from pkg_resources import resource_stream
import numpy as np
import pandas as pd
from ..utils import count_ctm_classes


_HERE = __name__

CTM_DATASETS = MappingProxyType({
    4: MappingProxyType({
        (1, 4): 'CTM-S4-B4-D1',
        (1, 5): 'CTM-S4-B5-D1',
        (1, 6): 'CTM-S4-B6-D1',
        (1, 9): 'CTM-S4-B9-D1'
    }),
    5: MappingProxyType({
        (1, 2): 'CTM-S5-B2-D1',
        (2, 2): 'CTM-S5-B2-D2'
    })
})
CTM_PICKLE_PROTOCOL = 4

INT_DTYPE = np.int64
FLOAT_DTYPE = np.float64

# -----------------------------------------------------------------------------

def _name_to_filepath(name):
    return name.lower()+'.pkl.gz'

@lru_cache(maxsize=2**ceil(log2(len(CTM_DATASETS))))
def _load_ctm_store(ndim, nsymbols, nstates=None):
    try:
        if nstates is not None:
            datasets = CTM_DATASETS[nstates]
        else:
            nstates = list(sorted(list(CTM_DATASETS.keys()), reverse=True))
            datasets = ChainMap(*(CTM_DATASETS[n] for n in nstates))
    except:
        if not isinstance(nstates, int):
            nstates = 'any number of'
        msg = "CTM dataset with {n} states, {d} dimensions and {s} symbols not found" \
            .format(n=nstates, d=ndim, s=nsymbols)
        raise LookupError(msg)

    name = datasets[(ndim, nsymbols)]
    nstates = int(name.split('-')[1][1:])
    filepath = _name_to_filepath(name)

    with resource_stream(_HERE, filepath) as stream:
        data = dict(pickle.loads(gzip.decompress(stream.read())))

    for shape in data:
        codes, cmx = data[shape]
        codes = codes.astype(INT_DTYPE, copy=False)
        cmx = cmx.astype(FLOAT_DTYPE, copy=False)
        order = np.argsort(-cmx)
        cmx = cmx[order]
        cmx.flags.writeable = False
        codes = codes[order]
        codes.flags.writeable = False
        ctm = pd.Series(cmx, index=codes)
        ctm.missing = cmx.max() + 1

        data[shape] = ctm

    return MappingProxyType(data), ndim, nsymbols, nstates


class CTMStore:
    """Store with precomputed CTM values for blocks of given shapes.

    Attributes
    ----------
    data : mappingproxy
        CTM data lookup table mapping shape to
        :py:class:`pandas.Series` objects with
        estimated CTM complexities.
    ndim : int, positive
        Number of dimensions.
    nsymbols : int, positive
        Number of symbols of explored Turing machines.
    nstates : int, positive
        Number of states of explored Turing machines.
        The more states the better approximation.
    name : str
        Dataset name.
    filepath : str
        Path to the data file.
    shapes : list of tuple of int
        List of all supported block shapes.
    coverage : dict
        Map from supported shapes to fractions of blocks
        for which CTM values are explicitly calculated.

    Notes
    -----
    CTM values may not be explicitly calculated for some objects.
    In such cases missing values are substituted with the maximum
    stored CTM value plus ``1`` bit.
    """
    def __init__(self, data, ndim, nsymbols, nstates):
        self.data = data
        self.ndim = ndim
        self.nsymbols = nsymbols
        self.nstates = nstates

    def __repr__(self):
        return "<{cn} ({d}D) with {s} states an {b} symbols at {id}>".format(
            cn=self.__class__.__name__,
            d=self.ndim,
            b=self.nsymbols,
            s=self.nstates,
            id=hex(id(self))
        )

    @property
    def name(self):
        return "CTM-S{nstates}-B{nsymbols}-D{ndim}".format(
            nstates=self.nstates,
            nsymbols=self.nsymbols,
            ndim=self.ndim
        )

    @property
    def filepath(self):
        return os.path.join(_HERE, _name_to_filepath(self.name))

    @property
    def shapes(self):
        return list(self.data.keys())

    @property
    def coverage(self):
        return {
            shape: len(self.data[shape]['data']) / \
                count_ctm_classes(prod(shape), self.nsymbols)
            for shape in self.data
        }

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_params(cls, ndim, nsymbols, nstates=None):
        """Initialize from main parameters."""
        data, ndim, nsymbols, nstates = _load_ctm_store(ndim, nsymbols, nstates)
        return cls(data, ndim, nsymbols, nstates)

    # -------------------------------------------------------------------------

    def get(self, shape, codes):
        """Get CTM values.

        Parameters
        ----------
        shape : tuple of int
            Block shape.
        codes : 1D integer array
            Array of block codes.
        """
        data = self.data[shape]
        return data.reindex(codes).fillna(data.missing)
