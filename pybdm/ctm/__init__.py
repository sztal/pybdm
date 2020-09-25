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
from functools import lru_cache, cached_property
from pkg_resources import resource_stream
import numpy as np
import pandas as pd
from scipy.special import factorial
from ..encoding import decode_sequences
from ..utils import count_ctm_classes, n_distinct


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
    # pylint: disable=too-many-locals
    def err(ndim, nsymbols, nstates):
        msg = "CTM dataset with {n} states, {d} dimensions and {s} symbols not found" \
            .format(n=nstates, d=ndim, s=nsymbols)
        return LookupError(msg)
    try:
        if nstates is not None:
            datasets = CTM_DATASETS[nstates]
        else:
            nstates = list(sorted(list(CTM_DATASETS.keys()), reverse=True))
            datasets = ChainMap(*(CTM_DATASETS[n] for n in nstates))
    except:
        if not isinstance(nstates, int):
            nstates = 'any number of'
        raise err(ndim, nsymbols, nstates)

    try:
        name = datasets[(ndim, nsymbols)]
    except KeyError:
        raise err(ndim, nsymbols, nstates)

    nstates = int(name.split('-')[1][1:])
    filepath = _name_to_filepath(name)

    with resource_stream(_HERE, filepath) as stream:
        data = dict(pickle.loads(gzip.decompress(stream.read())))

    missing = {}

    for shape in data:
        codes, cmx = data[shape]
        codes = codes.astype(INT_DTYPE, copy=False)
        cmx = cmx.astype(FLOAT_DTYPE, copy=False)
        order = np.argsort(-cmx)

        cmx = cmx[order]
        codes = codes[order]

        uniq = n_distinct(
            decode_sequences(codes, shape=shape, base=nsymbols),
            axis=1
        )
        cardinality = factorial(nsymbols) / factorial(nsymbols - uniq)

        for arr in (codes, cmx, cardinality):
            arr.flags.writeable = False

        df = pd.DataFrame({
            'cmx': cmx,
            'cardinality': np.round(cardinality).astype(int)
        }, index=codes)

        missing[shape] = cmx.max() + 1
        data[shape] = df

    return (
        MappingProxyType(data),
        MappingProxyType(missing),
        ndim,
        nsymbols,
        nstates
    )


class CTMStore:
    """Store with precomputed CTM values for blocks of given shapes.

    Attributes
    ----------
    data : mappingproxy
        CTM data lookup table mapping shape to
        :py:class:`pandas.Series` objects with
        estimated CTM complexities.
    missing : mappingproxy
        Map from shapes to CTM imputation values
        for handling missing equivalence classes.
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
    def __init__(self, data, missing, ndim, nsymbols, nstates):
        self.data = data
        self.missing = missing
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

    def __getitem__(self, shape):
        return self.data[shape].cmx

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

    @cached_property
    def info(self):
        """Basic information on the CTM dataset.

        The `info` report consists of the follwing fields:

        equiv_enum
            Number of enumerated equivalence classes.

        equiv_total
            Number of total existing equivalence classes.

        equiv_cov
            Fraction of enumerated equivalence classes.

        blocks_enum
            Number of unique blocks covered by the enumerated e
            quivalence classes.

        blocks_total
            Number of total unique blocks.

        blocks_cov
            Fraction of unique blocks covered by the enumerated equivalence
            classes.
            `
        Returns
        -------
        mappingproxy
            CTM report.
        """
        report = {}
        for shape in self.data:
            width = prod(shape)
            equiv_enum = self[shape].size
            equiv_total = count_ctm_classes(width, k=self.nsymbols)
            equiv_cov = equiv_enum / equiv_total
            blocks_enum = self.data[shape].cardinality.sum()
            blocks_total = self.nsymbols**width
            blocks_cov = blocks_enum / blocks_total
            report[shape] = MappingProxyType(dict(
                equiv_enum=equiv_enum,
                equiv_total=equiv_total,
                equiv_cov=equiv_cov,
                blocks_enum=blocks_enum,
                blocks_total=blocks_total,
                blocks_cov=blocks_cov
            ))
        return MappingProxyType(report)

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_params(cls, ndim, nsymbols, nstates=None):
        """Initialize from main parameters."""
        data, missing, ndim, nsymbols, nstates = \
            _load_ctm_store(ndim, nsymbols, nstates)
        return cls(data, missing, ndim, nsymbols, nstates)

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
        return self[shape].reindex(codes).fillna(self.missing[shape])
