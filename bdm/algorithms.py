"""Core algorithms operating on ``BDM`` objects."""
import numpy as np
from .utils import get_reduced_idx


class PerturbationExperiment:
    """Perturbation experiment class.

    Perturbation experiment studies change of BDM under changes
    applied to the underlying dataset.

    Attributes
    ----------
    bdm : BDMBase
        BDM object.
    data : array_like
        Dataset for perturbation analysis.
    counter : Counter
        Counter of BDM slices.
    **kwds :
        Keyword arguments passed to
        :py:meth:`bdm.base.BDMBase.count_and_lookup`
        if `counter` is ``None``.
    """
    def __init__(self, bdm, data, counter=None, **kwds):
        """Initialization method."""
        self.bdm = bdm
        self.data = data
        self.counter = counter if counter else bdm.count_and_lookup(data, **kwds)

    @property
    def dsize(self):
        """Data size getter."""
        return self.data.size

    @property
    def dshape(self):
        """Data shape getter."""
        return self.data.shape

    @property
    def dndim(self):
        """Data number of axes getter."""
        return self.data.ndim

    def num_to_idx(self, i):
        """Convert slice number to indexes.

        Parameters
        ----------
        i : int
            Slice number ranging from 0 to `self.dsize - 1`.
        """
        return get_reduced_idx(i, self.dshape)

    def idx_to_num(self, idx):
        """Convert indexes to a slice number.

        Parameters
        ----------
        idx : tuple
            Indexes of an entry.
        """
        mods = tuple(
            int(np.multiply.reduce(self.dshape[i:self.dndim-1]))
            for i in range(self.dndim)
        )
        return sum(x*y for x, y in zip(idx, mods))
