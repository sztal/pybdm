"""Core algorithms operating on ``BDM`` objects."""
from itertools import product
from numpy.random import choice
from bdm.utils import get_reduced_shape


class PerturbationExperiment:
    """Perturbation experiment class.

    Perturbation experiment studies change of BDM under changes
    applied to the underlying dataset.

    Attributes
    ----------
    X : array_like
        Dataset for perturbation analysis.
    bdm : BDMBase
        BDM object.
    counter : Counter
        Counter of BDM slices.
    """
    def __init__(self, X, bdm, counter=None):
        """Initialization method."""
        self.X = X
        self.bdm = bdm
        self.counter = counter if counter else bdm.count_and_lookup(X)
        self._r_shape = \
            get_reduced_shape(X, bdm.shape, shift=bdm.shift, size_only=False)

    @property
    def size(self):
        """Data size getter."""
        return self.X.size

    @property
    def shape(self):
        """Data shape getter."""
        return self.X.shape

    @property
    def ndim(self):
        """Data number of axes getter."""
        return self.X.ndim

    def _idx_to_slices(self, idx):
        def _slice(i, k):
            start = i - i % k
            end = start + k
            return slice(start, end)
        shift = self.bdm.shift
        if shift <= 0:
            s = tuple(_slice(i, k) for i, k in zip(idx, self.bdm.shape))
            yield s
            return
        idxs = tuple(range(max(i-k+1, 0), i+1) for i, k in zip(idx, self.bdm.shape))
        for i in product(*idxs):
            s = tuple(slice(m, m+n) for m, n in zip(i, self.bdm.shape))
            yield s

    def update(self, idx, value=None):
        """Update element of the dataset.

        Parameters
        ----------
        idx : tuple
            Element index tuple.
        value : int or None
            Value to assign.
            If ``None`` then new value is randomly selected from the set
            of other possible values.
            For binary data this is just a bit flip and no random numbers
            generation is involved in the process.
        """
        if value is None:
            if self.bdm.n_symbols <= 2:
                v = self.X[idx]
                self.X[idx] = 1 if v == 0 else 0
            else:
                v = self.X[idx]
                self.X[idx] = choice([ x for x in range(self.n_symbols) if x != v ])
        else:
            self.X[idx] = value

    def perturb(self, idx, *args, dry_run=True):
        """Perturb element of the dataset.

        Parameters
        ----------
        idx : tuple
            Index tuple of an element.
        updater : int or callable or None
            Value to switch to.
            If callable then it is called on the element's value
            to determine the new value.
            If ``None`` then the updater stored in the object attribute is used.
        *args :
            Positional arguments passed to
            :py:meth:`PerturbationExperiment.update`.
        dry_run : bool
            If ``True`` then change is not persisted.

        Returns
        -------
        float :
            BDM value change.
        """
        pass
