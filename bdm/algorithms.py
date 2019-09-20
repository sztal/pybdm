"""Core algorithms operating on ``BDM`` objects."""
from itertools import product
from random import choice
import numpy as np


class PerturbationExperiment:
    """Perturbation experiment class.

    Perturbation experiment studies change of BDM / entropy under changes
    applied to the underlying dataset. This is the main tool for detecting
    parts of a system having some causal significance as opposed
    to noise parts.

    Parts which after yield negative contribution to the overall
    complexity after change are likely to be important for the system,
    since their removal make it more noisy. On the other hand parts that yield
    positive contribution to the overall complexity after change are likely
    to be noise since they extend the system's description length.

    Attributes
    ----------
    X : array_like
        Dataset for perturbation analysis.
    bdm : BDMBase
        BDM object.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.
    """
    def __init__(self, bdm, X=None, metric='bdm'):
        """Initialization method."""
        self.bdm = bdm
        self.metric = metric
        self._counter = None
        self._value = None
        self._ncounts = None
        if self.metric == 'bdm':
            self._method = self._update_bdm
        elif self.metric == 'ent':
            self._method = self._update_ent
        else:
            raise AttributeError("Incorrect metric, not one of: 'bdm', 'ent'")
        if X is None:
            self.X = X
        else:
            self.set_data(X)

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

    def set_data(self, X):
        """Set dataset.

        Parameters
        ----------
        X : array_like
            Dataset to perturb.
        """
        self.X = X
        self._counter = self.bdm.lookup_and_count(X)
        if self.metric == 'bdm':
            self._value = self.bdm.compute_bdm(self._counter)
        elif self.metric == 'ent':
            self._value = self.bdm.compute_ent(self._counter)
            self._ncounts = sum(self._counter.values())

    def _idx_to_parts(self, idx):
        def _slice(i, k):
            start = i - i % k
            end = start + k
            return slice(start, end)
        try:
            shift = self.bdm.partition.shift
        except AttributeError:
            shift = 0
        shape = self.bdm.partition.shape
        if shift == 0:
            r_idx = tuple((k // l)*l for k, l in zip(idx, shape))
            idx = tuple(slice(k, k+l) for k, l in zip(r_idx, shape))
        else:
            idx = tuple(slice(max(0, k-l+1), k+l) for k, l in zip(idx, shape))
        yield from self.bdm.decompose(self.X[idx])

    def _update_bdm(self, idx, old_value, new_value, keep_changes):
        old_bdm = self._value
        new_bdm = self._value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter[(key, cmx)]
            if n > 1:
                new_bdm += np.log2((n-1) / n)
                if keep_changes:
                    self._counter[(key, cmx)] -= 1
            else:
                new_bdm -= cmx
                if keep_changes:
                    del self._counter[(key, cmx)]
        self.X[idx] = new_value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter.get((key, cmx), 0)
            if n > 0:
                new_bdm += np.log2((n+1) / n)
            else:
                new_bdm += cmx
            if keep_changes:
                self._counter.update([(key, cmx)])
        if not keep_changes:
            self.X[idx] = old_value
        else:
            self._value = new_bdm
        return new_bdm - old_bdm

    def _update_ent(self, idx, old_value, new_value, keep_changes):
        old_ent = self._value
        new_ent = self._value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter[(key, cmx)]
            p = n / self._ncounts
            new_ent += p*np.log2(p)
            if n > 1:
                p = (n-1) / self._ncounts
                new_ent -= p*np.log2(p)
                if keep_changes:
                    self._counter[(key, cmx)] -= 1
            elif keep_changes:
                del self._counter[(key, cmx)]
        self.X[idx] = new_value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter.get((key, cmx), 0) + 1
            p = n / self._ncounts
            new_ent -= p*np.log2(p)
            if n > 1:
                p = (n-1) / self._ncounts
                new_ent += p*np.log2(p)
            if keep_changes:
                self._counter.update([(key, cmx)])
        if not keep_changes:
            self.X[idx] = old_value
        else:
            self._value = new_ent
        return new_ent - old_ent

    def perturb(self, idx, value=-1, keep_changes=False):
        """Perturb element of the dataset.

        Parameters
        ----------
        idx : tuple
            Index tuple of an element.
        value : int or callable or None
            Value to assign.
            If negative then new value is randomly selected from the set
            of other possible values.
            For binary data this is just a bit flip and no random numbers
            generation is involved in the process.
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        float :
            BDM value change.

        Examples
        --------
        >>> from bdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> perturbation.perturb((10, ), -1) # doctest: +FLOAT_CMP
        26.91763012739709
        """
        old_value = self.X[idx]
        if value < 0:
            if self.bdm.nsymbols <= 2:
                value = 1 if old_value == 0 else 0
            else:
                value = choice([
                    x for x in range(self.bdm.nsymbols)
                    if x != old_value
                ])
        if old_value == value:
            return 0
        return self._method(idx, old_value, value, keep_changes)

    def run(self, idx=None, values=None, keep_changes=False):
        """Run perturbation experiment.

        Parameters
        ----------
        idx : array_like or None
            *Numpy* integer array providing indexes (in rows) of elements
            to perturb. If ``None`` then all elements are perturbed.
        values : array_like or None
            Value to assign during perturbation.
            Negative values correspond to changing value to other
            randomly selected symbols from the alphabet.
            If ``None`` then all values are assigned this way.
            If set then its dimensions must agree with the dimensions
            of ``idx`` (they are horizontally stacked).
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        array_like
            1D float array with perturbation values.

        Examples
        --------
        >>> from bdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> changes = np.array([10, 20])
        >>> perturbation.run(changes) # doctest: +FLOAT_CMP
        array([26.91763013, 27.34823681])
        """
        if idx is None:
            indexes = [ range(k) for k in self.X.shape ]
            idx = np.array([ x for x in product(*indexes) ], dtype=int)
        if values is None:
            values = np.full((idx.shape[0], ), -1, dtype=int)
        return np.apply_along_axis(
            lambda r: self.perturb(tuple(r[:-1]), r[-1], keep_changes=keep_changes),
            axis=1,
            arr=np.column_stack((idx, values))
        )
