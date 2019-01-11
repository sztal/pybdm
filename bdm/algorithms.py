"""Core algorithms operating on ``BDM`` objects."""
import numpy as np
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
    metric : {'bdm', 'entropy'}
        Which metric to use for perturbing.
    """
    def __init__(self, X, bdm, metric='bdm'):
        """Initialization method."""
        self.X = X
        self.bdm = bdm
        self.metric = metric
        self._counter = bdm.count_and_lookup(X)
        self._ncounts = None
        if self.metric == 'bdm':
            self._value = self.bdm.compute_bdm(self._counter)
            self._method = self._update_bdm
        elif self.metric == 'entropy':
            self._value = self.bdm.compute_entropy(self._counter)
            self._method = self._update_entropy
            self._ncounts = sum(self._counter.values())
        else:
            raise AttributeError("Incorrect metric, not one of: 'bdm', 'entropy'")
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

    def _idx_to_parts(self, idx):
        def _slice(i, k):
            start = i - i % k
            end = start + k
            return slice(start, end)
        shift = self.bdm.shift
        shape = self.bdm.shape
        if shift == 0:
            r_idx = tuple((k // l)*l for k, l in zip(idx, shape))
            idx = tuple(slice(k, k+l) for k, l in zip(r_idx, shape))
        else:
            idx = tuple(slice(max(0, k-l+1), k+l) for k, l in zip(idx, shape))
        yield from self.bdm.partition(self.X[idx])

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

    def _update_entropy(self, idx, old_value, new_value, keep_changes):
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
        >>> perturbation = PerturbationExperiment(X, bdm)
        >>> perturbation.perturb((10, ), -1) # doctest: +FLOAT_CMP
        26.91763012739709
        """
        old_value = self.X[idx]
        if value < 0:
            if self.bdm.n_symbols <= 2:
                value = 1 if old_value == 0 else 0
            else:
                value = choice([
                    x for x in range(self.bdm.n_symbols)
                    if x != old_value
                ])
        if old_value == value:
            return 0
        return self._method(idx, old_value, value, keep_changes)

    def run(self, changes, keep_changes=False):
        """Run perturbation experiment.

        Parameters
        ----------
        changes : array_like
            Integer *Numpy* array.
            First ``k`` columns must provide indices of elements
            to change and the last column new element values
            for :py:meth:`bdm.algorithms.PerturbationExperiment.perturb`.
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
        >>> perturbation = PerturbationExperiment(X, bdm)
        >>> changes = np.array([[10, -1], [20, -1]])
        >>> perturbation.run(changes) # doctest: +FLOAT_CMP
        array([26.91763013, 27.34823681])
        """
        return np.apply_along_axis(
            lambda r: self.perturb(tuple(r[:-1]), r[-1], keep_changes=keep_changes),
            axis=1,
            arr=changes
        )
