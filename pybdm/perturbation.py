"""Algorithms based on ``BDM`` objects."""
from collections.abc import Mapping, Sequence
from itertools import cycle
from functools import cached_property
import numpy as np
from tqdm import tqdm
from .blocks import get_block_idx, get_block


class PerturbationStep:
    """Perturbation step.

    Perturbation experiments are defined in steps each of which
    can be either a single update or a batch of multiple updates.
    Complexity changes are recorded for each step so both
    atomic and compound changes can be studied.

    Multiple sequential steps can be defined via single
    `Step` object with compund `idx` and `newvals` by
    setting `batch=False`.

    Attributes
    ----------
    idx : tuple, list of tuples, 2D ndarray, callable or None
        Indexes of elements to be perturbed.
        A single `tuple` specifies one element.
        Two-dimensional :py:class:`numpy.ndarray` objects
        should define one element index per row.
        Alternatively they can be specified as a boolean
        mask which will be internally converted into
        integer arrays with indexes.
        Both arrays and sequences of tuples are interpreted
        by default as batch perturbations.
        If `callable` then it is called on the dataset
        (:py:class:`Step` objects has to be bound)
        to generate indices (in a form of tuples of an array).
        If ``None`` then all elements are perturbed.
    newvals : int, sequence of ints, mapping, callable or None
        Single integers are applied to all elements.
        Sequences must match length of `idx` and define new values
        for individual elements. Mappings and callables are used
        to define custom rules transforming current values
        of elements given by `idx` to new values.
        ``None`` means that new values are selected at random
        from the set of possible values (in a given alphabet)
        different from the current value.
    vidx : ndarray
        The same as `idx` but represented in a way suitable
        for indexing other ndarrays to extract individual
        elements.
    batch : bool
       Should step be treated as batch update.
       If ``False`` then even if the step defines multiple
       updates at once they will be performed sequentially
       as separate steps. If ``None`` then perturbation context
       settings are used.
    ctx : Perturbation, optional
        Context in which a step is performed.
        May be changed with :py:meth:`bind` method.
    """
    def __init__(self, idx=None, newvals=None, batch=None, ctx=None):
        self._idx = idx
        self._newvals = newvals
        self.batch = batch
        self.ctx = ctx

    def __repr__(self):
        return "{cn}(idx={idx}, newvals={newvals}, batch={batch})".format(
            cn=self.__class__.__name__,
            idx=self._idx,
            newvals=self._newvals,
            batch=self.batch
        )

    @cached_property
    def idx(self):
        if callable(self._idx):
            if self.ctx is None:
                raise AttributeError("'ctx' has to be defined for callable 'idx'")
            _idx = self._idx(self.ctx.X)
        else:
            _idx = self._idx
        if _idx is None:
            _idx = np.argwhere(np.ones_like(self.ctx.X, dtype=bool))
        elif isinstance(_idx, np.ndarray) and _idx.dtype.type is np.bool_:
            _idx = np.argwhere(_idx)
        _idx = np.array(_idx)
        if isinstance(_idx, np.ndarray) and _idx.ndim == 1:
            _idx = _idx.reshape(1, -1)
        return _idx

    @property
    def vidx(self):
        idx = self.idx
        if idx.shape[0] == 1:
            idx = (idx.squeeze(),)
        else:
            idx = tuple(idx.T)
        return idx

    @cached_property
    def newvals(self):
        new = self._newvals
        if callable(new) or isinstance(new, Mapping) or new is None:
            values = self.ctx.X[self.vidx]
            new_values = np.empty_like(values)

            for k in range(self.ctx.nsymbols):
                mask = values == k

                if callable(new):
                    new_values[mask] = new(k)
                elif isinstance(new, Mapping):
                    new_values[mask] = new[k]
                else:
                    if self.ctx.nsymbols == 2:
                        new_values = np.where(values == 0, 1, 0)
                    else:
                        choices = np.array([
                            x for x in range(self.ctx.nsymbols)
                            if x != k
                        ])
                        new_values[mask] = \
                            np.random.choice(choices, size=mask.sum())
            return new_values
        return self._newvals

    def bind(self, ctx):
        """Bind to a perturbation experiment context.

        Parameters
        ----------
        ctx : Perturbation
            Perturbation experiment object.
        """
        self.ctx = ctx

    def to_sequence(self):
        newvals = self.newvals
        if newvals is None or np.isscalar(newvals):
            newvals = cycle(newvals)
        for i, v in zip(self.idx, newvals):
            yield self.__class__(
                idx=i,
                newvals=v,
                batch=True,
                ctx=self.ctx
            )

    def to_tuple(self):
        """Dump to tuple with indexes and new values."""
        return self.idx, self.newvals

    @classmethod
    def from_tuple(cls, tup):
        """Init from a tuple."""
        return cls(*tup)

    @classmethod
    def from_dict(cls, dct):
        """Init from a dict."""
        return cls(**dct)


class Perturbation:
    """Perturbation experiment class.

    Perturbation experiment studies change of BDM / entropy under changes
    applied to the underlying dataset. This is the main tool for detecting
    parts of a system having some causal significance as opposed
    to noise parts.

    Parts which when perturbed yield positive contribution to the overall
    complexity are likely to be important for the system, since their
    removal make it more noisy. On the other hand, parts with negative
    contribution to the complexity are likely to be noise as their
    removal drives the system towards shorter description length.

    Attributes
    ----------
    bdm : BDM
        BDM object to be used for estimating complexity.
    X : array_like (optional)
        Dataset for perturbation analysis. May be set later.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.
    keep_changes : bool
        Should changes be kept so perturbations are accumulated.
    batch : bool
       Should step be treated as batch update.
       If ``False`` then even if the step defines multiple
       updates at once they will be performed sequentially
       as separate steps.

    See also
    --------
    pybdm.bdm.BDM : BDM computations
    """
    def __init__(self, bdm, X=None, metric='bdm',
                 keep_changes=False, batch=True):
        self.bdm = bdm
        self.metric = metric
        self._counter = None
        self._cmx = None
        self._ncounts = None
        self.X = X
        self.keep_changes = keep_changes
        self.batch = batch

    def __repr__(self):
        cn = self.__class__.__name__
        bdm = str(self.bdm)[1:-1]
        return "<{}(metric={}) with {}>".format(cn, self.metric, bdm)

    # Properties --------------------------------------------------------------

    @property
    def method(self):
        return getattr(self, '_update_'+self.metric)

    @property
    def X(self):
        return self._X
    @X.setter
    def X(self, newval):
        if newval is not None:
            self.bdm.check_data(newval)
            self._counter = self.bdm.decompose_and_count(newval)
            if self.metric == 'bdm':
                self._cmx = self.bdm.calc_bdm(self._counter)
            elif self.metric == 'ent':
                self._cmx = self.bdm.calc_ent(self._counter)
                self._ncounts = \
                    sum(sum(x.values()) for x in self._counter.values())
        self._X = newval

    @property
    def shape(self):
        return self.bdm.shape

    @property
    def ndim(self):
        return self.bdm.ndim

    @property
    def nsymbols(self):
        return self.bdm.nsymbols

    @property
    def nstates(self):
        return self.bdm.nstates

    @property
    def options(self):
        return self.bdm.options

    # Methods -----------------------------------------------------------------

    def get_blocks(self, idx):
        """Get block from an index of its element.

        Parameters
        ----------
        idx : tuple of int
            Index of an element.

        Returns
        -------
        array_like
            Data block.

        Examples
        --------
        >>> import numpy as np
        >>> from pybdm import BDM, Perturbation
        >>> bdm = BDM(ndim=1)
        >>> X = np.arange(16)
        >>> P = Perturbation(bdm, X=X)
        >>> P.get_block((3,))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> P.get_block((14,))
        array([12, 13, 14, 15])
        """
        block_idx = get_block_idx(idx, shape=self.shape)
        if isinstance(block_idx, tuple):
            block = get_block(self.X, block_idx, shape=self.shape)
            if self.bdm.partition.block_predicate(block):
                yield block
        else:
            for i in block_idx:
                block = get_block(self.X, i, shape=self.shape)
                if self.bdm.partition.block_predicate(block):
                    yield block

    def _update_bdm(self, old, new):
        # Determine which CTM values to add
        add_idx = {
            k: v[:, 0] for k, v in
            new.keydiff(self._counter).items()
        }
        # Update global counter and calculate relative count changes
        change = new - old
        delta = np.array([
            1 + v / self._counter[shape][k]
            for shape, cnt in change.items()
            for k, v in cnt.items()
            if k in self._counter[shape]
        ])
        delta = delta[delta > 0]
        # Update global counter and find CTM values to remove
        self._counter.update(change)
        sub_idx = {
            k: v[:, 0] for k, v in
            change.keydiff(self._counter).items()
        }
        # Update current BDM value
        cmx_add = sum(
            self.bdm.ctm.get(shape, i).sum()
            for shape, i in add_idx.items()
        )
        cmx_sub = sum(
            self.bdm.ctm.get(shape, i).sum()
            for shape, i in sub_idx.items()
        )
        delta_cmx = cmx_add - cmx_sub + np.log2(delta).sum()
        return delta_cmx

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
            self._cmx = new_ent
        return new_ent - old_ent

    def make_step(self, step, keep_changes=None, batch=None, bind=True, **kwds):
        """Do perturbation step.

        See :py:meth:`run` for details.

        Returns
        -------
        float :
            Complexity after change.
        """
        if isinstance(step, Mapping):
            step = PerturbationStep.from_dict(step)
        elif isinstance(step, Sequence):
            step = PerturbationStep.from_tuple(step)
        if bind:
            step.bind(self)

        # Unwind sequential step
        if step.batch is False or (step.batch is None and batch is False):
            return [ self.make_step(
                step=s,
                keep_changes=keep_changes,
                bind=False,
                **kwds
            ) for s in step.to_sequence() ]

        # Perform single/batch step
        vidx = step.vidx
        newvals = step.newvals
        oldvals = self.X[vidx]
        old_cmx = self._cmx

        # Count current blocks (before applying changes)
        blocks = list(self.get_blocks(step.idx))
        old = self.bdm.count_blocks(blocks, **kwds)
        # Apply changes and count modified blocks
        self.X[vidx] = newvals
        new = self.bdm.count_blocks(blocks, **kwds)

        delta_cmx = self.method(old, new)
        new_cmx = old_cmx + delta_cmx

        if keep_changes is None:
            keep_changes = self.keep_changes
        if keep_changes:
            self._cmx = new_cmx
        else:
            self.X[vidx] = oldvals

        return new_cmx

    def run(self, *steps, keep_changes=None, batch=None,
            bind=True, verbose=True, **kwds):
        """Run perturbation experiment.

        Parameters
        ----------
        *steps :
            Perturbation steps defined as
            :py:class:`PerturbationStep` objects.
        keep_changes : bool
            Should changes be kept so perturbations are accumulated.
            Use instance attribute if ``None``.
        batch : bool
            Should step be treated as batch update.
            If ``False`` then even if the step defines multiple
            updates at once they will be performed sequentially
            as separate steps. Instance attribute is used if ``None``.
        bind : True
            Should steps be bound to `self`.
            Usually should be ``True``.
        **kwds :
            Passed to :py:meth:`pybdm.bdm.BDM.count_blocks`.

        Returns
        -------
        array_like
            1D float array with perturbation values.
        """
        cmx = [ self._cmx ]
        if keep_changes is None:
            keep_changes = self.keep_changes
        if batch is None:
            batch = self.batch

        for step in tqdm(steps, disable=not verbose):
            _cmx = self.make_step(
                step=step,
                keep_changes=keep_changes,
                batch=batch,
                bind=bind,
                **kwds
            )
            if isinstance(_cmx, Sequence):
                for x in _cmx:
                    cmx.append(x)
            else:
                cmx.append(_cmx)

        return np.array(cmx)
