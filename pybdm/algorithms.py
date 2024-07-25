"""Algorithms based on ``BDM`` objects."""
from itertools import product
from dataclasses import dataclass
from random import choice
import numpy as np

@dataclass
class DeconvolutionResult:
    """A deconvolution result data class.

    The deconvolution of a graph results in producing information signatures 
    of each edge and their differences when sorted from maximum information
    signature to minimum. Using an auxilliary cutoff, the edges for deletion 
    can be determined if the difference of one edge from another is greater 
    than auxilliary cutoff + log(2) (theoretically).

    Attributes:
        auxiliary_cutoff (float): 
            The cutoff value used to determine which edges are to be deleted.
        info_loss_values (np.array): 
            A *Numpy* array containing the information signature loss of 
            each edge. This is retrieved when an edge is perturbed.
        info_loss_edges (np.array):
            A *Numpy* array containing the edges of each loss value in 
            ``info_loss_values``. The ith edge in this list corresponds to the
            ith information loss value in ``info_loss_values``. 
        differences (np.array):
            A *Numpy* array that is produced after sorting the ``info_loss_values``
            from maximum to minimum, these are the differences of the ith 
            information loss value minus the i+1th information loss value.
        edges_for_deletion (np.array):
            A *Numpy* array that is produced based on the ``auxiliary_cutoff``
            and if the differences of edges are greater than this cuttoff.
        difference_filter (np.array):
            A *Numpy* array containing ``True`` or ``False`` if the ith edge
            in ``info_loss_edges`` is included in ``edges_for_deletion``
    """
    auxiliary_cutoff: float
    info_loss_values: np.array
    info_loss_edges: np.array
    differences : np.array
    edges_for_deletion: np.array
    difference_filter: np.array

class PerturbationExperiment:
    """Perturbation experiment class.

    Perturbation experiment studies change of BDM / entropy under changes
    applied to the underlying dataset. This is the main tool for detecting
    parts of a system having some causal significance as opposed
    to noise parts.

    Parts which when perturbed yield negative contribution to the overall
    complexity after change are likely to be important for the system,
    since their removal make it more noisy. On the other hand parts that yield
    positive contribution to the overall complexity after change are likely
    to be noise since they elongate the system's description length.

    Attributes
    ----------
    bdm : BDM
        BDM object. It has to be configured properly to handle
        the dataset that is to be studied.
    X : array_like (optional)
        Dataset for perturbation analysis. May be set later.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.

    See also
    --------
    pybdm.bdm.BDM : BDM computations

    Examples
    --------
    >>> import numpy as np
    >>> from pybdm import BDM, PerturbationExperiment
    >>> X = np.random.randint(0, 2, (100, 100))
    >>> bdm = BDM(ndim=2)
    >>> pe = PerturbationExperiment(bdm, metric='bdm')
    >>> pe.set_data(X)
    >>> idx = np.argwhere(X) # Perturb only ones (1 --> 0)
    >>> delta_bdm = pe.run(idx)
    >>> len(delta_bdm) == idx.shape[0]
    True

    More examples can be found in :doc:`usage`.
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

    def __repr__(self):
        cn = self.__class__.__name__
        bdm = str(self.bdm)[1:-1]
        return f"<{cn}(metric={self.metric}) with {bdm}>"

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
        """Set dataset for the perturbation experiment.

        Parameters
        ----------
        X : array_like
            Dataset to perturb.
        """
        if not np.isin(np.unique(X), range(self.bdm.nsymbols)).all():
            raise ValueError("'X' is malformed (too many or ill-mapped symbols)")
        self.X = X
        self._counter = self.bdm.decompose_and_count(X)
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
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> perturbation.perturb((10, ), -1) # doctest: +FLOAT_CMP
        np.float64(26.91763012739709)
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
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> changes = np.array([10, 20])
        >>> perturbation.run(changes) # doctest: +FLOAT_CMP
        array([26.91763013, 27.34823681])
        """
        if idx is None:
            indexes = [ range(k) for k in self.X.shape ]
            idx = np.array(list(product(*indexes)), dtype=int)
        if values is None:
            values = np.full((idx.shape[0], ), -1, dtype=int)
        return np.apply_along_axis(
            lambda r: self.perturb(tuple(r[:-1]), r[-1], keep_changes=keep_changes),
            axis=1,
            arr=np.column_stack((idx, values))
        )

    def _sort_info_loss_values(self, info_loss_values, info_loss_edges):

        sorted_values = np.argsort(-info_loss_values[:,0])
        info_loss_values = info_loss_values[sorted_values]
        info_loss_edges = info_loss_edges[sorted_values]

        return info_loss_values, info_loss_edges

    def _compute_differences(self, info_loss_values):
        return np.diff(info_loss_values[:, -1]) * -1

    def _filter_by_differences(self, auxiliary_cutoff, info_loss_edges, differences, is_directed):

        difference_filter = list(np.isin(
            np.arange(len(differences)),
            np.where(abs(differences - np.log2(2)) > auxiliary_cutoff)
        ))
        difference_filter.extend([False])

        edges_for_deletion = info_loss_edges[difference_filter]

        if not is_directed:
            edges_for_deletion = np.array([*edges_for_deletion, *edges_for_deletion[:, [1,0]]], dtype=int)

        return edges_for_deletion, difference_filter

    def _process_deconvolution(self, auxiliary_cutoff, info_loss_values, info_loss_edges, is_directed):

        info_loss_values, info_loss_edges = self._sort_info_loss_values(info_loss_values, info_loss_edges)
        differences = self._compute_differences(info_loss_values)
        edges_for_deletion, difference_filter = self._filter_by_differences(
            auxiliary_cutoff, info_loss_edges, differences, is_directed
        )

        return DeconvolutionResult(
            auxiliary_cutoff, info_loss_values, info_loss_edges,
            differences, edges_for_deletion, difference_filter
        )

    def deconvolve(self, auxiliary_cutoff, is_directed=False, keep_changes=False):
        """Run causal deconvolution.

        Parameters
        ----------
        auxiliary_cutoff : float
            Value to be used as the cutoff when cutting edges
            based on their information signature differences.
        is_directed : bool
            If ``True`` then considers the dataset to be a directed
            graph and thus retrieves the information signatures of all
            edges.
            If ``False`` then considers the dataset to be an undirected
            graph and thus considers edges (i,j) and (j,i) the same when
            retrieving the information signatures of each edge.
        keep_changes: bool
            If ``True`` then changes in the dataset are persistent,
            so all the edges that have been cut will be applied to the dataset.

        Returns
        -------
        DeconvolutionResult
            a dataclass that contains different values for evaluation.
        
        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=2,shape=(4,4))
        >>> X = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
        ...               [1, 0, 1, 1, 0, 0, 0, 0],
        ...               [1, 1, 0, 1, 0, 0, 0, 0],
        ...               [1, 1, 1, 0, 0, 0, 0, 0],
        ...               [1, 0, 0, 0, 0, 1, 1, 1],
        ...               [0, 0, 0, 0, 1, 0, 1, 1],
        ...               [0, 0, 0, 0, 1, 1, 0, 1],
        ...               [0, 0, 0, 0, 1, 1, 1, 0]])
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> result = perturbation.deconvolve(auxiliary_cutoff=20, is_directed=False)
        >>> result.edges_for_deletion
        array([[0, 4], [4, 0]])
        """
        if self.X.ndim != 2:
            raise ValueError('Deconvolution is only supported for 2D datasets')

        info_loss_values = np.empty((0, 1), dtype=float)
        info_loss_edges = np.empty((0, 2), dtype=int)
        deleted_edge_graph = np.copy(self.X)

        nonzero_edges = deleted_edge_graph if is_directed else np.triu(deleted_edge_graph)
        nonzero_edges = np.column_stack(np.nonzero(nonzero_edges))
        original_bdm = self.bdm.bdm(self.X)

        for edge in nonzero_edges:

            edges_to_perturb = ((edge[0], edge[1])) if is_directed else (edge, edge[::-1])

            deleted_edge_graph[edges_to_perturb] = 0

            deleted_edge_bdm = self.bdm.bdm(deleted_edge_graph)
            info_loss =  original_bdm - deleted_edge_bdm

            info_loss_edges = np.vstack((info_loss_edges, np.array([edge])))
            info_loss_values = np.vstack((info_loss_values, np.array([info_loss])))

            deleted_edge_graph[edges_to_perturb] = 1

        deconvolution_result = self._process_deconvolution(
            auxiliary_cutoff, info_loss_values, info_loss_edges, is_directed
        )

        if deconvolution_result.edges_for_deletion.size == 0:
            return deconvolution_result

        if not keep_changes:
            deleted_edge_graph[
                deconvolution_result.edges_for_deletion[:,0],
                deconvolution_result.edges_for_deletion[:,1]
            ] = 0
            return deconvolution_result

        self.run(idx=deconvolution_result.edges_for_deletion,keep_changes=keep_changes)
        return deconvolution_result
