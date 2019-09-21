===============
Theory & Design
===============

.. include:: shared/THEORY.rst

Algorithmic information theory
==============================

Here we give a super brief and simplified overview of the basic notions
of algorithmic information theory, which we will need to describe the
implementation of the package.

Algorithmic / Kolmogorv complexity  (also called K-complexity) is defined
formally as follows:

.. math::

    K_U = \min\{|p|, T(p) = s\}

where :math:`U` is a universal Turing machine, :math:`p` is a program,
:math:`|p|` is the lenght of the program, :math:`s` is a string
:math:`U(p) = s` denotes the fact that the program :math:`p` executed
on the universal Turing machine :math:`U` outputs :math:`s`.

The problem with Kolmogorov complexity is the fact that it is not computable
in the general case due to fundamental limits of computations that arise
from the halting problem
(impossibility to determine whether any given program will ever halt without
actually running this program, possibly for infinite time).

It is also possible to consider the notion of algorithmic probability,
which corresponds to a chance that a randomly selected program will output
:math:`s` when run through :math:`U`. It is defined as follows:

.. math::

    m_U(s) = \sum_{p:U(p) = s}1 / 2^{|p|}

Algorithmic probability is important because it is related directly to
algorithmic complexity via the following law:

.. math::

    K_U(s) = -\log_2 m_U(s) + O(1)

In other words, if there are many long programs that generate a dataset,
then there has to be also a shorter one. The arbitrary constant
:math:`O(1)` is dependent on the choice of a programming language.

Unfortunately, algorithmic probability is also uncomputable for the same reasons
as Kolmogorov complexity. However, it can be approximated in a very
straightforward fashion, since it is possible to explore a vast space
of Turing machines of a given type (i.e. fixed numbers of symbols and states)
and count how many of them produce a given
output and then divide by the total number of machines that halt.
Details of how this can be done can be found in
:cite:`soler-toscano_calculating_2014`. Thus, when exploring machines
with :math:`n` symbols and :math:`m` states algorithmic probability of
a string :math:`s` can be approximated as follows:

.. math::

    D(n, m)(s) = \frac{|\{T \in (n, m) : T \text{ outputs } s \}|}{|\{T \in (n, m) : T \text{ halts } \}|}

Based on that we can approximate Kolmogorov complexity
(via the so-called Coding Theorem Method) as follows:

.. math::

    CTM(n, m)(s) = -\log_2 D(n,m)(s)

This is the basic result that is used to define Block Decomposition Method.


Block Decomposition Method
==========================

The problem with CTM is that, although theoretically computable, it is still
extremely expensive in terms of computation time, since it depends on exploration
of vast spaces of possible Turing machines that may span bilions or even thousands
of bilions of instances. This problem is what Block Decomposition Method (BDM)
tries to address :cite:`zenil_decomposition_2018`.

The idea is to first precompute CTM values for all possible small objects
of a given type (e.g. all binary strings of up to 12 digits or all possible
square binary matrices up to 4x4) and store them in an efficient lookup table.
Then any arbitrarily large object can be decomposed into smaller slices of
appropriate sizes for which CTM values can be looked up very fast.
Finally, the CTM valeus for slices can be aggregated back to a global estimate
of Kolmogorov complexity for the entire object. The proper aggregation
rule is defined via the following BDM formula:

.. math::

    BDM(n,m)(s) = \sum_i CTM(n, m)(s_i) + \log_2(n_i)

where :math:`i` indexes the set of all unique slices
(i.e. CTM values are taken only once for each unique slice)
and :math:`n_i` correspond to the slices' numbers of occurences.


Boundary conditions
-------------------

A technical problem that arises in the context of BDM is what should be done
if a dataset can not be sliced into parts of the same extact shape?
There are at least three solutions:

  1. **Ignore.** Malformed parts can be just ignored.
  2. **Recursive.** Slice malformed parts into smaller pieces
     (down to some minimum size) and lookup CTM values for those smaller pieces.
  3. **Correlated.** Use sliding window instead of slicing. This way all slices
     will be of the proper shape, at least if the window is moved by one element
     at every step.

Let us show how this works with a simple example. Let us consider a 5-by-5
matrix:

+--+--+--+--+--+
|1 |2 |3 |4 |5 |
+--+--+--+--+--+
|6 |7 |8 |9 |10|
+--+--+--+--+--+
|11|12|13|14|15|
+--+--+--+--+--+
|16|17|18|19|20|
+--+--+--+--+--+
|21|22|23|24|25|
+--+--+--+--+--+

If the ignore boundary condition was used, only one 3-by-3 matrix would be
carved out of it:

+--+--+--+
|1 |2 |3 |
+--+--+--+
|6 |7 |8 |
+--+--+--+
|11|12|13|
+--+--+--+

If the recursive condition (with 2-by-2 as the minimum size) was used,
we would get the following slices:

+--+--+--+
|1 |2 |3 |
+--+--+--+
|6 |7 |8 |
+--+--+--+
|11|12|13|
+--+--+--+

+--+--+
|4 |5 |
+--+--+
|9 |10|
+--+--+

+--+--+
|16|17|
+--+--+
|21|22|
+--+--+

+--+--+
|18|18|
+--+--+
|23|24|
+--+--+

If the correlated condition (with step-size of 1) was used,
we would get nine slices like:

+--+--+--+
|1 |2 |3 |
+--+--+--+
|6 |7 |8 |
+--+--+--+
|11|12|13|
+--+--+--+

but each subsequent slice would be moved by one to the left or to the bottom
until its rightmost column or lowest row contain the values on the boundary
of the original matrix.

The condition can yield different results for small objects, but are consistent
asymptotically in the limit of large object sizes. Detailed discussion of
boundary conditions in BDM can be found in :cite:`zenil_decomposition_2018`.


Normalized BDM
--------------

It is also possible to define normalized BDM. First let us note that for any
object of arbitrary size it is possible to construct analogous objects
wit lowest and highest possible BDM values.

* **Least complex object.** This case is trivial. It is enough to consider
  an object filled with only one symbol (e.g. a binary string of only zeros).
* **Most complex object.** The maximum BDM value is given by an object which
  when decomposed (by a given decomposition algorithm) yields slices
  that cover the highest CTM values and are repeated only after all possible
  slices of a given shape have been used once.


Implementation design
=====================

The implementation uses the OOP pattern and follow the *split-apply-comine*
methodology. There are two main classes:

:py:class:`pybdm.bdm.BDM`
  Instances of this class contain pointers to appropriate precomputed CTM
  datasets. They configured by two main attributes:
  dimensionality of target objects (`ndim`)
  and number of symbols used (`nsymbols`). The class implements
  BDM in three stages. The first one is decomposition which relies
  on a particular partition algorithm object (below).
  The second one is lookup (CTM values for slices are looked up).
  The third one is aggregation, in which CTM values for slices are combined
  according to the BDM formula. This stage-wise implementation makes it
  easy to extend the package for instance with new partition algorithms
  and also makes it very easy to parallelize or distribute the entire
  process.
:py:mod:`pybdm.partitions`
  Decomposition stage is implemented by partition classes.
  They are instantiated with attributes describind desired shape of slices,
  step-sizes (`shift`) in correlated decomposition, minimum size in recursive
  decomposition etc. Partition objects are used by BDM objects during
  the decomposition stage.

See :doc:`/usage` for practical examples.


Missing CTM values
------------------

In some cases (especially for alphabets with more than 2 symbols) CTM values
for particular slices may not be available. They are imputed with the maximum
CTM value for slices of a given shape + 1 bit. This is justified because
the exploration of the spaces of Turing machines is done in a way that ensures
that missed value can be only larger than those that were computed.

By default BDM objects send warnings when this happens.
However, this may be turned off:

.. code-block:: python

    # For a particular BDM instance
    bdm = BDM(ndim=1, warn_if_missing_ctm=False)

    # Or globally for all BDM instances
    pybdm.options.set(warn_if_missing_ctm=False)


Block entropy
-------------

:py:class:`pybdm.bdm.BDM` class implements also `ent` method for computing
block entropy, which is useful for comparisons between algorithmic complexity
and entropy as another measure of description length.
Block entropy is just the entropy computed over the distribution of slices
as produced by the partition algorithm.



Perturbation analysis
=====================

:py:mod:`pybdm` provides also an efficient algorithm for perturbation analysis.
The goal of perturbation analysis is to study changes in complexity of a system
under small changes. This makes it possible to identify parts that drive
it towards noise (high complexity) or determinism / structure (low complexity).

For instance, it may be of interest to examine changes of complexity of
an adjacency matrix when particular edges are destroyed (ones switched to zeros).
Some practical applications of such analysis can be found in
:cite:`zenil_causal_2019`.


References
==========

.. bibliography:: references.bib
