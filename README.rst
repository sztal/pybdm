=============================================================
PyBDM: Python interface to the *Block Deincomposition Method*
=============================================================

.. image:: https://badge.fury.io/py/pybdm.png
    :target: http://badge.fury.io/py/pybdm

.. image:: https://travis-ci.org/sztal/pybdm.png?branch=master
    :target: https://travis-ci.org/sztal/pybdm


The Block Decomposition Method (BDM) approximates algorithmic complexity
of a dataset of arbitrary size, that is, the length of the shortest computer
program that generates it. This is not trivial as algorithmic complexity
is not a computable quantity in the general case and estimation of
algorithmic complexity of a dataset can be very useful as it points to
mechanistic connections between elements of a system, even such that
do not yield any regular statistical patterns that can be captured with
more traditional tools based on probability theory and information theory.

Currently 1D and 2D binary arrays are supported, but this may be extended to higher dimensionalities and more complex alphabets in the future.

BDM and the necessary parts of the algorithmic information theory
it is based on are described in `this paper <https://www.mdpi.com/1099-4300/20/8/605>`__.


Installation
============

Local development::

    git clone https://github.com/sztal/pybdm
    cd pybdm
    pip install --editable .

Development version installation::

    pip install git+https://github.com/sztal/pybdm.git

Standard installation::

    pip install bdm


Usage
=====

The BDM is implemented using the object-oriented approach and expects
input represented as `Numpy <http://www.numpy.org/>`__ arrays of integer type.

.. highlights::

   ``BDM`` objects operate exclusively on **integer arrays**.
   Hence, any alphabet must be first mapped to a set of integers ranging
   from ``0`` to ``k``.


Binary sequences (1D)
---------------------

.. code-block:: python

    import numpy as np
    from bdm import BDM

    # Create a dataset (must be of integer type)
    X = np.ones((100,), dtype=int)

    # Initialize BDM object
    # ndim argument specifies dimensionality of BDM
    bdm = BDM(ndim=1)

    # Compute BDM
    bdm.bdm(X)

    # BDM objects may also compute standard Shannon entropy in base 2
    bdm.entropy(X)


Binary matrices (2D)
--------------------

.. code-block:: python

    import numpy as np
    from bdm import BDM

    # Create a dataset (must be of integer type)
    X = np.ones((100, 100), dtype=int)

    # Initialize BDM object
    bdm = BDM(ndim=2)

    # Compute BDM
    bdm.bdm(X)

    # BDM objects may also compute standard Shannon entropy in base 2
    bdm.entropy(X)


Parallel processing
-------------------

*PyBDM* was designed with parallel processing in mind.
Using modern packages for parallelization such as
`joblib <https://joblib.readthedocs.io/en/latest/parallel.html>`__
makes it really easy to compute BDM for massive objects.

In this example we will slice a 1000x1000 dataset into 200x200 pieces
compute so-called counter objects (final BDM computation operates on such objects)
in parallel in 4 independent processes, and aggregate the results
into a single BDM approximation of the algorithmic complexity of the dataset.

.. code-block:: python

    import numpy as np
    from joblib import Parallel, delayed
    from bdm import BDM
    from bdm.utils import slice_dataset

    # Create a dataset (must be of integer type)
    X = np.ones((1000, 1000), dtype=int)

    # Initialize BDM object
    BDM = bdm(ndim=2)

    # Compute counter objects in parallel
    counters = Parallel(n_jobs=4) \
        (delayed(bdm.count_and_lookup)(d) for d in slice_dataset(X, (200, 200)))

    # Compute BDM
    bdm.compute_bdm(*counters)


Perturbation analysis
---------------------

Besides the main *Block Decomposition Method* implementation *PyBDM* provides
also an efficient algorithm for perturbation analysis based on *BDM*
(or standard Shannon entropy).

A perturbation experiment studies change of *BDM* / entropy under changes
applied to the underlying dataset. This is the main tool for detecting
parts of a system having some causal significance as opposed
to noise parts.

Parts which after yield negative contribution to the overall
complexity after change are likely to be important for the system,
since their removal make it more noisy. On the other hand parts that yield
positive contribution to the overall complexity after change are likely
to be noise since they extend the system's description length.

.. code-block:: python

    import numpy as np
    from bdm import BDM
    from bdm.algorithms import PerturbationExperiment

    # Create a dataset (must be of integer type)
    X = np.ones((100, 100), dtype=int)

    # Initialize BDM object
    BDM = bdm(ndim=2)

    # Initialize perturbation experiment object
    # (may be run for both bdm or entropy)
    perturbation = PerturbationExperiment(X, bdm, metric='bdm')

    # Compute BDM change for all data points
    delta_bdm = perturbation.run(changes=None)

    # Compute BDM change for selected data points and keep the changes while running
    # The last column in the changes array specifies the new value to apply.
    # If the new value is negative, then it is selected randomly
    # from the rest of alhpabet (the existing value is overwritten with some other value).
    # The first columsn specify indices of elements to perturb.
    changes = np.array([[0, 0, -1], [10, 10, -1]])
    delta_bdm = perturbation.run(changes=changes, keep_changes=True)


Authors & Contact
=================

* Szymon Talaga <stalaga@protonmail.com>
* Kostas Tsampourakis <kostas.tsampourakis@gmail.com>
