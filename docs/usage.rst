========
Usage
========

.. include:: shared/USAGE.rst


Boundary conditions
-------------------

Different boundary conditions (see :doc:`/theory`) are implemented by
:py:mod:`partitions` classes.

.. code-block:: python

    from pybdm import BDM
    from pybdm import PartitionIgnore, PartitionRecursive, PartitionCorrelated

    bdm_ignore = BDM(ndim=1, partition=PartitionIgnore)
    # This is default so it is equivalent to
    bdm_ignore = BDM(ndim=1)

    bdm_recurisve = BDM(ndim=1, partition=PartitionRecursive, min_length=2)
    # Minimum size is specified as length, since only symmetric slices
    # are accepted in the case of multidimensional objects.

    bdm_correlated = BDM(ndim=1, partition=PartitionCorrelated)
    # Step-size defaults to 1, so this is equivalent to
    bdm_correlated = BDM(ndim=1, partition=PartitionCorrelated, shift=1)


Normalized BDM
--------------

It is also possible to compute normalized BDM and block entropy values
which are always bounded in the [0, 1] interval.

.. code-block:: python

    import numpy as np
    for pybdm import BDM

    # Minimally complex data
    X = np.ones((100,), dtype=int)

    bdm = BDM(ndim=1)

    # Normalized BDM (equals zero in this case)
    bdm.nbdm(X)
    # Equivalent call
    bdm.bdm(X, normalized=True)

    # Normalized entropy (equals zero in this case)
    bdm.nent(X)
    # Equivalent call
    bdm.ent(X, normalized=True)


Global options
--------------

Some parts of the behavior of the package can be configured globally
via package-level options.

Options are documented in the module docstring for :py:mod:`pybdm.options`.

.. code-block:: python

    from pybdm import options
    # Get a copy of the current options dict
    options.get()
    # Get the current value of an option
    options.get('raise_if_zero')
    # Set and option
    options.set(raise_if_zero=False)


Advanced usage
--------------

Advanced usage and details can be found in the
:py:mod:`pybdm` module documentation.
