"""**PyBDM:** Block Decomposition Method

This package provides the :py:class:`pybdm.BDM` class for computing approximated
algorithmic complexity of arbitrarily large binary 1D and 2D arrays
as well as 1D arrays with 4, 5, 6 or 9 unique symbols based
on the *Block Decomposition Method* (**BDM**).
Theory and the design of the package are described in :doc:`theory`.
"""
from .bdm import BDM
from .partitions import PartitionIgnore, PartitionRecursive, PartitionCorrelated
from .algorithms import PerturbationExperiment

__author__ = 'AlgoDyn Development Team'
__email__ = 'stalaga@protonmail.com'
__version__ = '0.1.0'
