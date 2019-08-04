"""Approximation of algorithmic complexity by Block Decomposition Method.

This package provides the :py:class:`bdm.BDM` class for computing approximated
algorithmic complexity of arbitrary binary 1D and 2D arrays based
on the *Block Decomposition Method* (**BDM**). The method is descibed
`in this paper <https://www.mdpi.com/1099-4300/20/8/605>`__.
"""
from .bdm import BDMIgnore, BDMRecursive
from .algorithms import PerturbationExperiment
BDM = BDMIgnore

__author__ = 'AlgoDyn Development Team'
__email__ = 'stalaga@protonmail.com'
__version__ = '0.0.0'
