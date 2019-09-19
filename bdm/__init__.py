"""Approximation of algorithmic complexity by Block Decomposition Method.

This package provides the :py:class:`bdm.BDM` class for computing approximated
algorithmic complexity of arbitrary binary 1D and 2D arrays based
on the *Block Decomposition Method* (**BDM**).  The relevant theory
is described in
:cite:`soler-toscano_calculating_2014` and
:cite:`zenil_decomposition_2018`.
"""
from .bdm import BDMIgnore, BDMRecursive
from .algorithms import PerturbationExperiment
BDM = BDMIgnore

__author__ = 'AlgoDyn Development Team'
__email__ = 'stalaga@protonmail.com'
__version__ = '0.1.0'
