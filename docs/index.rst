.. complexity documentation master file, created by
   sphinx-quickstart on Tue Jul  9 22:26:36 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2

   installation
   usage
   theory
   contributing
   authors
   history
   modules

=============================================================
PyBDM: Python interface to the *Block Deincomposition Method*
=============================================================

.. image:: https://badge.fury.io/py/pybdm.png
    :target: http://badge.fury.io/py/pybdm

.. image:: https://travis-ci.org/sztal/pybdm.svg?branch=master
    :target: https://travis-ci.org/sztal/pybdm

.. image:: https://codecov.io/gh/sztal/pybdm/branch/master/graph/badge.svg?branch=master
    :target: https://codecov.io/gh/sztal/pybdm

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
it is based on are described in
:cite:`soler-toscano_calculating_2014` and
:cite:`zenil_decomposition_2018`.

Installation
============

.. include:: shared/INSTALLATION.rst


Feedback
========

If you have any suggestions or questions about **PyBDM** feel free to email me
at stalaga@protonmail.com.

If you encounter any errors or problems with **PyBDM**, please let me know!
Open an Issue at the GitHub http://github.com/sztal/pybdm main repository.
