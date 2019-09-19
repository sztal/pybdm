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

Local development::

    git clone https://github.com/sztal/pybdm
    cd pybdm
    pip install --editable .

Development version installation::

    pip install git+https://github.com/sztal/pybdm.git

Standard installation (not yet on *PyPI*)::

    pip install bdm


Usage
=====

.. include:: ../USAGE.rst


Authors & Contact
=================

* Szymon Talaga <stalaga@protonmail.com>
* Kostas Tsampourakis <kostas.tsampourakis@gmail.com>


References
==========

.. bibliography:: references.bib
