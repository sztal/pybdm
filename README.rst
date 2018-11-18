=============================
PyBDM
=============================

.. image:: https://badge.fury.io/py/pybdm.png
    :target: http://badge.fury.io/py/pybdm

.. image:: https://travis-ci.org/sztal/pybdm.png?branch=master
    :target: https://travis-ci.org/sztal/pybdm

Python implementation of block decomposition method for approximating
algorithmic complexity. It is based on a *split-apply-combine* approach.


Installation
------------

Local development::

    git clone https://github.com/sztal/pybdm
    cd pybdm
    pip install --editable .

Standard usage::

    pip install bdm


Collaboration & git workflow
----------------------------

The general idea is to separate ongoing work from the *master* branch
(main development history). This will help ensuree that no accidental
and unwanted changes will be pushed to overwrite the main codebase.
Thus, a following workflow should be kept at all times:

#. Start working on a new feature.
#. Synchronize the local repository with the remote
   This is done usually by invoking: ``git pull --rebase origin <branch>``,
   where ``<branch>`` is the name of a branch to which one synchronizes,
   usually this will be the *master* branch.
#. Create a new branch for feature development with ``git checkout -b <branch name>``.
#. Work on a feature on the new branch and make as many commits as needed.
#. When done, push the new branch to the repository and create a pull request.
   You can read more about pull requests
   `here <https://help.github.com/articles/about-pull-requests/>`__ and
   `here <https://help.github.com/articles/creating-a-pull-request/>`__.
#. The team discusses the propose changes and apply corrections and adjustments
   if needed.
#. When everyone agree that the feature is ready, the new branch is merged with
   a target branch (usually the *master* branch) by the repository owner.

The above workflow is an example of *shared repository* approach.
An alternative called *fork-and-pull* approach is based on everyone
working on a completely separate forks of the main repository and doing
between repository pull requests. This approach seems to be a little bit
of an overkill for a team of few persons.


Static code analysis
--------------------

At least for the beginnign let's try to keep to the *PyLint* configuration
provided in the repo.


Unit testing
------------

It is preferable, especially in the longterm, to follow the
*Test Driven Development* approach. This is about writing modular code
(decompose complicated logic into simple methods/functions) alongside with
unit tests (simple automatically run functions that test the implementations).

PyTest_ is a very powerful and flexible library for unit testing, so it
seems to be a good choice for this. It is quite complicated in advanced usecases,
but standard usage is extremely simple. It also make benchmarking and profiling
code very easy.

Documentation
-------------

I recommend to use the so-called
`Numpy style <https://numpydoc.readthedocs.io/en/latest/format.html>`__
documentation. It makes it very easy to read docstrings in the source code and
is not to hard to write (check the skeleton code of the ``BDM`` class).
It can also utilize all the power of the
`reStructuredText <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`__
markup.


Package structure
-----------------

TODO

At this point it is important to mention only one special directory

:``_ref``:
    It contains the reference python implementation of 2D BDM.
    It will not be included in the final package, but it will be perhaps
    of a great use during the development
    (i.e. for informing and validating the new implementation).


Useful VS Code extensions
-------------------------

Here is a short list of extensions for the VS Code IDE, that are often useful:

* `Python extension <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`__
* `Anaconda extension pack <https://marketplace.visualstudio.com/items?itemName=ms-python.anaconda-extension-pack>`__
  (for those who use *Anaconda* python distribution)
* `RST preview <https://marketplace.visualstudio.com/items?itemName=tht13.rst-vscode>`__
  (useful for previewing *reStructuredText* files, such as this README)

Features
--------

* TODO


.. _PyTest: https://docs.pytest.org/en/latest/
