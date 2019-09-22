=======
Roadmap
=======

Next major release
==================

* Support for sparse arrays
* Perturbation experiment for growing/shrinking systems
* Implement Bayesian framework for approximating probability of
  a stochastic generating source
* Add a partition algorithm with the periodic boundary condition
* Use integer-based conding of dataset blocks
  (to lower memory-footprint). This will be done only if it will be possible
  to use integer coding without significantly negative impact on the performance.
* Configure automatic tests for OSX and Windows.

Distant goals
=============

* Reimplement core algorithms and classes in C++
  and access them via cython
