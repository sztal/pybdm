TODO (v1.0.0)
-------------

1. Update metadata in setup.py (including the classifiers)
2. Add changelog (once everything else is done)
3. Reimplement core algorithms and data structures so CTM datasets are indexed by integers instead of strings.
   This will reduce the memory footpring of the package significantly and probably will also improve performance.
4. Review and update core API to make it more flexible and user friendly. For instance, it should be possible to specify partition methods by both passing classes and initialized objects.
5. Implement efficient algorithms for mapping positions between full and reduced representations.
6. Reimplement core algorithms for sparse matrices.
7. Adjust perturbation methods so they can handle sparse matrices.
8. Entropy should count the actual number of bits and use different bases.
9. Add more tests.
10. Check and update documentation.
