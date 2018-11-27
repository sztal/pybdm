"""Benchmarks for `bdm` module."""
import pytest
import numpy as np

# TODO: wait until pytest-benchmark maintainers comply with pytest v4.0.0
# @pytest.mark.benchmark(min_rounds=100)
# def benchmark_bdm_d1_complexity(bdm_d1, benchmark):
#     data = np.random.randint(0, 1, size=512)
#     benchmark(bdm_d1.complexity, data)
