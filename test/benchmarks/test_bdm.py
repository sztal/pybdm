"""Benchmarks for `bdm` module."""
import pytest

def some_function():
    pass

@pytest.mark.benchmark(min_rounds=30)
def test_benchmark_some_function(benchmark):
    benchmark(some_function)
