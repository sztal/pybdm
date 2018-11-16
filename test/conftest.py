"""*PyTest* configuration and general purpose fixtures."""
import pytest


def pytest_addoption(parser):
    """Custom `pytest` command-line options."""
    parser.addoption(
        '--benchmarks', action='store_true', default=False,
        help="Run benchmarks (instead of tests)."
    )
    parser.addoption(
        '--slow', action='store_true', default=False,
        help="Run slow tests / benchmarks."""
    )

def pytest_collection_modifyitems(config, items):
    """Modify test runner behaviour based on `pytest` settings."""
    run_benchmarks = config.getoption('--benchmarks')
    run_slow = config.getoption('--slow')
    if run_benchmarks:
        skip_test = \
            pytest.mark.skip(reason="Only benchmarks are run with --benchmarks")
        for item in items:
            if 'benchmark' not in item.keywords:
                item.add_marker(skip_test)
    else:
        skip_benchmark = \
            pytest.mark.skip(reason="Benchmarks are run only with --run-benchmark")
        for item in items:
            if 'benchmark' in item.keywords:
                item.add_marker(skip_benchmark)
    if not run_slow:
        skip_slow = pytest.mark.skip(reason="Slow tests are run only with --slow")
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip_slow)
