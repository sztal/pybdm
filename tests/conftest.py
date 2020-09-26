"""*PyTest* configuration and general purpose fixtures."""
import pytest
from pybdm import BDM
from pybdm.partitions import PartitionRecursive


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
    parser.addoption(
        '--theory', action='store_true', default=False,
        help="Run slow enumerative tests of theoretical properties."
    )

def pytest_collection_modifyitems(config, items):
    """Modify test runner behaviour based on `pytest` settings."""
    run_benchmarks = config.getoption('--benchmarks')
    run_slow = config.getoption('--slow')
    run_theory = config.getoption('--theory')
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
    if not run_theory:
        skip_theory = pytest.mark.skip(reason="Theory tests are run only with --theory")
        for item in items:
            if 'theory' in item.keywords:
                item.add_marker(skip_theory)


# Fixtures --------------------------------------------------------------------

@pytest.fixture(scope='session')
def bdm_d1():
    return BDM(ndim=1)

@pytest.fixture(scope='session')
def bdm_d2():
    return BDM(ndim=2)

@pytest.fixture(scope='session')
def bdm_d2_rec():
    return BDM(ndim=2, partition='recursive', min_length=2)

@pytest.fixture(scope='session')
def bdm_d1_b9():
    return BDM(ndim=1, nsymbols=9, partition=PartitionRecursive, min_length=1)
