"""Tests for `bdm` module."""
import pytest
from bdm import BDM


@pytest.fixture(scope='module')
def bdmobj():
    """Fixture: BDM object for testing."""
    return BDM(dtype='sequence')


class TestBDM:

    def test_split(self, bdmobj):
        pass

    def test_apply(self, bdmobj):
        pass

    def test_combine(self, bdmobj):
        pass

    def test_complexity(self, bdmobj):
        pass
