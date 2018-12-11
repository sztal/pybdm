"""Tests for `bdm` module."""
# pylint: disable=W0621
import os
import pytest
from bdm.encoding import array_from_string

_dirpath = os.path.join(os.path.split(__file__)[0], 'data')
# Get test input data and expected values
bdm_test_input = []
with open(os.path.join(_dirpath, 'bdm-b2-d4x4-test-input.tsv'), 'r') as stream:
    for line in stream:
        string, bdm = line.strip().split("\t")
        bdm = float(bdm.strip())
        arr = array_from_string(string.strip())
        bdm_test_input.append((arr, bdm))


class TestBDM:

    @pytest.mark.parametrize('x,expected', [
        (array_from_string('000000000001'), 0.175035961691245),
        (array_from_string('000000000010'), 0.0996187277370206)
    ])
    def test_complexity_d1(self, bdm_d1, x, expected):
        output = bdm_d1.bdm(x)
        assert output == expected

    @pytest.mark.parametrize('x,expected', bdm_test_input)
    def test_complexity_d2(self, bdm_d2, x, expected):
        output = bdm_d2.bdm(x)
        assert output == expected
