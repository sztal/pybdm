"""Test options module."""
# pylint: disable=broad-except
import pytest
from pybdm import options
from pybdm.options import  _OPTIONS


@pytest.mark.parametrize('name,expected', [
    (None, _OPTIONS),
    ('bdm_buffer_size', _OPTIONS['bdm_buffer_size']),
    ('invalid_name', NameError)
])
def test_getopt(name, expected):
    try:
        output = options.get(name)
        assert output == expected
    except Exception as exc:
        assert isinstance(exc, expected)

@pytest.mark.parametrize('name,value,expected', [
    ('bdm_if_zero', 'raise', None),
    ('bdm_if_zero', 'ignore', None),
    ('bdm_if_zero', 1, ValueError),
    ('bdm_buffer_size', -1, None),
    ('bdm_buffer_size', 1000, None),
    ('bdm_buffer_size', True, ValueError),
    ('bdm_check_data', True, None),
    ('bdm_check_data', False, None),
    ('bdm_check_data', 1, ValueError)
])
def test_setopt(name,value,expected):
    try:
        options.set(**{name: value})
        assert options.get(name) == value
    except Exception as exc:
        assert isinstance(exc, expected)
