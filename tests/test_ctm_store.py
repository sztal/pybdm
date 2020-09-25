import pytest
from pybdm import BDM


@pytest.mark.parametrize('ndim,nsymbols', [
    (1, 2), (2, 2),
    (1, 4), (1, 5), (1, 6), (1, 9)
])
def test_cardinalities(ndim, nsymbols):
    bdm = BDM(ndim=ndim, nsymbols=nsymbols)
    for info in bdm.ctm.info.values():
        assert info['equiv_cov'] <= 1
        assert info['blocks_cov'] <= 1
