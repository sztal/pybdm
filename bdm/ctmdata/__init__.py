"""Resources submodule with reference dataset containing
precomputed approximated algorithmic complexity values for
simple objects based on the *Coding Theorem Method*.

All datasets' names use the following naming scheme: ``ctm-bX-dY``.

Datasets
--------
:``ctm-b2-d12.pickle``:
    Binary strings of length from 1 to 12.
:``ctm-b2-d4x4.pickle``:
    Square binary matrices of width from 1 to 4.
"""

CTM_DATASETS = {
    'CTM-B2-D12': 'ctm-b2-d12.pkl',
    'CTM-B2-D12B': 'ctm-b2-d12b.pkl',
    'CTM-B2-D4x4': 'ctm-b2-d4x4.pkl'
}
