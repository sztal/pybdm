"""Resources submodule with reference dataset containing
precomputed approximated algorithmic complexity values for
simple objects based on *Coding Theorem Method*
(see :doc:`theory`).

All datasets' names use the following naming scheme: ``ctm-bX-dY``.

Datasets
--------
:``ctm-b2-d12.pkl``:
    Binary strings of length from 1 to 12.
:``ctm-b4-d12.pkl``:
    4-symbols strings of length from 1 to 12.
:``ctm-b5-d12.pkl``:
    5-symbols strings of length from 1 to 12.
:``ctm-b6-d12.pkl``:
    6-symbols strings of length from 1 to 12.
:``ctm-b9-d12.pkl``:
    9-symbols strings of length from 1 to 12.
:``ctm-b2-d4x4.pkl``:
    Square binary matrices of width from 1 to 4.
"""

CTM_DATASETS = {
    # 1D datasets
    'CTM-B2-D12': 'ctm-b2-d12.pkl.gz',
    'CTM-B4-D12': 'ctm-b4-d12.pkl.gz',
    'CTM-B5-D12': 'ctm-b5-d12.pkl.gz',
    'CTM-B6-D12': 'ctm-b6-d12.pkl.gz',
    'CTM-B9-D12': 'ctm-b9-d12.pkl.gz',
    # 2D datasets
    'CTM-B2-D4x4': 'ctm-b2-d4x4.pkl.gz'
}
