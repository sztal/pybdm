"""Global package options.

Use :py:func:`set` and :py:func:`get` functions
to set and retrieve options.

Attributes
----------
bdm_if_zero : {'raise', 'ignore'}
    Should error be raised in the case of zero BDM value,
    which is usually indicative of malformed data.
bdm_buffer_size : int
    Size of block chunks to use to speed up BDM calculations.
    The bigger chunks, the faster computations and the higher
    the memory cost. Non-positive values mean all blocks will be processed
    at once (not recommended for large datasets).
bdm_check_data : bool
    Should data format be checked before running BDM calculations.
    May be disabled to gain some speed.
ent_rv : bool
    Should entropy of a single random block be calculated
    instead of a sum over all blocks.
"""
# pylint: disable=redefined-builtin,protected-access
import numpy as np

_OPTIONS = {
    'bdm_if_zero': 'raise',
    'bdm_buffer_size': 5000,
    'bdm_check_data': True,
    'ent_rv': False
}


class Options:
    """Options table.

    Attributes
    ----------
    bdm_if_zero : {'raise', 'ignore'}
        Should error be raised in the case of zero BDM value,
        which is usually indicative of malformed data.
    bdm_buffer_size : int
        Size of block chunks to use to speed up BDM calculations.
        The bigger chunks, the faster computations and the higher
        the memory cost. Non-positive values mean all blocks will be processed
        at once (not recommended for large datasets).
    bdm_check_data : bool
        Should data format be checked before running BDM calculations.
        May be disabled to gain some speed.
    """
    def __init__(self, **kwds):
        self._dct = {}
        for k in kwds:
            self._check_key(k)
        for k, v in kwds.items():
            setattr(self, k, v)

    def __repr__(self):
        return "{cn}({d})".format(
            cn=self.__class__.__name__,
            d={ **_OPTIONS, **self._dct }
        )

    def __getitem__(self, key):
        self._check_key(key)
        return self._dct.get(key, _OPTIONS[key])

    def __setitem__(self, key, value):
        self._check_key(key)
        setattr(self, key, value)

    @staticmethod
    def _check_key(key):
        if key not in _OPTIONS:
            raise NameError("'{}' is not a valid option".format(key))


    # Properties --------------------------------------------------------------

    @property
    def _opt(self):
        return { **_OPTIONS, **self._dct }

    @property
    def bdm_if_zero(self):
        return self._opt['bdm_if_zero']
    @bdm_if_zero.setter
    def bdm_if_zero(self, newval):
        valid = ('raise', 'ignore')
        if not newval in valid:
            raise ValueError("'bdm_if_zero' has to be one of {}".format(valid))
        self._dct['bdm_if_zero'] = newval

    @property
    def bdm_buffer_size(self):
        return self._opt['bdm_buffer_size']
    @bdm_buffer_size.setter
    def bdm_buffer_size(self, newval):
        if not isinstance(newval, (int, np.integer)) or isinstance(newval, bool):
            raise ValueError("'bdm_buffer_size' has to be an integer")
        self._dct['bdm_buffer_size'] = newval

    @property
    def bdm_check_data(self):
        return self._opt['bdm_check_data']
    @bdm_check_data.setter
    def bdm_check_data(self, newval):
        if not isinstance(newval, bool):
            raise ValueError("'bdm_check_data' has to be a boolean")
        self._dct['bdm_check_data'] = newval

    @property
    def ent_rv(self):
        return self._opt['ent_rv']
    @ent_rv.setter
    def ent_rv(self, newval):
        if not isinstance(newval, bool):
            raise ValueError("'ent_rv' has to be a boolean")
        self._dct['ent_rv'] = newval


def set(**kwds):
    """Set global package options.

    New values are passed via keyword arguments.
    Names must match existing options.

    See :py:mod:`pybdm.options` for the list of options.

    Raises
    ------
    NameError
        When an invalid option is provided.
    """
    opt = Options(**kwds)
    _OPTIONS.update(opt._dct)

def get(name=None):
    """Get option value or options dict.

    See :py:mod:`pybdm.options` for the list of options.

    Parameters
    ----------
    name : str or None
        If ``None`` then the copy of the option dict is returned.
        If ``str`` then the given option value is returned.

    Raises
    ------
    NameError
        If `name` does not match any proper option name.
    """
    opt = Options()
    if name is None:
        return opt._opt.copy()
    return opt[name]
