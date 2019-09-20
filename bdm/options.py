"""Global package options.

Attributes
----------
warn_if_missing_ctm : bool
        Should warnings for missing CTM values be sent.
raise_if_zero : bool
    Should error be raised in the case of zero BDM value,
    which is usually indicative of malformed data.
"""
# pylint: disable=global-statement,redefined-builtin


_options = {
    'warn_if_missing_ctm': True,
    'raise_if_zero': True
}


def set(warn_if_missing_ctm=None, raise_if_zero=None):
    """Set global package options.

    Parameters
    ----------
    warn_if_missing_ctm : bool
        Should warnings for missing CTM values be sent.
    raise_if_zero : bool
        Should error be raised in the case of zero BDM value,
        which is usually indicative of malformed data.
    """
    global _options
    new_vals = {}
    if warn_if_missing_ctm is not None:
        new_vals['warn_if_missing'] = warn_if_missing_ctm
    if raise_if_zero is not None:
        new_vals['raise_if_zero'] = raise_if_zero
    _options.update(**new_vals)

def get(name=None):
    """Get option value or options dict.

    Parameters
    ----------
    name : str or None
        If ``None`` then the copy of the option dict is returned.
        If ``str`` then the given option value is returned.

    See also
    --------
    set_options : description of the available global options

    Raises
    ------
    KeyError
        If `name` does not give a proper option name.
    """
    global _options
    if name is None:
        return _options.copy()
    try:
        return _options[name]
    except KeyError:
        raise KeyError("there is no '{}' option".format(name))


