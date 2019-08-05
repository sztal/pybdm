"""*PyBDM* exception and warning classes."""


class BDMRuntimeWarning(RuntimeWarning):
    """General BDM related runtime warning class."""
    pass


class CTMDatasetNotFoundError(LookupError):
    """Missing CTM exception class."""
    pass
