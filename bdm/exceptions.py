"""*PyBDM* exception and warning classes."""


class BDMRuntimeWarning(RuntimeWarning):
    """General BDM related runtime warning class."""


class CTMDatasetNotFoundError(LookupError):
    """Missing CTM exception class."""
