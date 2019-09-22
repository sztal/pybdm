"""*PyBDM* exception and warning classes."""


class BDMRuntimeWarning(RuntimeWarning):
    """General BDM related runtime warning class."""


class BDMConfigurationError(AttributeError):
    """General BDM configuration error."""


class CTMDatasetNotFoundError(LookupError):
    """Missing CTM exception class."""
