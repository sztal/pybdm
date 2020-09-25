"""Block counter class.

This class is designed only for the context of `pybdm`.
Should be use for different purposes with caution.
"""
from collections import Counter
import numpy as np


class BlockCounter:
    """Block counter.

    `BlockCounter` is a wrapper around a map from block shapes
    to block :py:class:`collections.Counter` objects with methods
    for adding, subtracting and crossing different counters.
    This is useful in particular in the context of perturbation analyses.

    Attributes
    ----------
    counters : mapping
        A mapping from shape tuples to
        :py:class:`collections.Counter` objects.
    """
    def __init__(self, counters):
        self.counters = dict(counters)

    def __repr__(self):
        return "{cn}({counters})".format(
            cn=self.__class__.__name__,
            counters=self.counters
        )

    def __iter__(self):
        return self.counters.__iter__()

    def __getitem__(self, key):
        return self.counters[key]

    def __setitem__(self, key, value):
        self.counters[key]= value

    def __delitem__(self, key):
        del self.counters[key]

    def __add__(self, other):
        """Add other block counter object."""
        out = self.copy()
        for shape, counter in other.items():
            out[shape].update(counter)
        return out._clean()

    def __sub__(self, other):
        """Subtract other block counter object."""
        out = self.copy()
        for shape, counter in other.items():
            out[shape].subtract(counter)
        return out._clean()

    def copy(self):
        return self.__class__({
            k: v.copy() for k, v in self.counters.items()
        })

    def get(self, key, fallback=None):
        return self.counters.get(key, fallback)

    def items(self):
        return self.counters.items()

    def values(self):
        return self.counters.values()

    def update(self, other):
        """Update `self` with respect to `other`.

        Non-positive counts can be present in `other`,
        but are dropped after updating from `self`.
        """
        for shape in other:
            self[shape] += other[shape]

    # -------------------------------------------------------------------------

    def keydiff(self, other):
        """Get set-difference of keys between two block counters.

        Returns
        -------
        dict
            From shapes to :py:class:`numpy.ndarrays` with
            normalized block codes.
        """
        dct = {
            shape: np.array([
                k for k in counter if k not in other.get(shape, {})
            ]) for shape, counter in self.items()
        }
        return { k: v for k, v in dct.items() if v.size > 0 }

    def _clean(self):
        for shape in self:
            self[shape] = Counter({
                k: v for k, v in self[shape].items() if v != 0
            })
        self.counters = { k: v for k, v in self.items() if v }
        return self

    # -------------------------------------------------------------------------