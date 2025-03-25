""" Utility functions for wrangler. """

import numpy as np
import scipy



def all_subclasses(cls):
    """
    Collect all the subclasses of the provided class.

    The search follows the inheritance to the highest-level class.  Intermediate
    base classes are included in the returned set, but not the base class itself.

    Thanks to:
    https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name

    Args:
        cls (object):
            The base class

    Returns:
        :obj:`set`: The unique set of derived classes, including any
        intermediate base classes in the inheritance thread.
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


# NOTE: This is a factor of a few times faster than the previous version.  The
# speed improvement is better for smaller images.  For larger images (e.g.,
# 2048x2048), the improvement is about a factor of 3.
def grow_mask(mask, radius):
    """
    Grow pixels flagged as True in a boolean mask by the provided radius.

    This is largely a convience wrapper for `scipy.ndimage.binary_dilation`_.

    Args:
        mask (`numpy.ndarray`_):
            Boolean mask to process.  Pixels flagged as True are expanded into
            circles with the provided radius.
        radius (scalar-like):
            Radius in pixels to grow the mask.

    Returns:
        `numpy.ndarray`_: The boolean mask grown with the masked region grown by
        the provided radius.
    """
    # Prep for the dilation structure
    size = int(radius*2+1)
    if size % 2 == 0:
        size += 1
    x, y = np.meshgrid(np.arange(size) - size//2, np.arange(size) - size//2)
    # Dilate the mask
    return scipy.ndimage.binary_dilation(mask, structure=x**2 + y**2 <= radius**2)
