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


def match_ids(IDs, match_IDs, require_in_match=True, assume_unique:bool=False):
    """ Match input IDs to another array of IDs (usually in a table)
    Return the rows aligned with input IDs

    Parameters
    ----------
    IDs : ndarray
        IDs that are to be found in match_IDs
    match_IDs : ndarray
        IDs to be searched
    require_in_match : bool, optional
        Require that each of the input IDs occurs within the match_IDs
    assume_unique : bool, optional
        Assume that both input arrays have unique IDs

    Returns
    -------
    rows : ndarray
      Rows in match_IDs that match to IDs, aligned
      -1 if there is no match
    """
    rows = -1 * np.ones_like(IDs).astype(int)
    # Find which IDs are in match_IDs
    in_match = np.isin(IDs, match_IDs, assume_unique=assume_unique)
    if require_in_match:
        if not np.all(in_match):
            raise IOError("wrangler.tables.utils.match_ids: One or more input IDs not in match_IDs")
    rows[~in_match] = -1
    #
    IDs_inmatch = IDs[in_match]

    # Find indices of input IDs in meta table -- first instance in meta only!
    xsorted = np.argsort(match_IDs)
    ypos = np.searchsorted(match_IDs, IDs_inmatch, sorter=xsorted)
    indices = xsorted[ypos]
    rows[in_match] = indices

    return rows

def calc_grad2(field:np.ndarray, dx:float):
    """
    Calculate the squared magnitude of the gradient of a 2D field.

    This function computes the squared magnitude of the gradient of a 2D array
    (field) using finite differences. The gradient is calculated along both
    the x and y axes, and the squared magnitude is returned.

    Parameters:
        field (np.ndarray): A 2D array representing the field for which the 
                            gradient magnitude is to be calculated.
        dx (float): The grid spacing (assumed to be uniform) in both x and y 
                    directions.

    Returns:
        np.ndarray: A 2D array of the same shape as `field`, containing the 
                    squared magnitude of the gradient at each point.
    """

    # Gradient
    dfdx = np.gradient(field, axis=1) / dx
    dfdy = np.gradient(field, axis=0) / dx

    # Magnitude
    grad_f2 = dfdx**2 + dfdy**2

    return grad_f2