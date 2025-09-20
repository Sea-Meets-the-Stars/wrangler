
import numpy as np

def stats(field:np.ndarray):
    """
    Compute statistical metadata for a given NumPy array.

    Parameters:
    -----------
    field : np.ndarray
        A NumPy array for which statistical metadata will be calculated.

    Returns:
    --------
    meta_dict : dict
        A dictionary containing the following statistical metadata:
        - 'mx': Maximum value in the array.
        - 'mn': Minimum value in the array.
        - 'p10': Value at the 10th percentile.
        - 'p90': Value at the 90th percentile.
        - 'mu': Mean value of the array.
    """
    ff = field.flatten()
    #
    meta_dict = {}
    srt = np.argsort(ff)
    meta_dict['mx'] = ff[srt[-1]]
    meta_dict['mn'] = ff[srt[0]]
    i10 = int(0.1*field.size)
    i90 = int(0.9*field.size)
    meta_dict['p10'] = ff[srt[i10]]
    meta_dict['p90'] = ff[srt[i90]]

    # Mean
    meta_dict['mu'] = np.mean(ff)

    # Return
    return meta_dict