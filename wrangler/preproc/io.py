""" Module for I/O related to pre-processing"""
import os
from pkg_resources import resource_filename

import json

def load_options(filename:str):
    """
    Load a PreProc options dict

    Parameters
    ----------
    filename (str) : filename of preproc JSON file

    Returns
    -------
    pdict : dict

    """
    # Tuples
    with open(filename, 'rt') as fh:
        pdict = json.load(fh)
    # Tuple me
    for key in ['med_size', 'dscale_size']:
        if key in pdict:
            pdict[key] = tuple(pdict[key])
    # Return
    return pdict

