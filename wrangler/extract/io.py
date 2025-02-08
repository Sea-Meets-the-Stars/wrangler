
""" Module for I/O related to pre-processing"""
import os
import json

from remote_sensing import io as rs_io

def load_options(filename:str):
    """
    Load a PreProc options dict

    Parameters
    ----------
    root (str) : Root of preproc JSON file

    Returns
    -------
    edict : dict

    """
    # Load
    edict = rs_io.loadjson(filename)

    # Tuple me
    for key in ['med_size', 'dscale_size']:
        if key in edict:
            edict[key] = tuple(edict[key])
    # Return
    return edict
