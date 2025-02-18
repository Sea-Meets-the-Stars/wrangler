""" Catalog utility methods """
import numpy as np

from wrangler import defs

import pandas

from IPython import embed


def vet_main_table(table:pandas.DataFrame, cut_prefix=None,
                   data_model=None, return_disallowed=False):
    """Check that the main table is AOK

    Args:
        table (pandas.DataFrame or dict): [description]
        cut_prefix (str or list, optional): 
            Allow this prefix on the standard datamodel
        data_model (dict, optional): Data model to test
            against.  If None, use the main Ulmo data model

    Returns:
        bool: True = passed the vetting
    """
    if data_model is None:
        data_model = defs.tbl_dmodel
    if cut_prefix is not None:
        # Make the list
        if not isinstance(cut_prefix, list):
            list_cut_prefix = [cut_prefix]
        else:
            list_cut_prefix = cut_prefix

    chk = True
    # Loop on the keys
    disallowed_keys = []
    badtype_keys = []
    for key in table.keys():
        # Allow for cut prefix
        skey = key
        if cut_prefix is not None: 
            # Loop over em
            for icut_prefix in list_cut_prefix:
                if len(key) > len(icut_prefix) and (
                    key[:len(icut_prefix)] == icut_prefix):
                    skey = key[len(icut_prefix):]
        # In data model?
        if not skey in data_model.keys():
            disallowed_keys.append(key)
            chk = False
            continue
        # Allow for dict
        item = table[key] if isinstance(
            table, dict) else table.iloc[0][key] 
        # Check datat type
        if not isinstance(item, data_model[skey]['dtype']):
            badtype_keys.append(key)
            chk = False
    # Required
    missing_required = []
    if 'required' in data_model.keys():
        for key in data_model['required']:
            if key not in table.keys():
                chk=False
                missing_required.append(key)
    # Report
    if len(disallowed_keys) > 0:
        print("These keys are not in the datamodel: {}".format(disallowed_keys))
    if len(badtype_keys) > 0:
        print("These keys have the wrong data type: {}".format(badtype_keys))
    if len(missing_required) > 0:
        print("These required keys were not present: {}".format(missing_required))

    # Return
    if return_disallowed:
        return chk, disallowed_keys
    else:
        return chk