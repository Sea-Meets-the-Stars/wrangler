""" Extraction methods for OGCM data, e.g. LLC4320 """

import numpy as np
import pandas
import xarray

import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from wrangler.ogcm import llc as wr_llc
from wrangler.preproc.pp_ogcm import gradb2_cutout, b_cutout 
from wrangler.preproc.pp_ogcm import gradfield2_cutout
from wrangler.preproc.pp_ogcm import Fs_cutout, current_cutout
from wrangler.preproc import field as pp_field
from wrangler import utils as wr_utils

from IPython import embed


def llc_datetime(llc_table:pandas.DataFrame, field:str, udate:str, pdict:str, 
                     cutout_size=(64,64), fixed_km=None, n_cores=10, 
                     coords_ds=None, test_failures:bool=False, 
                     test_process:bool=False, debug=False):
    """Main routine to extract and pre-process LLC data for a single 
        timestamp

    Args:
        llc_table (pandas.DataFrame): cutout table
            Should have been cut to a single datetime matching udate
        field (str): Field to extract
            Allowed options are: 'SST', 'SSS', 'Divb2', 'normDivb2', 'b', 'SSH'
        pdict (dict): Preprocessing steps. 
        cutout_size (tuple, optional): Defines cutout shape. Defaults to (64,64).
        fixed_km (float, optional): Require cutout to be this size in km
        n_cores (int, optional): Number of cores for parallel processing. Defaults to 10.
        valid_fraction (float, optional): [description]. Defaults to 1..
        dlocal (bool, optional): Data files are local? Defaults to False.
        override_RAM (bool, optional): Over-ride RAM warning?

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: Modified in place table
        success (np.ndarray): Bool array of successful extractions
        pp_fields (np.ndarray): Pre-processed fields
        final_meta (pandas.DataFrame): Meta data for each cutout

    """
    # Check date
    assert np.all(llc_table.datetime == udate)

    # Prepping
    R_earth = 6371. # km
    circum = 2 * np.pi* R_earth
    km_deg = circum / 360.

    # Load coords?
    if fixed_km is not None and coords_ds is None:
        coords_ds = wr_llc.load_coords()
    
    # Setup for parallel
    if field in ['SST','SSS','SSTK', 'SSSs', 'SSH', 'SSHa', 'U', 'V', 'SSHs']:
        map_fn = partial(pp_field.multi_process, pdict=pdict)
    elif field in ['DivSST2', 'DivSSS2']:
        map_fn = partial(gradfield2_cutout, **pdict)
    elif field in ['Divb2']:
        map_fn = partial(gradb2_cutout, **pdict)
    elif field in ['b']:
        map_fn = partial(b_cutout, **pdict)
    elif field in ['Fs']:  # Frontogenesis tendency
        map_fn = partial(Fs_cutout, **pdict)
    elif field in ['OW', 'strain_rate', 'divergence', 'vorticity', 
                   'Cu', 'L']:  # Current fields
        pdict['field'] = field
        map_fn = partial(current_cutout, **pdict)
    else:
        raise IOError(f"Not ready for this field {field}")

    # Load data
    filename = wr_llc.grab_llc_datafile(udate)
    ds = xarray.open_dataset(filename)

    # Field
    data2, data3, data4 = None, None, None
    if field in ['SST', 'DivSST2', 'SSTK']:
        data = ds.Theta.values
        if field == 'SSTK':
            data += 273.15 # Kelvin
    elif field in ['SSH', 'SSHs', 'SSHa']:
        data = ds.Eta.values
    elif field in ['SSS', 'SSSs', 'DivSSS2']:
        data = ds.Salt.values
    elif field in ['Divb2', 'b']:
        data = ds.Theta.values
        data2 = ds.Salt.values
    elif field == 'U':
        data = ds.U.values
    elif field == 'V':
        data = ds.V.values
    elif field == 'Fs':
        data = ds.U.values
        data2 = ds.V.values
        data3 = ds.Theta.values
        data4 = ds.Salt.values
    elif field in ['OW', 'strain_rate', 'divergence', 
                   'vorticity', 'Cu', 'L']:
        data = ds.U.values
        data2 = ds.V.values
    else:
        raise IOError(f"Not ready for this field {field}")

    # Parse 
    sub_UID = llc_table.UID.values

    # Load up the cutouts
    fields, fields2, fields3, fields4, smooth_pixs = [], [], [], [], []
    for r, c in zip(llc_table.row, llc_table.col):
        if fixed_km is None:
            dr = cutout_size[0]
            dc = cutout_size[1]
        else:
            dlat_km = (coords_ds.lat.data[r+1,c]-coords_ds.lat.data[r,c]) * km_deg
            dr = int(np.round(fixed_km / dlat_km))
            dc = dr
        # Deal with smoothing
        if 'smooth_km' in pdict.keys():
            smooth_pix = int(np.round(pdict['smooth_km'] / dlat_km))
            pad = 2*smooth_pix
            #
            use_r = r - pad
            dr += 2*pad
            use_c = c - pad
            dc += 2*pad
            smooth_pixs.append(smooth_pix)
        else:
            use_r, use_c = r, c
        # Off the image?
        if (r+dr >= data.shape[0]) or (c+dc > data.shape[1]) or (
            use_r < 0) or (use_c < 0):
            fields.append(None)
        else:
            fields.append(data[use_r:use_r+dr, use_c:use_c+dc])

        # More?
        for dataN, fieldsN in zip([data2, data3, data4],
                               [fields2, fields3, fields4]):
            if dataN is None:
                continue
            if (r+dr >= dataN.shape[0]) or (c+dc > dataN.shape[1]) or (
                use_r < 0) or (use_c < 0):
                fieldsN.append(None)
            else:
                fieldsN.append(dataN[use_r:use_r+dr, use_c:use_c+dc])

    # Other special cases
    if field in ['Cu', 'L']:  # This needs to come after the above for loop
        # Generate f
        fields3 = wr_utils.coriolis(llc_table.lat.values).tolist()

    print("Cutouts loaded for {}".format(filename))

    # Prep items
    zipitems = [fields]
    for fieldsN in [fields2, fields3, fields4]:
        if len(fieldsN) > 0:
            zipitems.append(fieldsN)
    zipitems.append(sub_UID)
    if 'smooth_km' in pdict.keys():
        zipitems.append(smooth_pixs)
    items = [item for item in zip(*zipitems)]

    #embed(header='debug 176 of ex_ogcm.py')
    # Multi-process time
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    print("Done processing")
    # Debuggin
    if test_failures:
        answers[50] = [None, answers[50][1], None]

    # Deal with failures
    #answers = [f for f in answers if f is not None]
    img_UID = [item[1] for item in answers]

    # Slurp
    pp_fields = [item[0] for item in answers]
    meta = [item[2] for item in answers]

    # Clean up
    del answers, fields, items

    # Fuss with indices
    tbl_UID = llc_table.UID.values
    img_UID = np.array(img_UID)
    img_idx = np.arange(len(img_UID))


    # Clean up time (indices and bad data)

    # Find the bad ones (if any)
    bad_idx = [int(item[1]) for item in zip(pp_fields, img_idx) if item[0] is None]
    good_idx = np.array([int(item[1]) for item in zip(pp_fields, img_idx) if item[0] is not None])

    success = np.ones(len(pp_fields), dtype=bool)

    # Replace with -1 images
    if len(bad_idx) > 0:
        size = pp_fields[good_idx[0]].shape
        bad_img = -1*np.ones(size)
        for ii in bad_idx:
            pp_fields[ii] = bad_img.copy()
            success[ii] = False

    # Align pp_fields with the Table
    ppf_idx = wr_utils.match_ids(img_UID, tbl_UID, require_in_match=True)
    pp_fields = np.array(pp_fields)[ppf_idx]

    # Meta time
    good_meta = pandas.DataFrame([item for item in meta if item is not None])
    final_meta = pandas.DataFrame()
    #tbl_idx = wr_utils.match_ids(img_UID[good_idx], tbl_UID, require_in_match=True)

    for key in good_meta.keys():
        final_meta[key] = np.zeros(len(ppf_idx))
        # Align with images
        final_meta.loc[good_idx, key] = good_meta[key].values
        #final_meta.loc[tbl_idx, key] = good_meta[key].values

    # Add index of the images
    final_meta['gidx'] = -1
    final_meta.loc[good_idx, 'gidx'] = np.arange(len(good_idx))

    # Return
    return success, pp_fields, final_meta, filename
