""" Extraction methods for OGCM data, e.g. LLC4320 """

import numpy as np
import pandas
import xarray

import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from wrangler.ogcm import llc as wr_llc
from wrangler.ogcm.preproc import gradb2_cutout, b_cutout 
from wrangler.ogcm.preproc import Fs_cutout
from wrangler.preproc import field as pp_field
from wrangler import defs as wr_defs
from wrangler import utils as wr_utils

from IPython import embed


def preproc_datetime(llc_table:pandas.DataFrame, field:str, udate:str, pdict:str, 
                     cutout_size=(64,64), fixed_km=None, n_cores=10, 
                     coords_ds=None, test_failures:bool=False, 
                     test_process:bool=False, debug=False):
    """Main routine to extract and pre-process LLC data for later SST analysis
    The llc_table is modified in place (and also returned).

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
    if field in ['SST','SSS','DivSST2','SSTK', 'SSSs', 'SSH']:
        map_fn = partial(pp_field.multi_process, pdict=pdict)
    elif field in ['Divb2']:
        map_fn = partial(gradb2_cutout, **pdict)
    elif field in ['b']:
        map_fn = partial(b_cutout, **pdict)
    elif field in ['Fs']:  # Frontogenesis tendency
        map_fn = partial(Fs_cutout, **pdict)
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
    elif field in ['SSH', 'SSHs']:
        data = ds.Eta.values
    elif field in ['SSS', 'SSSs']:
        data = ds.Salt.values
    elif field in ['Divb2', 'b']:
        data = ds.Theta.values
        data2 = ds.Salt.values
    elif field == 'Fs':
        data = ds.U.values
        data2 = ds.V.values
        data3 = ds.Theta.values
        data4 = ds.Salt.values
    else:
        raise IOError(f"Not ready for this field {field}")

    # Parse 
    sub_UID = llc_table.UID.values

    # Load up the cutouts
    fields, fields2, fields3, fields4, smooth_pixs = [], [], []
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
    print("Cutouts loaded for {}".format(filename))

    # Prep items
    zipitems = [fields]
    for fieldsN in [fields2, fields3, fields4]:
        if fieldsN is not None:
            zipitems.append(fieldsN)
    zipitems.append(sub_UID)
    if 'smooth_km' in pdict.keys():
        zipitems.append(smooth_pixs)
    items = [item for item in zip(*zipitems)]

    # Test processing
    if test_process:
        embed(header='extract.py/preproc_field 145')
        idx = 50
        img, tmeta = process.preproc_field(fields[idx], None, **pdict)
        #img, iidx, tmeta = po_fronts.anly_cutout(
        #    items[idx], **pdict)
        '''
        # Smoothing
        img, tmeta = process.preproc_field(fields[idx], None,
                                smooth_pix=smooth_pixs[idx], 
                                **pdict)
        # 
        '''
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(8,8))
        plt.clf()
        plt.imshow(img, origin='lower')
        plt.show()

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
    tbl_idx = wr_utils.match_ids(img_UID[good_idx], tbl_UID, require_in_match=True)

    for key in good_meta.keys():
        final_meta[key] = np.zeros(len(ppf_idx))
        #
        final_meta.loc[tbl_idx, key] = good_meta[key].values

    # Return
    return success, pp_fields, final_meta, filename



def extract_llc(llc_table:pandas.DataFrame, aios_ds, pp_dict:dict,
                out_file:str, zarr_path:str=None,
                n_cores:int=10, debug:bool=True):
    """Main routine to extract and pre-process LLC data for later analysis

    The llc_table is modified in place (and also returned).

    Args:
        llc_table (pandas.DataFrame): cutout table
            Requires 'row', 'col', 'datetime' columns
        aios_ds (object): Dataset object with field information
        pp_dict (dict): Preprocessing dictionary with keys:
        n_cores (int, optional): Number of cores for multi-processing. Defaults to 10.
        zarr_path (str, optional): Path to zarr files. Defaults to None.
        debug (bool, optional): Debug mode. Defaults to True.

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: Modified in place table

    """
    raise DeprecationWarning("Use fronts.llc.extract.preproc_field instead")
    # Prepping
    for key in ['filename', 'pp_file']:
        if key not in llc_table.keys():
            llc_table[key] = ''
    llc_table['pp_idx'] = -1

    if zarr_path is not None:
        ds_zarr = xarray.open_zarr(zarr_path, consolidated=False)
        face = 1
        t0 = pandas.Timestamp('2011-09-13T00:00:00')

    # Load coords?
    fixed_km = None if 'fixed_km' not in pp_dict.keys() else pp_dict['fixed_km']
    if fixed_km is not None:
        coords_ds = wr_llc.load_coords()
        R_earth = 6371. # km
        circum = 2 * np.pi* R_earth
        km_deg = circum / 360.
    
    # Setup for parallel
    map_fn = partial(pp_field.multi_process, pdict=pp_dict)

    '''
    # Kinematics
    if calculate_kin:
        if kin_stat_dict is None:
            raise IOError("You must provide kin_stat_dict with calculate_kin")
        # Prep
        if 'calc_FS' in kin_stat_dict.keys() and kin_stat_dict['calc_FS']:
            map_kin = partial(kinematics.cutout_kin, 
                         kin_stats=kin_stat_dict,
                         extract_kin=extract_kin,
                         field_size=field_size[0])
    '''

    # Setup for dates
    uni_date = np.unique(llc_table.datetime)

    # Init
    pp_fields, meta, img_idx, all_sub = [], [], [], []
    '''
    if calculate_kin:
        kin_meta = []
    else:
        kin_meta = None
    if extract_kin:  # Cutouts of kinematic information
        Fs_fields, divb_fields = [], []
    '''

    # Prep LLC Table
    #llc_table = pp_utils.prep_table_for_preproc(
    #    llc_table, preproc_root, field_size=field_size)
    
    # Loop
    if debug:
        uni_date = uni_date[0:1]

    for udate in uni_date:

        # Data format
        if zarr_path is not None:
            ts = pandas.Timestamp(udate)
            # Convert date into time index
            dt = ts - t0
            time = int(dt / pandas.Timedelta(hours=1))
            # 
            variable = ds_zarr[aios_ds.variable].isel(time=time,face=face).values
        else:
            # Parse filename
            filename = wr_llc.grab_llc_datafile(udate)
            ds = xarray.open_dataset(filename)
            variable = ds[aios_ds.variable].values

        # Parse 
        gd_date = llc_table.datetime == udate
        sub_idx = np.where(gd_date)[0]
        all_sub += sub_idx.tolist()  # These really should be the indices of the Table
        coord_tbl = llc_table[gd_date]

        # Add to table
        llc_table.loc[gd_date, 'filename'] = filename

        # Load up the cutouts
        fields, rs, cs, drs = [], [], [], []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            if fixed_km is None:
                dr = pp_dict['field_size']
                dc = pp_dict['field_size']
            else:
                dlat_km = (coords_ds.lat.data[r+1,c]-coords_ds.lat.data[r,c]) * km_deg
                dr = int(np.round(fixed_km / dlat_km))
                dc = dr
                # Save for kinematics
                drs.append(dr)
                rs.append(r)
                cs.append(c)
            #
            if (r+dr >= variable.shape[0]) or (c+dc > variable.shape[1]):
                fields.append(None)
            else:
                fields.append(variable[r:r+dr, c:c+dc])
        print("Cutouts loaded for {}".format(filename))

        # Multi-process time
        # 
        items = [item for item in zip(fields,sub_idx)]

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))

        # Deal with failures
        answers = [f for f in answers if f is not None]
        cur_img_idx = [item[1] for item in answers]

        # Slurp
        pp_fields += [item[0] for item in answers]
        img_idx += cur_img_idx
        meta += [item[2] for item in answers]

        del answers, fields, items

        '''
        # Kinmatics
        if calculate_kin:
            # Assuming FS for now
            #if 'calc_FS' in kin_stat_dict.keys() and kin_stat_dict['calc_FS']:

            # Grab the data fields (~5 Gb RAM)
            U = ds.U.values
            V = ds.V.values
            Salt = ds.Salt.values

            # Build cutouts
            items = []
            print("Building Kinematic cutouts")
            for jj in cur_img_idx:
                # Re-index
                ii = np.where(sub_idx == jj)[0][0]
                # Saved
                r = rs[ii]
                c = cs[ii]
                dr = drs[ii]
                dc = dr
                #
                items.append(
                    (U[r:r+dr, c:c+dc],
                    V[r:r+dr, c:c+dc],
                    sst[r:r+dr, c:c+dc],
                    Salt[r:r+dr, c:c+dc],
                    jj)
                )

            # Process em
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_kin, items,
                                             chunksize=chunksize), total=len(items)))
            kin_meta += [item[1] for item in answers]
            if extract_kin:
                Fs_fields += [item[2] for item in answers]
                divb_fields += [item[3] for item in answers]
            del answers
        '''

        ds.close()
        #embed(header='extract 223')

    # Fuss with indices
    #ex_idx = np.array(all_sub)
    ppf_idx = np.array(img_idx)

    # There can be some loss due to preprocessing
    pp_fields = np.stack(pp_fields)

    #if debug:
    #    embed(header='206 of ogcm.py/extract_llc')

    # Cut the Table and re-order to images
    llc_table = llc_table.iloc[ppf_idx].copy()
    llc_table.reset_index(inplace=True, drop=True)

    # Fuss with indexing
    # Valid and train indices
    valid = llc_table.pp_type == wr_defs.tbl_dmodel['pp_type']['valid']
    valid_idx = llc_table.index[valid].to_numpy()
    llc_table.loc[valid_idx, 'pp_idx'] = np.arange(np.sum(valid))

    train = llc_table.pp_type == wr_defs.tbl_dmodel['pp_type']['train']
    train_idx = llc_table.index[train].to_numpy()
    llc_table.loc[train_idx, 'pp_idx'] = np.arange(np.sum(train))

    # ###################
    # Write to disk (avoids holding another 20Gb in memory)
    print("Writing: {}".format(out_file))
    with h5py.File(out_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=pp_fields[valid_idx].astype(np.float32))
        # Metadata
        #dset = f.create_dataset('valid_metadata', data=main_tbl.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        #dset.attrs['columns'] = clms
        # Train
        f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
        #dset = f.create_dataset('train_metadata', data=main_tbl.iloc[train_idx].to_numpy(dtype=str).astype('S'))
        #dset.attrs['columns'] = clms
    print("Wrote: {}".format(out_file))

    '''
    # Write kin?
    if extract_kin:
        # F_s
        Fs_local_file = local_file.replace('.h5', '_Fs.h5')
        pp_utils.write_extra_fields(Fs_fields, llc_table, Fs_local_file)
        # divb
        divb_local_file = local_file.replace('.h5', '_divb.h5')
        pp_utils.write_extra_fields(divb_fields, llc_table, divb_local_file)
    '''
    
    # Return
    return llc_table 