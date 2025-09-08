""" Extraction methods for OGCM data, e.g. LLC4320 """

import numpy as np
import pandas
import xarray

import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from wrangler.ogcm import llc as wr_llc
from wrangler.preproc import field as pp_field
from wrangler import defs as wr_defs

from IPython import embed

def extract_llc(llc_table:pandas.DataFrame, aios_ds, pp_dict:dict,
                out_file:str,
                n_cores:int=10, debug:bool=True):
    """Main routine to extract and pre-process LLC data for later analysis

    The llc_table is modified in place (and also returned).

    Args:
        llc_table (pandas.DataFrame): cutout table
            Requires 'row', 'col', 'datetime' columns
        aios_ds (object): Dataset object with field information
        pp_dict (dict): Preprocessing dictionary with keys:
        n_cores (int, optional): Number of cores for multi-processing. Defaults to 10.

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: Modified in place table

    """
    # Prepping
    for key in ['filename', 'pp_file']:
        if key not in llc_table.keys():
            llc_table[key] = ''
    llc_table['pp_idx'] = -1

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
        # Parse filename
        filename = wr_llc.grab_llc_datafile(udate)

        # Allow for s3
        ds = xarray.open_dataset(filename)
        #ds = llc_io.load_llc_ds(filename, local=dlocal)

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

    if debug:
        embed(header='206 of ogcm.py/extract_llc')

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