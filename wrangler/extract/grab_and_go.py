""" Routines to grab and extract dataasets in one go """

import os
import asyncio
import datetime

import numpy as np
import pandas

import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from remote_sensing.download import podaac

from wrangler.datasets.loader import load_dataset
from wrangler.extract import sst as ex_sst
from wrangler.extract import io as ex_io
from wrangler.tables import utils as tbl_utils
from wrangler.tables import io as tbl_io

from IPython import embed

async def grab(aios_ds, t0, t1,
               verbose:bool=True,
               skip_download:bool=False):

    local_files = None
    if aios_ds.source == 'PODAAC':
        # Grab the file list
        files, _ = podaac.grab_file_list(
            aios_ds.podaac_collection, 
            time_range=(t0, t1),
            verbose=verbose)

        # Download
        if not skip_download:  # for testing
            local_files = podaac.download_files(files, verbose=verbose)
    else:
        raise ValueError("Only PODAAC datasets supported")

    # Return
    return local_files

async def extract(aios_ds, local_files:str,
                  exdict:dict, n_cores:int,
                  debug:bool=False,
                  verbose:bool=True):

    if aios_ds.field == 'SST':
        map_fn = partial(ex_sst.extract_file,
                     aios_ds=aios_ds,
                     field_size=(exdict['field_size'], exdict['field_size']),
                     CC_max=1.-exdict['clear_threshold'] / 100.,
                     nadir_offset=exdict['nadir_offset'],
                     temp_bounds=tuple(exdict['temp_bounds']),
                     nrepeat=exdict['nrepeat'],
                     sub_grid_step=exdict['sub_grid_step'],
                     inpaint=exdict['inpaint'])
    else:
        raise ValueError("Only SST datasets supported so far")


    if debug:
        # Single process
        local_file = local_files[0]
        items = map_fn(local_file)
        import pdb; pdb.set_trace()

    # Multi-process
    metadata = None
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(local_files) // n_cores if len(local_files) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, local_files,
                                            chunksize=chunksize), 
                            total=len(local_files)))
    # Trim None's
    answers = [f for f in answers if f is not None]
    # Unpack
    fields = np.concatenate([item[0] for item in answers])
    inpainted_masks = np.concatenate([item[1] for item in answers])
    metadata = np.concatenate([item[2] for item in answers])
    times = np.concatenate([item[3] for item in answers])

    # Return
    return fields, inpainted_masks, metadata, times

async def run(dataset:str, tstart, tend, eoption_file:str,
              ex_file:str, tbl_file:str,
              n_cores:int, tdelta:dict={'days':1}, 
              verbose:bool=True, debug:bool=False, 
              save_local_files:bool=False):

    # Load options
    exdict = ex_io.load_options(eoption_file)

    # Convert tstart, tend to datetime
    tstart = pandas.to_datetime(tstart)
    tend = pandas.to_datetime(tend)
    tdelta = pandas.to_timedelta(datetime.timedelta(**tdelta))

    # Instantiate the AIOS_DataSet
    aios_ds = load_dataset(dataset)

    # Begin downloading
    t0 = tstart
    iproc = None

    # Local file for writing
    f_h5 = h5py.File(ex_file, 'w')
    print("Opened local file: {}".format(ex_file))

    times = []
    first_time = True
    while t0 < tend:
        # Increment
        t1 = t0 + tdelta

        # Convert to ISO
        t0s = t0.isoformat()
        t1s = t1.isoformat()

        if verbose:
            print(f"Working on {t0}")

        # Start the grab asynchronous
        igrab = asyncio.create_task(grab(aios_ds, t0s, t1s))
        # Wait for it
        local_files = await igrab
        #import pdb; pdb.set_trace()
        #embed(header='104 of grab_and_go') 

        # Wait for the previous process to end
        if iproc is not None:
            fields, inpainted_masks, imetadata, itimes  = await iproc
            times.append(itimes)
            # Write
            if first_time:
                f_h5.create_dataset('fields', data=fields, 
                                    compression="gzip", chunks=True,
                                    maxshape=(None, fields.shape[1], fields.shape[2]))
                f_h5.create_dataset('inpainted_masks', data=inpainted_masks,
                                    compression="gzip", chunks=True,
                                    maxshape=(None, inpainted_masks.shape[1], inpainted_masks.shape[2]))
                metadata = imetadata
                first_time = False
            else:
                # Resize
                for key in ['fields', 'inpainted_masks']:
                    f_h5[key].resize((f_h5[key].shape[0] + fields.shape[0]), axis=0)
                # Fill
                f_h5['fields'][-fields.shape[0]:] = fields
                f_h5['inpainted_masks'][-fields.shape[0]:] = inpainted_masks
                metadata += imetadata
        
            # Delete the local files
            if not save_local_files:
                for local_file in previous_local_files:
                    os.remove(local_file)

        # Hold the local_files for removing
        previous_local_files = [ifile for ifile in local_files]

        # Process
        iproc = asyncio.create_task(extract(aios_ds, local_files,
                                            exdict, n_cores, debug=debug))

        # Increment
        t0 += tdelta

    # Finish
    # Metadata
    import pdb; pdb.set_trace()
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']
    dset = f_h5.create_dataset('metadata', data=np.concatenate(metadata).astype('S'))
    dset.attrs['columns'] = columns
    # Close
    f_h5.close() 

    # Table time
    viirs_table = pandas.DataFrame()
    viirs_table['filename'] = [item[0] for item in metadata]
    viirs_table['row'] = [int(item[1]) for item in metadata]
    viirs_table['col'] = [int(item[2]) for item in metadata]
    viirs_table['lat'] = [float(item[3]) for item in metadata]
    viirs_table['lon'] = [float(item[4]) for item in metadata]
    viirs_table['clear_fraction'] = [float(item[5]) for item in metadata]
    viirs_table['field_size'] = exdict['field_size']

    # Time
    viirs_table['datetime'] = times
    # Output filename
    viirs_table['ex_filename'] = ex_file

    # Vet
    assert tbl_utils.vet_main_table(viirs_table)

    # Final write
    tbl_io.write_main_table(viirs_table, tbl_file)
    