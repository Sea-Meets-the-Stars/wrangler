""" Routines to grab and extract dataasets in one go """

import os
import asyncio
import pandas
import datetime

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import subprocess
from tqdm import tqdm

from remote_sensing.download import podaac

from wrangler.datasets.loader import load_dataset
from wrangler.extract import sst as ex_sst
from wrangler.extract import io as ex_io

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
                  exdict:dict,
                  debug:bool=False,
                  verbose:bool=True):

    if aios_ds.field == 'SST':
        map_fn = partial(ex_sst.extract_file,
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
        for local_file in local_files:
            import pdb; pdb.set_trace()
            map_fn(aios_ds, local_file)

    # Multi-process
    for local_file in local_files:
            ex_sst.extract_file(aios_ds, local_file)

async def run(dataset:str, tstart, tend, eoption_file:str,
              outfile:str, tdelta:dict={'days':1}, 
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
            await iproc
            # Delete the local files
            if not save_local_files:
                for local_file in previous_local_files:
                    os.remove(local_file)

        # Hold the local_files for removing
        previous_local_files = [ifile for ifile in local_files]

        # Process
        iproc = asyncio.create_task(extract(aios_ds, local_files,
                                            exdict))

        # Increment
        t0 += tdelta