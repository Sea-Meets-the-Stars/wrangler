""" Routines to grab and extract dataasets in one go """

import asyncio
import pandas

from remote_sensing.download import podaac

from wrangler.datasets.loader import load_dataset
from wrangler.extract import sst as ex_sst

async def grab(aios_ds, t0, t1,
               verbose:bool=True):

    if aios_ds.source == 'PODAAC':
        # Grab the file list
        files, _ = podaac.grab_file_list(
            aios_ds.collection, 
            time_range=(t0, t1),
            verbose=verbose)

        # Download
        local_files = podaac.download_files(files, verbose=verbose)
    else:
        raise ValueError("Only PODAAC datasets supported")

    # Return
    return local_files

async def extract(aios_ds, local_files:str,
                  extract_options:dict={},
                  verbose:bool=True):

    if aios_ds.field == 'SST':
        for local_file in local_files:
            ex_sst.extract_file(aios_ds, local_file, **extract_options)
    else:
        raise ValueError("Only SST datasets supported so far")


async def run(dataset:str, tstart, tend, outfile:str,
        tdelta:dict={'days':1}, verbose:bool=True,
        debug:bool=False):

    # Convert tstart, tend to datetime
    tstart = pandas.to_datetime(tstart)
    tend = pandas.to_datetime(tend)
    tdelta = pandas.to_timedelta(**tdelta)

    # Instantiate the AIOS_DataSet
    aios_ds = load_dataset(dataset)

    # Begin downloading
    t0 = tstart
    iproc = None
    for step in range(99999999999):
        t1 = t0 + tdelta
        # Check
        if t0 > tend:
            break

        if verbose:
            print(f"Working on {t0}")

        # Wait for the process to finish
        if iproc is None:
            continue
        else:
            pass

        # Start the grab asynchronous
        igrab = asyncio.create_task(grab(aios_ds, t0, t1))
        # Wait for it
        local_files = await igrab

        # Wait for the previous process to end
        if iproc is not None:
            await iproc

        # Process
        iproc = asyncio.create_task(extract(local_files))

        # Increment
        t0 += tdelta
