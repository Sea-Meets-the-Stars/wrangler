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

async def async_grab(aios_ds, t0, t1, verbose:bool=True, skip_download:bool=False):
    pass

def grab(aios_ds, t0, t1, verbose:bool=True, skip_download:bool=False):
    """
    Asynchronously grabs and optionally downloads files from a specified data source within a given time range.

    Parameters:
        aios_ds (object): An object containing dataset information, including the source and collection details.
        t0 (datetime): The start time of the time range for which files are to be grabbed.
        t1 (datetime): The end time of the time range for which files are to be grabbed.
        verbose (bool, optional): If True, enables verbose output. Defaults to True.
        skip_download (bool, optional): If True, skips the download step (useful for testing). Defaults to False.

    Returns:
        list or None: A list of local file paths if files are downloaded, otherwise None.

    Raises:
        ValueError: If the data source specified in aios_ds is not supported.
    """

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


def extract(aios_ds, local_files:str,
                  exdict:dict, n_cores:int,
                  debug:bool=False,
                  single:bool=False,
                  verbose:bool=True):
    """
    Extracts data from local files using the specified parameters.

    Parameters:
        aios_ds (object): The dataset object containing the field information.
        local_files (str): Path to the local files to be processed.
        exdict (dict): Dictionary containing extraction parameters.
            - field_size (int): Size of the field to be extracted.
            - clear_threshold (float): Threshold for clear conditions.
            - nadir_offset (float): Offset for nadir.
            - temp_bounds (tuple): Temperature bounds for extraction.
            - nrepeat (int): Number of repetitions for extraction.
            - sub_grid_step (int): Step size for sub-grid extraction.
            - inpaint (bool): Flag to indicate if inpainting is required.
        n_cores (int): Number of cores to use for multiprocessing.
        debug (bool, optional): Flag to enable debugging mode. Defaults to False.
        single (bool, optional): Flag to enable single process mode. Defaults to False.
        verbose (bool, optional): Flag to enable verbose output. Defaults to True.

    Returns:
        tuple: A tuple containing the following elements:
            - fields (np.ndarray): Extracted fields.
            - inpainted_masks (np.ndarray): Inpainted masks.
            - metadata (np.ndarray): Metadata associated with the extracted fields.
            - times (np.ndarray): Timestamps of the extracted fields.

    Raises:
        ValueError: If the dataset field is not 'SST'.
    """

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


    if debug and single:
        # Single process
        local_file = local_files[0]
        items = map_fn(local_file)
        import pdb; pdb.set_trace()

    # Multi-process on ex_sst.extract_file
    metadata = None
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(local_files) // n_cores if len(local_files) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, local_files,
                                            chunksize=chunksize), 
                            total=len(local_files)))
    # Trim None's
    answers = [f for f in answers if f[0] is not None]

    # Unpack
    #if debug:
    #    import pdb; pdb.set_trace()
    fields = np.concatenate([item[0] for item in answers])
    inpainted_masks = np.concatenate([item[1] for item in answers])
    metadata = np.concatenate([item[2] for item in answers])

    # Make the same size as fields
    times = np.concatenate([np.array([item[3]]*item[0].shape[0], dtype='datetime64[ns]') for item in answers]) 

    # Return
    return fields, inpainted_masks, metadata, times

async def async_run(dataset:str, tstart, tend, eoption_file:str,
              ex_file:str, tbl_file:str,
              n_cores:int, tdelta:dict={'days':1}, 
              verbose:bool=True, debug:bool=False, 
              debug_noasync:bool=False,
              save_local_files:bool=False):
    return run(dataset, tstart, tend, eoption_file, ex_file, tbl_file, n_cores, tdelta, verbose, debug, debug_noasync, save_local_files)

def run(dataset:str, tstart, tend, eoption_file:str,
              ex_file:str, tbl_file:str,
              n_cores:int, tdelta:dict={'days':1}, 
              verbose:bool=True, debug:bool=False, 
              debug_noasync:bool=False,
              save_local_files:bool=False):
    """ Grab and extract data from a dataset in one go

    The code will grab the data from the dataset between tstart and tend
    in time increments of tdelta. The data will be extracted using the
    extraction options in eoption_file. The extracted data will be saved
    in ex_file and the metadata will be saved in tbl_file. The number of
    cores to use is n_cores.

    Outputs:
    - ex_file: HDF5 file with the extracted data
    - tbl_file: Parquet file with the metadata
    

    Args:
        dataset (str): Name of the dataset
            e.g. 'VIIRS_NPP'
        tstart (str): Start time in ISO format
            e.g. '2020-01-01T00:00:00'
        tend (str): End time in ISO format
        eoption_file (str): Filename of the extraction options
        ex_file (str): Filename of the extraction file
            e.g. 'ex_VIIRS_NPP_2020.h5'
        tbl_file (str): Filename of the table file
        n_cores (int): Number of cores to use
        tdelta (dict, optional): Time delta. Defaults to {'days':1}.
        verbose (bool, optional): Print verbose output. Defaults to True.
        debug (bool, optional): Debug mode. Defaults to False.
        debug_noasync (bool, optional): Debug without async. Defaults to False.
        save_local_files (bool, optional): Save local files. Defaults to False.
            These are the files downloaded from the remote server

    Returns:
        None
    """

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
    second_time = False
    time_to_break = False
    while True:  # yes this is crazy
        # Increment
        t1 = t0 + tdelta

        # Convert to ISO
        t0s = t0.isoformat()
        t1s = t1.isoformat()

        if verbose:
            print(f"Working on {t0}")

        # Grab 
        if t1 <= tend:
            local_files = grab(aios_ds, t0s, t1s)
        else:
            time_to_break = True

        # Hold the local_files for removing
        previous_local_files = [ifile for ifile in local_files]


        print("Starting extraction")
        fields, inpainted_masks, imetadata, itimes  = extract(aios_ds, local_files,
                                            exdict, n_cores, debug=debug)
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
            metadata = np.concatenate([metadata, imetadata])
            second_time = True
    
        # Delete the local files
        if not save_local_files:
            for local_file in previous_local_files:
                os.remove(local_file)

        # Process
        if time_to_break:
            break

        # Increment
        t0 += tdelta

        if debug and second_time:
            break

    # Finish
    # Metadata
    #if debug:
    #    embed(header='277 of grab_and_go')
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']
    dset = f_h5.create_dataset('metadata', data=np.concatenate(metadata).astype('S'))
    dset.attrs['columns'] = columns
    # Close
    f_h5.close() 

    # Table time
    table = pandas.DataFrame()
    table['filename'] = [item[0] for item in metadata]
    table['row'] = [int(item[1]) for item in metadata]
    table['col'] = [int(item[2]) for item in metadata]
    table['lat'] = [float(item[3]) for item in metadata]
    table['lon'] = [float(item[4]) for item in metadata]
    table['clear_fraction'] = [float(item[5]) for item in metadata]
    table['field_size'] = exdict['field_size']

    # Time
    #if debug:
    #    import pdb; pdb.set_trace()
    table['datetime'] = np.concatenate(times)
    # Output filename
    table['ex_filename'] = ex_file

    # Vet
    assert tbl_utils.vet_main_table(table)

    # Final write
    tbl_io.write_main_table(table, tbl_file)
    