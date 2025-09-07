import numpy as np

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from wrangler.extract import sst as ex_sst

def main(aios_ds, local_files:str,
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
                     grow_mask=exdict['grow_mask'],
                     nrepeat=exdict['nrepeat'],
                     sub_grid_step=exdict['sub_grid_step'],
                     inpaint=exdict['inpaint'])
    else:
        raise ValueError("Only SST datasets supported so far")


    #single=True
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
    if len(answers) == 0:
        return None, None, None, None

    fields = np.concatenate([item[0] for item in answers])
    inpainted_masks = np.concatenate([item[1] for item in answers])
    metadata = np.concatenate([item[2] for item in answers])

    # Make the same size as fields
    times = np.concatenate([np.array([item[3]]*item[0].shape[0], dtype='datetime64[ns]') for item in answers]) 

    # Return
    return fields, inpainted_masks, metadata, times