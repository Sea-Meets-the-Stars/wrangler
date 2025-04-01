The grab_and_go Module
==================

The ``grab_and_go`` module provides a streamlined interface for downloading and processing satellite data in a single operation. It handles the entire workflow from retrieving data from remote servers to extracting fields of interest and saving them to disk.

Functions
--------

.. function:: grab(aios_ds, t0, t1, verbose=True, skip_download=False)

   Retrieves files from a data source within a given time range.

   :param aios_ds: Dataset object containing source and collection details
   :type aios_ds: AIOS_DataSet
   :param t0: Start time
   :type t0: str or datetime
   :param t1: End time
   :type t1: str or datetime
   :param verbose: Enable verbose output
   :type verbose: bool
   :param skip_download: Skip the download step (for testing)
   :type skip_download: bool
   :return: List of local file paths if files are downloaded, otherwise None
   :rtype: list or None
   :raises ValueError: If the data source is not supported

.. function:: extract(aios_ds, local_files, exdict, n_cores, debug=False, single=False, verbose=True)

   Extracts data from local files using specified extraction parameters.

   :param aios_ds: Dataset object containing field information
   :type aios_ds: AIOS_DataSet
   :param local_files: Path to local files to process
   :type local_files: list
   :param exdict: Dictionary of extraction parameters
   :type exdict: dict
   :param n_cores: Number of cores to use for multiprocessing
   :type n_cores: int
   :param debug: Enable debugging mode
   :type debug: bool
   :param single: Enable single process mode
   :type single: bool
   :param verbose: Enable verbose output
   :type verbose: bool
   :return: Tuple of fields, inpainted masks, metadata, and times
   :rtype: tuple
   :raises ValueError: If the dataset field is not supported

.. function:: run(dataset, tstart, tend, eoption_file, ex_file, tbl_file, n_cores, tdelta={'days':1}, verbose=True, debug=False, debug_noasync=False, save_local_files=False)

   Complete end-to-end pipeline to grab and extract data from a dataset.

   :param dataset: Name of the dataset (e.g., 'VIIRS_NPP')
   :type dataset: str
   :param tstart: Start time in ISO format (e.g., '2020-01-01')
   :type tstart: str
   :param tend: End time in ISO format
   :type tend: str
   :param eoption_file: Filename of extraction options
   :type eoption_file: str
   :param ex_file: Output HDF5 filename for extracted data
   :type ex_file: str
   :param tbl_file: Output parquet filename for metadata
   :type tbl_file: str
   :param n_cores: Number of cores to use
   :type n_cores: int
   :param tdelta: Time delta for processing chunks
   :type tdelta: dict
   :param verbose: Enable verbose output
   :type verbose: bool
   :param debug: Enable debug mode
   :type debug: bool
   :param debug_noasync: Debug without async
   :type debug_noasync: bool
   :param save_local_files: Keep downloaded files after processing
   :type save_local_files: bool
   :return: None

Extraction Parameters
-------------------

The extraction options file (``eoption_file``) should be a JSON file with the following parameters:

* ``field_size`` (int): Size of the field to extract in pixels
* ``clear_threshold`` (float): Percentage threshold for clear conditions
* ``nadir_offset`` (int): Offset from nadir in pixels
* ``temp_bounds`` (list): Temperature bounds [min, max] in degrees Celsius
* ``nrepeat`` (int): Number of repetitions for extraction
* ``sub_grid_step`` (int): Step size for sub-grid extraction
* ``grow_mask`` (bool): Whether to grow the cloud mask
* ``inpaint`` (bool): Whether to perform inpainting on masked regions

Example Usage
-----------

Basic usage with VIIRS NPP data:

.. code-block:: python

    import asyncio
    from wrangler.grab_and_go import run
    
    # Define extraction options file
    extract_file = 'extract_viirs_std.json'
    
    # Run the pipeline to download and process data
    run(
        dataset='VIIRS_NPP',          # Dataset name
        tstart='2024-01-01',          # Start date
        tend='2024-01-02',            # End date
        eoption_file=extract_file,    # Extraction options
        ex_file='output.h5',          # Output data file
        tbl_file='metadata.parquet',  # Output metadata file
        n_cores=4                     # Number of processing cores
    )

Handling Larger Time Periods
-------------------------

For processing larger time periods efficiently:

.. code-block:: python

    import pandas as pd
    from datetime import timedelta
    from wrangler.grab_and_go import run
    
    # Process one week at a time
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-01-31')
    
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + timedelta(days=7)
        
        # Ensure we don't go past the end date
        if next_date > end_date:
            next_date = end_date
        
        # Process this time chunk
        run(
            dataset='VIIRS_NPP',
            tstart=current_date.isoformat(),
            tend=next_date.isoformat(),
            eoption_file='extract_viirs_std.json',
            ex_file=f'viirs_{current_date.strftime("%Y%m%d")}.h5',
            tbl_file=f'viirs_meta_{current_date.strftime("%Y%m%d")}.parquet',
            n_cores=4
        )
        
        current_date = next_date

Output Structure
--------------

The extraction process produces two main outputs:

1. HDF5 File (``ex_file``)
   - ``fields``: Extracted field data (n_fields × field_size × field_size)
   - ``inpainted_masks``: Inpainted mask data
   - ``metadata``: Array of metadata for each field

2. Parquet File (``tbl_file``)
   - Contains all metadata in tabular format:
     - ``filename``: Original source file
     - ``row``, ``col``: Position in the original granule
     - ``lat``, ``lon``: Geographic coordinates
     - ``clear_fraction``: Fraction of clear pixels
     - ``field_size``: Size of the extracted field
     - ``datetime``: Timestamp of the data
     - ``ex_filename``: Path to the extraction file

Notes
-----

* Currently supports PODAAC data sources
* Only SST (Sea Surface Temperature) fields are supported
* Uses multiprocessing for parallel extraction of fields
* Automatically validates the metadata table before saving
