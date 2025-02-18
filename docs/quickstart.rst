Quick Start Guide
================

This guide will help you get started with wrangler, demonstrating the basic workflow for downloading and processing satellite data.

Basic Usage
----------

Loading a Dataset
^^^^^^^^^^^^^^

Start by importing and loading your desired dataset:

.. code-block:: python

    from wrangler.datasets.loader import load_dataset
    
    # Load VIIRS NPP dataset
    viirs_npp = load_dataset('VIIRS_NPP')

Download and Process Data
^^^^^^^^^^^^^^^^^^^^^^

The main workflow combines downloading and processing using the grab_and_go module:

.. code-block:: python

    import asyncio
    from wrangler.grab_and_go import run
    
    # Define your extraction options file
    extract_file = 'extract_viirs_std.json'
    
    # Run the pipeline
    asyncio.run(run(
        dataset='VIIRS_NPP',          # Dataset name
        tstart='2024-01-01',          # Start date
        tend='2024-01-02',            # End date
        eoption_file=extract_file,    # Extraction options
        ex_file='output.h5',          # Output HDF5 file
        tbl_file='metadata.parquet',  # Output metadata file
        n_cores=4                     # Number of processing cores
    ))

Extraction Configuration
^^^^^^^^^^^^^^^^^^^^^

Create an extraction options JSON file (e.g., 'extract_viirs_std.json'):

.. code-block:: json

    {
        "field_size": 192,
        "clear_threshold": 5,
        "nadir_offset": 0,
        "temp_bounds": [-3, 34],
        "nrepeat": 1,
        "sub_grid_step": 4,
        "inpaint": true
    }

Working with Processed Data
------------------------

Reading the Output
^^^^^^^^^^^^^^^

After processing, you can work with the output files:

.. code-block:: python

    import h5py
    import pandas as pd
    
    # Read the HDF5 file
    with h5py.File('output.h5', 'r') as f:
        # Access the fields
        fields = f['fields'][:]
        masks = f['inpainted_masks'][:]
    
    # Read the metadata
    metadata = pd.read_parquet('metadata.parquet')

Visualizing Fields
^^^^^^^^^^^^^^^

Use the cutout module to visualize processed fields:

.. code-block:: python

    import numpy as np
    from wrangler.cutout import show_image
    
    # Display a single field
    show_image(fields[0], cbar=True, clbl='Temperature (°C)')

Advanced Usage
------------

Manual Download and Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need more control over the pipeline, you can separate the download and processing steps:

.. code-block:: python

    import asyncio
    from wrangler.grab_and_go import grab, extract
    
    # First, download the files
    local_files = await grab(viirs_npp, '2024-01-01', '2024-01-02')
    
    # Then process them
    fields, masks, metadata, times = await extract(
        viirs_npp,
        local_files,
        extract_options,
        n_cores=4
    )

Field Preprocessing
^^^^^^^^^^^^^^^

For custom preprocessing of fields:

.. code-block:: python

    from wrangler.field import main as process_field
    
    # Process a single field
    processed_field, meta = process_field(
        field,
        mask,
        inpaint=True,
        median=True,
        med_size=(3,1),
        downscale=True,
        dscale_size=(2,2)
    )

Common Patterns
-------------

1. Quality Control
^^^^^^^^^^^^^^^

Filter data based on quality thresholds:

.. code-block:: python

    # Filter by clear fraction
    good_data = metadata[metadata['clear_fraction'] > 0.95]

2. Geographic Selection
^^^^^^^^^^^^^^^^^^^

Select data from specific regions:

.. code-block:: python

    # Filter by latitude/longitude
    region_data = metadata[
        (metadata['lat'].between(32, 40)) &
        (metadata['lon'].between(-128, -118))
    ]

3. Batch Processing
^^^^^^^^^^^^^^^^

Process multiple time periods:

.. code-block:: python

    from datetime import datetime, timedelta
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    # Process one day at a time
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)
        asyncio.run(run(
            dataset='VIIRS_NPP',
            tstart=current_date.strftime('%Y-%m-%d'),
            tend=next_date.strftime('%Y-%m-%d'),
            eoption_file='extract_viirs_std.json',
            ex_file=f'output_{current_date.strftime("%Y%m%d")}.h5',
            tbl_file=f'metadata_{current_date.strftime("%Y%m%d")}.parquet',
            n_cores=4
        ))
        current_date = next_date

Tips and Best Practices
--------------------

1. Memory Management
   - Process data in smaller time chunks for large datasets
   - Use the `n_cores` parameter appropriately for your system
   - Clean up downloaded files when no longer needed

2. Quality Control
   - Always check the clear_fraction in the metadata
   - Verify temperature bounds are appropriate for your region
   - Inspect inpainted masks for data quality

3. Performance
   - Use multiple cores for processing when available
   - Consider downscaling for large datasets
   - Use appropriate batch sizes for your memory constraints

Next Steps
---------

- Explore the API documentation for more detailed information
- Check out the example notebooks in the repository
- Join the community and contribute to the project