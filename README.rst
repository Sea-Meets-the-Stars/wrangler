.. |forks| image:: https://img.shields.io/github/forks/AI-for-Ocean-Science/wrangler?style=social 
   :target: https://github.com/AI-for-Ocean-Science/wrangler

.. |stars| image:: https://img.shields.io/github/stars/AI-for-Ocean-Science/wrangler?style=social
   :target: https://github.com/AI-for-Ocean-Science/wrangler


Wrangler |forks| |stars|
========================

.. image:: https://readthedocs.org/projects/wrangler/badge/?version=latest
    :target: https://wrangler.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Wrangler is a Python package for efficiently downloading, processing, and analyzing satellite data, with a current focus on VIIRS Sea Surface Temperature (SST) data from PODAAC.

Key Features
-----------

* Download and process VIIRS satellite data from PODAAC
* Extract and preprocess Sea Surface Temperature (SST) fields
* Support for multiple VIIRS satellites (NPP, N20, N21)
* Robust data validation and quality control
* Efficient parallel processing capabilities
* Flexible I/O operations for various file formats

Quick Start
----------

Installation
^^^^^^^^^^^

Clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/AI-for-Ocean-Science/wrangler.git
    cd wrangler
    pip install -e .

Basic Usage
^^^^^^^^^^

Here's a simple example of downloading and processing VIIRS data:

.. code-block:: python

    import asyncio
    from wrangler.datasets.loader import load_dataset
    from wrangler.grab_and_go import run

    # Run the pipeline
    asyncio.run(run(
        dataset='VIIRS_NPP',          # Dataset name
        tstart='2024-01-01',          # Start date
        tend='2024-01-02',            # End date
        eoption_file='extract_viirs_std.json',  # Extraction options
        ex_file='output.h5',          # Output HDF5 file
        tbl_file='metadata.parquet',  # Output metadata file
        n_cores=4                     # Number of processing cores
    ))

Dependencies
-----------

Core Requirements:
* numpy
* pandas
* scipy
* scikit-image
* h5py
* tqdm
* boto3
* smart_open

Optional Requirements:
* seaborn
* matplotlib

Documentation
------------

Full documentation is available at `ReadTheDocs <https://wrangler.readthedocs.io/>`_.

Contributing
-----------

We welcome contributions! Here's how you can help:

1. Check for open issues or open a new issue to start a discussion
2. Fork the repository on GitHub
3. Write tests for new features
4. Write code
5. Send a pull request

Please make sure to update tests as appropriate and follow the existing coding style.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Citation
--------

If you use this software in your research, please cite:

.. code-block:: text

    @software{wrangler,
      author = {{AI for Ocean Science}},
      title = {Wrangler: A Python package for processing satellite data},
      url = {https://github.com/AI-for-Ocean-Science/wrangler},
      version = {0.1.0},
      year = {2024},
    }

Contact
-------

* Issue Tracker: https://github.com/AI-for-Ocean-Science/wrangler/issues
* Documentation: https://wrangler.readthedocs.io/

Acknowledgments
--------------

This project is developed and maintained by the AI for Ocean Science team.