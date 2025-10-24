Wrangler Documentation
=====================

Welcome to the Wrangler documentation. Wrangler is a Python library for downloading, processing, and analyzing satellite data.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   grab_and_go
   field_preprocessing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/datamodel
   api/datasets
   api/extraction
   api/preprocessing
   api/tables
   api/visualization

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_extraction
   examples/preprocessing_pipeline
   examples/regional_analysis

Features
--------

* Downloading satellite data from various providers (PODAAC, etc.)
* Extracting fields of interest from large granules
* Processing and quality control of extracted fields
* Managing metadata in tabular format
* Visualization tools for satellite data

Installation
-----------

You can install Wrangler using pip:

.. code-block:: bash

    pip install wrangler

Requirements
-----------

* Python 3.8+
* numpy
* pandas
* h5py
* scipy
* scikit-image
* boto3 (for S3 access)
* tqdm

Indices and Tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
