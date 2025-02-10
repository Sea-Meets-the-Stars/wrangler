Data Model
==========

This document describes the data model used in wrangler for satellite data processing and analysis. The model is defined in ``wrangler.defs``.

Required Fields
-------------

The following fields must be included in your data tables:

* **lat** : float
    Latitude of the center of the cutout (degrees)
* **lon** : float
    Longitude of the center of the cutout (degrees)
* **datetime** : pandas.Timestamp
    Timestamp of the cutout

Field Parameters
--------------

* **field_size** : int
    Size of the cutout side (pixels)
* **col** : int
    Column of lower-left corner of the cutout in the granule
* **row** : int
    Row of lower-left corner of the cutout in the granule

File Information
--------------

* **filename** : str
    Filename of the original data file from which the cutout was extracted
* **ex_filename** : str
    Filename of the extraction file holding the cutouts

Data Quality Metrics
-----------------

* **clear_fraction** : float
    Fraction of the cutout clear from clouds
* **LL** : float
    Log-likelihood of the cutout from Ulmo

Temperature Statistics
-------------------

* **mu** : float
    Average SSHa of the cutout
* **Tmin** : float
    Minimum temperature of the cutout (°C)
* **Tmax** : float
    Maximum temperature of the cutout (°C)
* **T10** : float
    10th percentile of temperature of the cutout (°C)
* **T90** : float
    90th percentile of temperature of the cutout (°C)
* **DT** : float
    Temperature difference metric of the cutout (°C)
* **DT40** : float
    DT for inner 40x40 pixels

Processing Information
--------------------

* **pp_root** : str
    Describes the pre-processing steps applied
* **pp_file** : str
    Filename of the pre-processed file holding the cutout
* **pp_idx** : int
    Index describing position of the cutout in the pp_file
* **pp_type** : int
    Type indicator for the cutout:
        * -1: illdefined
        * 0: valid
        * 1: train

Usage Example
-----------

Here's an example of creating a data table with the required fields:

.. code-block:: python

    import pandas as pd
    import numpy as np
    
    # Create a sample data table
    data = {
        'lat': 32.5,
        'lon': -117.8,
        'datetime': pd.Timestamp('2024-01-01 12:00:00'),
        'field_size': 192,
        'clear_fraction': 0.95,
        'Tmin': 15.2,
        'Tmax': 18.7,
        'T10': 15.8,
        'T90': 18.2
    }
    
    df = pd.DataFrame([data])

Validation
---------

You can validate your data table using the utils module:

.. code-block:: python

    from wrangler.tables import utils as tbl_utils
    
    # Validate the table
    is_valid = tbl_utils.vet_main_table(df)
    if not is_valid:
        print("Table is invalid - missing required fields")

Data Types
---------

The data model supports the following data types:

* Numeric fields: Both Python native types and numpy numeric types (float, int)
* String fields: Python strings
* Datetime fields: pandas.Timestamp objects
* Boolean fields: Python booleans (True/False)

Notes
----

* The required fields (lat, lon, datetime) must be present in all data tables
* Temperature values are stored in degrees Celsius
* The pp_type field is used differently in different contexts:
    - In Ulmo: 1 for subset of training, 0 for the rest
    - In SSL: 1 for train, 0 for validation, -1 for the rest
* Field_size refers to the pixel dimensions of the cutout and should match the extraction parameters
* Clear_fraction should be between 0 and 1, representing the fraction of non-cloudy pixels