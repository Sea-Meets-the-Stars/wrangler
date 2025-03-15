""" Define data model, options, etc. for Nenya models and analysis"""

import numpy as np
import pandas

# Wrangler options

# Wrangle extraction options

ex_dmodel = {
    'field_size': dict(dtype=(int, np.integer),
                help='Size of the cutout side (pixels)'),
    'clear_threshold': dict(dtype=(int, np.integer),
                help='Threshold for clear fraction (percent)'),
    'nadir_offset': dict(dtype=(int, np.integer),
                help='Offset from nadir for cutout center'),
    'temp_bounds': dict(dtype=(float,np.floating),
                help='Temperature bounds for cutout'),
    'nrepeat': dict(dtype=(int, np.integer),
                help='Number of times to repeat extraction'),
    'sub_grid_step': dict(dtype=(int, np.integer),
                help='Fraction of field-size to use for sub-gridding'),
    'inpaint': dict(dtype=bool,
                help='Inpaint the cutout'),
}
    

# Wrangler table data model
tbl_dmodel = {
    'field_size': dict(dtype=(int, np.integer),
                help='Size of the cutout side (pixels)'),
    'lat': dict(dtype=(float,np.floating),
                help='Latitude of the center of the cutout (deg)'),
    'lon': dict(dtype=(float,np.floating),
                help='Longitude of the center of the cutout (deg)'),
    'col': dict(dtype=(int, np.integer),
                help='Column of lower-left corner of the cutout in the granule'),
    'row': dict(dtype=(int, np.integer),
                help='Row of lower-left corner of the cutout in the granule'),
    'filename': dict(dtype=str,
                help='Filename of the original data file from which the cutout was extracted'),
    'ex_filename': dict(dtype=str,
                help='Filename of the extraction file holding the cutouts'),
    'datetime': dict(dtype=pandas.Timestamp,
                help='Timestamp of the cutout'),
    'LL': dict(dtype=(float,np.floating),
                help='Log-likelihood of the cutout from Ulmo'),
    'clear_fraction': dict(dtype=float,
                help='Fraction of the cutout clear from clouds'),
    'mu': dict(dtype=(float,np.floating),
                help='Average SSHa of the cutout'),
    'Tmin': dict(dtype=(float,np.floating),
                help='Minimum T of the cutout (C deg)'),
    'Tmax': dict(dtype=(float,np.floating),
                help='Maximum T of the cutout (C deg)'),
    'T10': dict(dtype=(float,np.floating),
                help='10th percentile of T of the cutout (C deg)'),
    'T90': dict(dtype=(float,np.floating),
                help='90th percentile of T of the cutout'),
    'DT': dict(dtype=(float,np.floating),
                help='90th percentile of T of the cutout (C deg)'),
    'DT40': dict(dtype=(float,np.floating),
                help='DT for inner 40x40 pixels'),
    'pp_root': dict(dtype=str,
                help='Describes the pre-processing steps applied'),
    'pp_file': dict(dtype=str,
                help='Filename of the pre-processed file holding the cutout'),
    'pp_idx': dict(dtype=(int,np.integer), 
                help='Index describing position of the cutout in the pp_file'),
    'pp_type': dict(dtype=(int, np.integer), allowed=(-1, 0,1), 
                    valid=0, train=1, init=-1,
                    help='-1: illdefined, 0: valid, 1: train'),
                    # In Ulmo, we use 1 for the subset of training and 0 for the rest
                    # In SSL, we use 1 for train, 0 for validation and -1 for the rest [but not always]
    'images_file': dict(dtype=str,
                help='Name of the images file, likely hdf5'),
    'data_folder': dict(dtype=str,
                help='Path to the images_file'),
    'train_key': dict(dtype=str,
                help='Dataset for training'),
    'valid_key': dict(dtype=str,
                help='Dataset for validation'),
    's3_outdir': dict(dtype=str,
                help='s3 bucket+path for model output'),
    # REQUIRED
    'required': ('lat', 'lon', 'datetime')
}
