""" Extraction routines for VIIRS """

import os
import numpy as np
from scipy.ndimage import uniform_filter

from remote_sensing.netcdf import sst as rs_nc_sst
from remote_sensing.netcdf import utils as rs_nc_utils

from wrangler.datasets.base import AIOS_DataSet
from wrangler.preproc import field as pp_field

#from ulmo.viirs import io as viirs_io 
#from ulmo.preproc import utils as pp_utils
#from ulmo.preproc import extract

from IPython import embed


def clear_grid(mask, field_size, method, CC_max=0.05,
                 nsgrid_draw=1, return_fracCC=False,
                 return_CC_mask=False,
                 sub_grid_step=4):
    """
    Identify grid locations in the granule where the cloud 
    fraction is satisfied 

    Parameters
    ----------
    mask : np.ndarray
    field_size : int
    method : str
        'random'
        'lower_corner'
    CC_max : float
        Maximum cloudy fraction allowed
    ndraw_mnx
    sub_grid_step : int, optional
        Fraction of field-size to use for sub-gridding
    nsgrid_draw : int, optional
        Number of fields to draw per sub-grid
    return_fracCC : bool, optional
        Return the fraction of the image satisfying the CC value
    return_CC_mask : bool, optional
        Simply Return the CC_mask

    Returns
    -------
    picked_row, picked_col, CC_mask[idx_clear][keep] : rows, columns, CC_fraction

    """
    # Some checks
    if nsgrid_draw > 1 and method == 'lower_corner':
        raise IOError("Not ready for this..")

    # Sum across the image
    CC_mask = uniform_filter(mask.astype(float), field_size, mode='constant', cval=1.)

    # Clear
    mask_edge = np.zeros_like(mask, dtype=bool)
    mask_edge[:field_size//2,:] = True
    mask_edge[-field_size//2:,:] = True
    mask_edge[:,-field_size//2:] = True
    mask_edge[:,:field_size//2] = True

    if return_CC_mask:
        return CC_mask, mask_edge

    # Evaluate
    clear = (CC_mask <= CC_max) & np.invert(mask_edge)  # Added the = sign on 2021-01-12
    if return_fracCC:
        return np.sum(clear)/((clear.shape[0]-field_size)*(clear.shape[1]-field_size))

    # Indices
    idx_clear = np.where(clear)
    nclear = idx_clear[0].size
    keep = np.zeros_like(idx_clear[0], dtype=bool)

    # Enough clear?
    if nclear < nsgrid_draw:
        return None, None, None

    # Sub-grid me
    sub_size = field_size // sub_grid_step
    rows = np.arange(mask.shape[0]) // sub_size + 1
    sub_nrows = rows[-1]  # The 1 was already added in
    cols = np.arange(mask.shape[1]) // sub_size * rows[-1]
    sub_grid = np.outer(rows, np.ones(mask.shape[1], dtype=int)) + np.outer(
        np.ones(mask.shape[0], dtype=int), cols)

    # Work through each sub_grid
    sub_values = sub_grid[idx_clear]
    uni_sub, counts = np.unique(sub_values, return_counts=True)
    for ss, iuni, icount in zip(np.arange(counts.size), uni_sub, counts):
        mt = np.where(sub_values == iuni)[0]
        if method == 'random':
            r_idx = np.random.choice(icount, size=min(icount, nsgrid_draw))
            keep[mt[r_idx]] = True
        elif method == 'center':
            # Grid lower corners
            sgrid_col = (iuni - 1) // sub_nrows
            sgrid_row = (iuni-1) - sgrid_col*sub_nrows
            # i,j center
            iirow = sgrid_row * sub_size + sub_size // 2
            jjcol = sgrid_col * sub_size + sub_size // 2
            # Distanc3
            dist2 = (idx_clear[0][mt]-iirow)**2 + (idx_clear[1][mt]--jjcol)**2
            # Min and keep
            imin = np.argmin(dist2)
            keep[mt[imin]] = True
            #if (ss % 100) == 0:
            #    print('ss=', ss, counts.size, datetime.datetime.now())
        else:
            raise IOError("Bad method option")
        #if ss > 1000:
        #    break

    # Offset to lower left corner
    picked_row = idx_clear[0][keep] - field_size//2
    picked_col = idx_clear[1][keep] - field_size//2

    # Return
    return picked_row, picked_col, CC_mask[idx_clear][keep]



def extract_file(filename:str, 
                 aios_ds:AIOS_DataSet=None,
                 field_size=(192,192),
                 nadir_offset=0,
                 CC_max=0.05, 
                 qual_thresh=5,
                 temp_bounds = (-3, 34),
                 nrepeat=1,
                 sub_grid_step=2,
                 lower_qual=False,
                 inpaint=True, debug=False):
    """Method to extract a single file.
    Usually used in parallel

    This is very similar to the MODIS routine

    Args:
        filename (str): VIIRS datafile with path
        aiost_ds (AIOS_DataSet): AIOS dataset
            Required!
        field_size (tuple, optional): Size of the cutout side (pixels)
        nadir_offset (int, optional): Maximum offset from nadir for cutout center.
            Zero means any nadir pixel is valid
        CC_max (float, optional): [description]. Defaults to 0.05.
        qual_thresh (int, optional): [description]. Defaults to 2.
        lower_qual (bool, optional): 
            If False, threshold is an upper bound for masking
        temp_bounds (tuple, optional): [description]. Defaults to (-2, 33).
        nrepeat (int, optional): [description]. Defaults to 1.
        sub_grid_step (int, optional):  Sets how finely to sample the image.
            Larger means more finely
        inpaint (bool, optional): [description]. Defaults to True.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        tuple: raw_SST, inpainted_mask, metadata, time
    """

    # Load the image
    #embed(header='51 of extract_file')
    if aios_ds.field == 'SST':
        dfield, qual, latitude, longitude, time = rs_nc_sst.load(filename, verbose=False)
    else:
        raise ValueError("Only SST datasets supported so far")

    if dfield is None:
        return

    # Generate the masks
    masks = rs_nc_utils.build_mask(dfield, qual, 
                                qual_thresh=qual_thresh,
                                temp_bounds=temp_bounds, 
                                lower_qual=lower_qual)

    # Restrict to near nadir
    nadir_pix = dfield.shape[1] // 2
    if nadir_offset > 0:
        lb = nadir_pix - nadir_offset
        ub = nadir_pix + nadir_offset
        dfield = dfield[:, lb:ub]
        masks = masks[:, lb:ub].astype(np.uint8)
    else:
        lb = 0

    # Random clear rows, cols
    rows, cols, clear_fracs = clear_grid(
        masks, field_size[0], 'center', 
        CC_max=CC_max, nsgrid_draw=nrepeat,
        sub_grid_step=sub_grid_step)
    if rows is None:
        print(f"No clear fields for {filename}")
        return None, None, None, None

    # Extract
    fields, inpainted_masks = [], []
    metadata = []
    for r, c, clear_frac in zip(rows, cols, clear_fracs):
        # Inpaint?
        field = dfield[r:r+field_size[0], c:c+field_size[1]]
        mask = masks[r:r+field_size[0], c:c+field_size[1]]
        if inpaint:
            inpainted, _ = pp_field.main(field, mask, only_inpaint=True)
        if inpainted is None:
            continue
        # Null out the non inpainted (to preseve memory when compressed)
        inpainted[~mask] = np.nan
        # Append SST raw + inpainted
        fields.append(field.astype(np.float32))
        inpainted_masks.append(inpainted)
        # meta
        row, col = r, c + lb
        lat = latitude[row + field_size[0] // 2, col + field_size[1] // 2]
        lon = longitude[row + field_size[0] // 2, col + field_size[1] // 2]
        metadata.append([filename, str(row), str(col), str(lat), str(lon), str(clear_frac)])

    del dfield, masks

    if len(fields) == 0:
        print(f"No fields for: {filename}")
        return None, None, None, None
    return np.stack(fields), np.stack(inpainted_masks), np.stack(metadata), time
