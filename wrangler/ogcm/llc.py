""" Methods related to ECCO LLC datasets """
import os
import numpy as np

import xarray

import pandas

# Astronomy tools
import astropy_healpix

from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from wrangler.tables import io as tbl_io

def add_days(llc_table:pandas.DataFrame, dti:pandas.DatetimeIndex, outfile=None):
    """Add dates to an LLC table

    Args:
        llc_table (pandas.DataFrame): [description]
        dti (pandas.DatetimeIndex): [description]
        outfile ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    # Check
    if 'datetime' in llc_table.keys():
        print("Dates already specified.  Not modifying")
        return llc_table

    # Do it
    llc_table['datetime'] = dti[0]
    for date in dti[1:]:
        new_tbl = llc_table[llc_table['datetime'] == dti[0]].copy()
        new_tbl['datetime'] = date
        #llc_table = llc_table.append(new_tbl, ignore_index=True)
        llc_table = pandas.concat([llc_table, new_tbl], ignore_index=True)

    # Drop index
    llc_table.drop(columns=['index'], inplace=True)

    # Write
    if outfile is not None:
        #assert catalog.vet_main_table(llc_table)
        tbl_io.write_main_table(llc_table, outfile)

    # Return
    return llc_table


def build_table(debug=False, resol=0.5, minmax_lat=None):
    """ Get the show started by sampling uniformly
    in space and and time

    Args:
        debug (bool, optional): _description_. Defaults to True.
        resol (float, optional): 
            Typical separation of images in deg
        minmax_lat (tuple, optional): Restrict to latitudes given by this range

    """
    # Begin 
    llc_table = coords(resol=resol, minmax_lat=minmax_lat, field_size=(64,64))

    # Temporal sampling
    if debug:
        # Extract 6 days across the full range;  ends of months
        dti = pandas.date_range('2011-09-13', periods=6, freq='2M')
    else:
        # Extract 24 days across the full range;  ends of months; every 2 weeks
        dti = pandas.date_range('2011-09-13', periods=24, freq='2W')
    llc_table = add_days(llc_table, dti)

    # Return
    return llc_table

def coords(resol, field_size, CC_max=1e-4, outfile=None, 
           minmax_lat=None, localCC=True,
           rotate:float=None):
    """
    Use healpix to setup a uniform extraction grid

    Args:
        resol (float): Typical separation on the healpix grid
        minmax_lat (tuple): Restrict to latitudes given by this range
        field_size (tuple): Cutout size in pixels
        outfile (str, optional): If provided, write the table to this outfile.
            Defaults to None.
        localCC (bool, optional):  If True, load the CC_mask locally.
        rotate (float, optional): Rotate the grid by this angle (deg)

    Returns:
        pandas.DataFrame: Table containing the coords
    """
    # Load up CC_mask
    CC_mask = load_CC_mask(field_size=field_size, local=localCC)

    # Cut
    good_CC = CC_mask.CC_mask.values < CC_max
    good_CC_idx = np.where(good_CC)

    # Build coords
    llc_lon = CC_mask.lon.values[good_CC].flatten()
    llc_lat = CC_mask.lat.values[good_CC].flatten()
    print("Building LLC SkyCoord")
    llc_coord = SkyCoord(llc_lon*units.deg + 180.*units.deg, 
                         llc_lat*units.deg, 
                         frame='galactic')

    # Healpix time
    nside = astropy_healpix.pixel_resolution_to_nside(resol*units.deg)
    hp = astropy_healpix.HEALPix(nside=nside)
    hp_lon, hp_lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    if rotate is not None:
        hp_lon = hp_lon + rotate*np.pi/180. * units.rad

    # Coords
    hp_coord = SkyCoord(hp_lon, hp_lat, frame='galactic')
                        
    # Cross-match
    print("Cross-match")
    idx, sep2d, _ = match_coordinates_sky(hp_coord, llc_coord, nthneighbor=1)
    good_sep = sep2d < hp.pixel_resolution

    # Build the table
    llc_table = pandas.DataFrame()
    llc_table['lat'] = llc_lat[idx[good_sep]]  # Center of cutout
    llc_table['lon'] = llc_lon[idx[good_sep]]  # Center of cutout

    llc_table['row'] = good_CC_idx[0][idx[good_sep]] - field_size[0]//2 # Lower left corner
    llc_table['col'] = good_CC_idx[1][idx[good_sep]] - field_size[0]//2 # Lower left corner

    # Cut on latitutde?
    if minmax_lat is not None:
        print(f"Restricting to latitudes = {minmax_lat}")
        #gd_lat = np.abs(llc_table.lat) < max_lat
        gd_lat = (llc_table.lat > minmax_lat[0]) & (llc_table.lat < minmax_lat[1])
        llc_table = llc_table[gd_lat].copy()

    llc_table.reset_index(inplace=True)
    
    # Write
    if outfile is not None:
        tbl_io.write_main_table(llc_table, outfile)

    # Return
    return llc_table

def grab_llc_datafile(datetime=None, root='LLC4320_', chk=True, data_path:str=None):
    """Generate the LLC datafile name from the inputs

    Args:
        datetime (pandas.TimeStamp, optional): Date. Defaults to None.
        root (str, optional): [description]. Defaults to 'LLC4320_'.
        chk (bool, optional): [description]. Defaults to True.
        data_path (str, optional): Path to the LLC data. 
            If None, will use $OS_OGCM/LLC/data/ThetaUVSalt

    Returns:
        str: LLC datafile name
    """
    if data_path is None:
        data_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'data',
                                      'ThetaUVSalt')
        
    if datetime is not None:
        sdate = str(datetime).replace(':','_')[:19]
        # Add T?
        if sdate[10] == ' ':
            sdate = sdate.replace(' ', 'T')
        # Finish
        datafile = os.path.join(data_path, root+sdate+'.nc')
    if chk: 
        assert os.path.isfile(datafile)
    # Return
    return datafile
                    
def load_coords(verbose=True):
    """Load LLC coordinates

    Args:
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        xarray.DataSet: contains the LLC coordinates
    """
    coord_file = os.path.join(os.getenv('OS_OGCM'), 
                              'LLC', 'data', 'CC', 'LLC_coords.nc')
    if verbose:
        print("Loading LLC coords from {}".format(coord_file))
    coord_ds = xarray.load_dataset(coord_file)
    return coord_ds


def load_CC_mask(field_size=(64,64), verbose=True, local=True):
    """Load up a CC mask.  Typically used for setting coordinates

    Args:
        field_size (tuple, optional): Field size of the cutouts. Defaults to (64,64).
        verbose (bool, optional): Defaults to True.
        local (bool, optional): Load from local hard-drive. 
            Requires LLC_DATA env variable.  Defaults to True (these are 3Gb files)

    Returns:
        xr.DataSet: CC_mask
    """
    CC_mask_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'data', 'CC',
                                   'LLC_CC_mask_{}.nc'.format(field_size[0]))
    CC_mask = xarray.open_dataset(CC_mask_file)
    if verbose:
        print("Loaded LLC CC mask from {}".format(CC_mask_file))
    # Return
    return CC_mask
