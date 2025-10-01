
import numpy as np

from IPython import embed

# Prepping
R_earth = 6371. # km
circum = 2 * np.pi* R_earth
km_deg = circum / 360.

def latlon_for_cutout(latlon:tuple, cutout_size:int, dx:float):

    # Lats
    lats = latlon[0] + np.arange(cutout_size)*dx / km_deg
    lat_img = np.outer(lats, np.ones(cutout_size))

    # lons (approximate)
    lons = latlon[1] + np.arange(cutout_size)*dx / (km_deg * np.cos(latlon[0]*np.pi/180.))
    lon_img = np.outer(np.ones(cutout_size), lons)

    # Return
    return lat_img, lon_img

def latlons_for_cutouts(latlons:np.ndarray, cutout_size:int, dx:float):

    # Unpack
    lats, lons = latlons
    ncutouts = lats.size

    # Resize for array math
    lats.resize(ncutouts,1,1)
    lons.resize(ncutouts,1,1)
    
    # Latitudes
    dlats = (np.arange(cutout_size) - cutout_size//2)*dx / km_deg
    dlat_img = np.outer(dlats, np.ones(cutout_size))

    lat_imgs = np.ones((ncutouts, cutout_size, cutout_size)) * lats
    lat_imgs += dlat_img

    # Longitudes
    lon_imgs = np.ones((ncutouts, cutout_size, cutout_size)) * lons

    dlon_img = dlat_img.T
    dlons = np.ones((ncutouts, cutout_size, cutout_size)) * dlon_img / np.cos(lons*np.pi/180.)

    lon_imgs += dlons

    # Return
    return lat_imgs, lon_imgs