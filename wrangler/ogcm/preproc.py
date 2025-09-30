""" Preprocssing steps related to OGCM """

import numpy as np
from skimage.transform import resize_local_mean


from wrangler.preproc import meta 
from wrangler import utils

try:
    from gsw import density
except ImportError:
    print("gsw not imported;  cannot do density calculations")

def check_items(items):

    # Check for None:
    for item in items:
        if item is None:
            return False

    # If ndarray, check for NaNs
    for item in items:
        if isinstance(item, np.ndarray):
            if np.any(np.isnan(item)):
                return False

    return True


def gradfield2_cutout(item:tuple, resize:bool=False, cutout_size:int=None, 
                dx:float=None, **kwargs):
    """ Generate |grad field|^2 cutout
    
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
            field_cutout, Salt_cutout, idx
        resize (bool, optional): Resize output?. Defaults to False.
        cutout_size (int, optional): cutout size. Defaults to None.
        dx (float, optional): Grid spacing in km
        norm_by_b (bool, optional): Normalize by median buoyancy in the image. Defaults to False.

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray (F_s, gradb)
    """
    # Checks
    if not check_items(item):
        return None, item[-1], None

    # Unpack
    field_cutout, idx = item

    # Calculate
    grad = utils.calc_grad2(field_cutout, dx)

    # Resize
    if resize:
        grad = resize_local_mean(grad, (cutout_size, cutout_size))

    # Meta
    meta_dict = meta.stats(grad)

    # Return
    return grad, idx, meta_dict

def gradb2_cutout(item:tuple, resize:bool=False, cutout_size:int=None, 
                dx:float=None, norm_by_b:bool=False, **kwargs):
    """ Generate |grad b|^2 cutout
    
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
            Theta_cutout, Salt_cutout, idx
        resize (bool, optional): Resize output?. Defaults to False.
        cutout_size (int, optional): cutout size. Defaults to None.
        dx (float, optional): Grid spacing in km
        norm_by_b (bool, optional): Normalize by median buoyancy in the image. Defaults to False.

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray (F_s, gradb)
    """
    # Checks
    if not check_items(item):
        return None, item[-1], None

    # Unpack
    Theta_cutout, Salt_cutout, idx = item

    # Calculate
    gradb = calc_gradb2(Theta_cutout, Salt_cutout, dx=dx,
                       norm_by_b=norm_by_b)

    # Resize
    if resize:
        gradb = resize_local_mean(gradb, (cutout_size, cutout_size))

    # Meta
    meta_dict = meta.stats(gradb)

    # Return
    return gradb, idx, meta_dict


def Fs_cutout(item:tuple, resize:bool=False, cutout_size:int=None, 
                dx:float=None, norm_by_b:bool=False, **kwargs):
    """Simple function to measure front related stats
    for a cutout
    
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
        cutout_size (int, optional): cutout size. Defaults to None.
        dx (float, optional): Grid spacing in km
        norm_by_b (bool, optional): Normalize by median buoyancy in the image. Defaults to False.

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray (F_s, gradb)
    """
    # Checks
    if not check_items(item):
        return None, item[-1], None

    # Unpack
    U_cutout, V_cutout, Theta_cutout, Salt_cutout, idx = item

    # Calculate
    Fs = calc_F_s(U_cutout, V_cutout, Theta_cutout, Salt_cutout, dx=dx)

    # Resize
    if resize:
        Fs = resize_local_mean(Fs, (cutout_size, cutout_size))

    # Meta
    meta_dict = meta.stats(Fs)

    # Return
    return Fs, idx, meta_dict

def b_cutout(item:tuple, resize:bool=False, cutout_size:int=None, 
             ref_rho:float=1025., g=0.0098, **kwargs):
    """Simple function to grab a density cutout
    
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
        cutout_size (int, optional): cutout size. Defaults to None.

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray (F_s, gradb)
    """
    # Checks
    if not check_items(item):
        return None, item[-1], None

    # Unpack
    Theta_cutout, Salt_cutout, idx = item

    # Calculate
    rho = density.rho(Salt_cutout, Theta_cutout, np.zeros_like(Salt_cutout))
    b = g*rho/ref_rho

    # Resize
    if resize:
        b = resize_local_mean(b, (cutout_size, cutout_size))

    # Meta
    meta_dict = meta.stats(b)

    # Return
    return b, idx, meta_dict

def okuboweiss_cutout(item:tuple, resize:bool=False, cutout_size:int=None, 
                dx:float=None, **kwargs):
    """ Generate Okubo-Weiss
    
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
            U_cutout, V_cutout, idx
            U, V assumed to be in m/s
        resize (bool, optional): Resize output?. Defaults to False.
        cutout_size (int, optional): cutout size. Defaults to None.
        dx (float, optional): Grid spacing in km
        norm_by_b (bool, optional): Normalize by median buoyancy in the image. Defaults to False.

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray (F_s, gradb)
    """
    # Checks
    if not check_items(item):
        return None, item[-1], None

    # Unpack
    U_cutout, V_cutout, idx = item

    # Calculate
    gradb = calc_okubo_weiss(U_cutout, V_cutout, dx=dx)

    # Resize
    if resize:
        gradb = resize_local_mean(gradb, (cutout_size, cutout_size))

    # Meta
    meta_dict = meta.stats(gradb)

    # Return
    return gradb, idx, meta_dict



def calc_gradb2(Theta:np.ndarray, Salt:np.ndarray,
             ref_rho:float=1025., g=0.0098, dx=2.,
             norm_by_b:bool=False):
    """Calculate |grad b|^2

    Args:
        Theta (np.ndarray): SST field
        Salt (np.ndarray): Salt field
        ref_rho (float, optional): Reference density
        g (float, optional): Acceleration due to gravity
            in km/s^2
        dx (float, optional): Grid spacing in km

    Returns:
        np.ndarray: |grad b|^2 field
    """
    # Buoyancy
    rho = density.rho(Salt, Theta, np.zeros_like(Salt))
    b = g*rho/ref_rho

    # Normalize by b?
    if norm_by_b:
        b /= np.median(b)

    return utils.calc_grad2(b, dx)


def calc_F_s(U:np.ndarray, V:np.ndarray,
             Theta:np.ndarray, Salt:np.ndarray,
             add_gradb=False,
             ref_rho=1025., g=0.0098, dx=2.,
             calc_T_SST:bool=False):
    """Calculate the Frontogenesis tendency 

    Default is the standard (density) approach
    but one can optinally use SST

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field
        Theta (np.ndarray): SST field
        Salt (np.ndarray): Salt field
        ref_rho (float, optional): Reference density
        add_gradb (bool, optional): Calculate+return gradb 
        g (float, optional): Acceleration due to gravity
            in km/s^2
        dx (float, optional): Grid spacing in km
        calc_T_SST (bool, optional): Calculate the SST tendency?

    Returns:
        np.ndarray or tuple: F_s field (, gradb2)
    """
    dUdx = np.gradient(U, axis=1)
    dVdx = np.gradient(V, axis=1)
    #
    dUdy = np.gradient(U, axis=0)
    dVdy = np.gradient(V, axis=0)

    # Buoyancy
    if calc_T_SST:
        dbdx = -1*np.gradient(Theta, axis=1) / dx
        dbdy = -1*np.gradient(Theta, axis=0) / dx
    else:
        rho = density.rho(Salt, Theta, np.zeros_like(Salt))
        dbdx = -1*np.gradient(g*rho/ref_rho, axis=1) / dx
        dbdy = -1*np.gradient(g*rho/ref_rho, axis=0) / dx

    # Terms
    F_s_x = -1 * (dUdx*dbdx + dVdx*dbdy) * dbdx 
    F_s_y = -1 * (dUdy*dbdx + dVdy*dbdy) * dbdy 

    # Finish
    F_s = F_s_x + F_s_y

    # div b too?
    if add_gradb:
        grad_b2 = dbdx**2 + dbdy**2
        return F_s, grad_b2
    else:
        return F_s

def calc_div(U:np.ndarray, V:np.ndarray):
    """Calculate the divergence

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: Divergence array
    """
    dUdx = np.gradient(U, axis=1)
    dVdy = np.gradient(V, axis=0)
    div = dUdx + dVdy
    #
    return div

def calc_curl(U:np.ndarray, V:np.ndarray):  # Also the relative or vertical vorticity?!
    """Calculate the curl (aka relative vorticity)

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: Curl
    """
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    curl = dVdx - dUdy
    # Return
    return curl


def calc_normal_strain(U:np.ndarray, V:np.ndarray):
    """Calculate the normal strain

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: normal strain
    """
    dUdx = np.gradient(U, axis=1)
    dVdy = np.gradient(V, axis=0)
    norm_strain = dUdx - dVdy
    # Return
    return norm_strain

def calc_shear_strain(U:np.ndarray, V:np.ndarray):
    """Calculate the shear strain

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: shear strain
    """
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    shear_strain = dUdy + dVdx
    # Return
    return shear_strain

def calc_lateral_strain_rate(U:np.ndarray, V:np.ndarray):
    """Calculate the lateral strain rate

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: alpha
    """
    dUdx = np.gradient(U, axis=1)
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    dVdy = np.gradient(V, axis=0)
    #
    alpha = np.sqrt((dUdx-dVdy)**2 + (dVdx+dUdy)**2)
    return alpha

def calc_okubo_weiss(U:np.ndarray, V:np.ndarray, dx:float=2.):
    """Calculate Okubo-Weiss

    Args:
        U (np.ndarray): U velocity field
            Assumed m/s
        V (np.ndarray): V velocity field
            Assumed m/s
        dx (float, optional): Grid spacing in km

    Returns:
        np.ndarray: okubo-weiss
    """
    s_n = calc_normal_strain(U, V)
    s_s = calc_shear_strain(U, V)
    w = calc_curl(U, V)  # aka relative vorticity
    #
    W = s_n**2 + s_s**2 - w**2

    # Return
    return W / (dx*1e3)**2