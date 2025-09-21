""" Preprocessing of one field """

import numpy as np

from skimage.restoration import inpaint as sk_inpaint
from scipy.ndimage import median_filter
from scipy import special
from skimage.transform import downscale_local_mean, resize_local_mean
from skimage import filters

from wrangler.preproc import meta

from IPython import embed

def multi_process(item:tuple, pdict:dict, use_mask=False,
                  inpainted_mask=False):
    """
    Simple wrapper for main()
    Mainly for multi-processing

    Parameters
    ----------
    item : tuple
        field, idx or field,mask,idx (use_mask=True)
    pdict : dict
        Preprocessing dict
    use_mask : bool, optional
        If True, allow for an input mask
    inpainted_mask : bool, optional
        If True, the tuple includes an inpainted_mask
        instead of a simple mask.

    Returns
    -------
    pp_field, idx, meta : np.ndarray, int, dict

    """
    # Unpack
    mask = None
    smooth_pix = None
    if use_mask:
        field, mask, idx = item
        if inpainted_mask:
            true_mask = np.isfinite(mask)
            # Fill-in inpainted values
            field[true_mask] = mask[true_mask]
            # Overwrite
            mask = true_mask
    elif 'smooth_km' in pdict.keys():
        field, idx, smooth_pix = item
    else:
        field, idx = item

    # Junk field?  (e.g. LLC)
    if field is None:
        return None, idx, None

    # Run
    pp_field, imeta = main(field, mask, smooth_pix=smooth_pix, **pdict)

    # Failed?
    if pp_field is None:
        return None, idx, None

    # Return
    return pp_field.astype(np.float32), idx, imeta


def main(cutout, mask, inpaint=False, 
         median=False, med_size=(3,1),
         downscale=False, dscale_size=(2,2), 
         sigmoid=False, scale=None,
         expon=None, only_inpaint=False, gradient=False,
         min_mean=None, de_mean=False,
         div2=None,
         cutout_size=None, resize:bool=False,
         smooth_pix:int=None,
         noise=None,
         log_scale=False, **kwargs):
    """
    For multi-processing, it is best to wrap this in a simple function
    that takes a single cutout and returns the pre-processed cutout.
    
    Preprocess an input cutout image with a series of steps:
        1. Inpainting
        2. Resize based on fixed_km (LLC)
        3. Add noise
        4. Median
        5. Downscale
        6. Sigmoid
        7. Scale
        8. Remove mean
        9. Sobel
        10. Log

    Parameters
    ----------
    cutout : np.ndarray
    mask : np.ndarray or None
        Data mask.  True = masked
        Required for inpainting but otherwise ignored
    inpaint : bool, optional
        if True, inpaint masked values
    smoooth_pix : int, optional
        Smooth the cutout with a Gaussian filter of this size
    median : bool, optional
        If True, apply a median filter
    med_size : tuple
        Median window to apply
    downscale : bool, optional
        If True downscale the image
    dscale_size : tuple, optional
        Size to rescale by
    noise : float, optional
        If provided, add white noise with this value
    scale : float, optional
        Scale the SSTa values by this multiplicative factor
    expon : float
        Exponate the SSTa values by this exponent
    gradient : bool, optional
        If True, apply a Sobel gradient enhancing filter
    de_mean : bool, optional
        If True, subtract the mean
    min_mean : float, optional
        If provided, require the image has a mean exceeding this value
    resize : bool, optional
        If provided, resize the input imzge to cutout_size  x cutout_size  
    **kwargs : catches extraction keywords

    Returns
    -------
    pp_cutout, meta_dict : np.ndarray, dict
        Pre-processed cutout, mean temperature

    """

    # Inpaint?
    if inpaint:
        if mask.dtype.name != 'uint8':
            mask = np.uint8(mask)
        cutout = sk_inpaint.inpaint_biharmonic(cutout, mask, channel_axis=None)

    if only_inpaint:
        if np.any(np.isnan(cutout)):
            return None, None
        else:
            return cutout, None

    # Smooth?
    if smooth_pix is not None:
        cutout = filters.gaussian(cutout, smooth_pix)
        # Crop
        cutout = cutout[2*smooth_pix:-2*smooth_pix, 2*smooth_pix:-2*smooth_pix]

    # Resize?
    if resize is not None:
        cutout = resize_local_mean(cutout, (cutout_size, cutout_size))

    # Capture metadata
    meta_dict = meta.stats(cutout)

    # Add noise?
    if noise is not None:
        cutout += np.random.normal(loc=0., 
                                  scale=noise, 
                                  size=cutout.shape)
    # Median
    if median:
        cutout = median_filter(cutout, size=med_size)

    # Reduce to 64x64
    if downscale:
        cutout = downscale_local_mean(cutout, dscale_size)

    # Check for junk
    if np.any(np.isnan(cutout)):
        return None, None

    # Check mean
    if min_mean is not None and meta_dict['mu'] < min_mean:
        return None, None

    # De-mean the field
    if de_mean:
        pp_cutout = cutout - cutout.mean()
    else:
        pp_cutout = cutout

    # Sigmoid?
    if sigmoid:
        pp_cutout = special.erf(pp_cutout)

    # Scale?
    if scale is not None:
        pp_cutout *= scale

    # Exponate?
    if expon is not None:
        neg = pp_cutout < 0.
        pos = np.logical_not(neg)
        pp_cutout[pos] = pp_cutout[pos]**expon
        pp_cutout[neg] = -1 * (-1*pp_cutout[neg])**expon

    # Sobel Gradient?
    if gradient:
        pp_cutout = filters.sobel(pp_cutout)
        # Meta
        srt = np.argsort(pp_cutout.flatten())
        i10 = int(0.1*pp_cutout.size)
        i90 = int(0.9*pp_cutout.size)
        meta_dict['G10'] = pp_cutout.flatten()[srt[i10]]
        meta_dict['G90'] = pp_cutout.flatten()[srt[i90]]
        meta_dict['Gmax'] = pp_cutout.flatten()[srt[-1]]

    # |grad f|^2?
    if div2 is not None:
        pp_cutout = po_utils.calc_grad2(pp_cutout, div2)

    # Log?
    if log_scale:
        if not gradient:
            raise IOError("Only implemented with gradient=True so far")
        # Set 0 values to the lowest non-zero value
        zero = pp_cutout == 0.
        if np.any(zero):
            min_nonz = np.min(pp_cutout[np.logical_not(zero)])
            pp_cutout[zero] = min_nonz
        # Take log
        pp_cutout = np.log(pp_cutout)


    # Return
    return pp_cutout, meta_dict