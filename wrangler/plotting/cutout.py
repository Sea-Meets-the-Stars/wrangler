""" Methods for plotting cutout images"""

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def show_image(img:np.ndarray, cm=None, cbar:bool=True, 
               vmnx=(None,None), show=False, set_aspect=None, clbl=None,
               ax=None):
    """Dispay the cutout image
    Args:
        img (np.ndarray): cutout image
        cm ([type], optional): Color map to use. Defaults to None.
            If None, load the heatmap above
        cbar (bool, optional): If True, show a color bar. Defaults to True.
        clbl ([type], optional): Color bar label. Defaults to None.
        vmnx (tuple, optional): Set vmin, vmax. Defaults to None
        set_aspect (str, optional):
            Passed to ax.set_aspect() if provided
        ax (matplotlib.Axis, optional): axis to use for the plot
    Returns:
        matplotlib.Axis: axis containing the plot
    """
    if cm is None:
        cm = 'jet'
    #
    ax = sns.heatmap(np.flipud(img), xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1], ax=ax,
                     yticklabels=[], cmap=cm, cbar=cbar, 
                     cbar_kws={'label': clbl})# 'fontsize': 20})
    #plt.savefig('image', dpi=600)
    
    if show:
        plt.show()
    if set_aspect is not None:
        ax.set_aspect(set_aspect)
    #
    return ax
