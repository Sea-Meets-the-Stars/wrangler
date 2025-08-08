""" Script to inspect a H5 file """

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Show a cutout from an HDF5 file')
    parser.add_argument("hdf5_file", type=str, help="File+path to HDF5 file")
    parser.add_argument("partition", type=str, help="Partition of the HDF5 file")
    parser.add_argument("idx", type=int, help="Index of the cutout to view")
    parser.add_argument("--cm", type=str, default='jet', help="Colormap to use for the image (default: 'jet')")
    parser.add_argument("--clbl", type=str, help="Colormap label")
    #parser.add_argument("--land", action='store_true', help="Overlay land mask")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs



def main(pargs):
    """ Run
    """
    import h5py

    from wrangler.plotting import cutout as plot_cutout

    # Grab the image
    with h5py.File(pargs.hdf5_file, 'r') as h5file:
        # Check the partition
        if pargs.partition not in h5file.keys():
            raise ValueError(f"Partition '{pargs.partition}' not found in {pargs.hdf5_file}")
        
        # Get the cutout
        cutout = h5file[pargs.partition][pargs.idx,...]
        print(f"Cutout shape: {cutout.shape}")

    # Remove the channels
    if cutout.ndim == 3:
        cutout = cutout[0,...]
    elif cutout.ndim == 4:
        cutout = cutout[0,0,...]

    # Max, min
    print(f"Cutout min: {cutout.min()}, max: {cutout.max()}")

    # Plot
    plot_cutout.show_image(cutout, show=True, cm=pargs.cm, clbl=pargs.clbl)
    