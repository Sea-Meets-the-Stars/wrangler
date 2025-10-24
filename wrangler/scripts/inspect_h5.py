""" Script to inspect a H5 file """

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Inspect a HDF5 file')
    parser.add_argument("hdf5_file", type=str, help="File+path to HDF5 file")
    parser.add_argument("--key", type=str, help="Key to view (or a 'shortcut', e.g. sst)")
    # Land?
    #parser.add_argument("--land", action='store_true', help="Overlay land mask")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs



def main(pargs):
    """ Run
    """
    import glob
    import h5py

    # Open me
    h5file = h5py.File(pargs.hdf5_file, 'r')
    # Get the keys
    keys = list(h5file.keys())
    print("-"*80)
    print(f"Keys: {keys}")
    print("-"*80)

    # Key?
    if pargs.key is not None:
        # Shape
        print(f"Key: {pargs.key}")
        print(f"Shape: {h5file[pargs.key].shape}")
    else:
        for key in keys:
            print(f"Key: {key}")
            if hasattr(h5file[key], 'shape'):
                # Shape
                print(f"Shape: {h5file[key].shape}")