""" Scripts to wrangle the PNGs into a single h5 file """

import os
import glob

import numpy as np
import h5py

import pandas

import matplotlib.image as mpimg

from IPython import embed


def wrangle_one_pass(ipass:int, path:str=None,
                     npix:int=55, debug:bool=False):
    if path is None:
        path = os.getenv('SWOT_PNGs')

    files = glob.glob(os.path.join(
        path, f'Pass_{ipass:03d}', 'ssr_*.png'))

    # Sort the files
    files = sorted(files)

    # Loop on the files
    all_imgs = []
    idx0, idx1, idx2 = [], [], []
    row, col = [], []
    sv_files = []
    for kk, ifile in enumerate(files):
        if debug and kk > 2:
            break
        print(f'Processing {kk+1}/{len(files)}')
        # Read the image
        img = mpimg.imread(ifile)
        # Indices
        basef = os.path.basename(ifile)
        # 
        sub_imgs = []
        for irow in range(img.shape[0]//npix):
            for jcol in range(img.shape[1]//npix):
                sub_imgs.append(img[irow*npix:(irow+1)*npix, jcol*npix:(jcol+1)*npix, 0])
                #
                idx0.append(int(basef.split('_')[1]))
                idx1.append(int(basef.split('_')[2]))
                idx2.append(int(basef.split('_')[3]))
                # Row, col
                row.append(irow)
                col.append(jcol)
                # File
                sv_files.append(ifile)
        all_imgs.append(np.array(sub_imgs))

    # Stack the images
    all_imgs = np.concatenate(all_imgs)#, axis=0)
    # Recast
    idx0 = np.array(idx0, dtype=np.int32)
    idx1 = np.array(idx1, dtype=np.int32)
    idx2 = np.array(idx2, dtype=np.int32)
    row = np.array(row, dtype=np.int32)
    col = np.array(col, dtype=np.int32)
    sv_files = np.array(sv_files, dtype='S100')

    # Create a simple table of metadata
    df = pandas.DataFrame()
    df['filename'] = sv_files
    df['idx0'] = idx0
    df['idx1'] = idx1
    df['idx2'] = idx2
    df['row'] = row
    df['col'] = col

    # Write to disk as h5py
    outf = os.path.join(path, f'Pass_{ipass:03d}.h5')
    tbl_outf = os.path.join(path, f'Pass_{ipass:03d}.parquet')

    #embed(header='Check 51')

    with h5py.File(outf, 'w') as f:
        f.create_dataset('imgs', data=all_imgs)
        # Add the metadata
        #dset = f.create_dataset('metadata', 
        #                        data=df.to_numpy(dtype=str).astype('S'))
        #dset.attrs['columns'] = clms
        #f.create_dataset('metadata', data=df)
    print(f'Wrote {outf} with {all_imgs.shape[0]} images of size {all_imgs.shape[1:]}')

    # Write table
    df.to_parquet(tbl_outf, index=False)
    print(f'Wrote {tbl_outf} with {df.shape[0]} rows')



def main(flg):
    flg= int(flg)

    # Pass 003 only
    if flg == 3:
        wrangle_one_pass(3)#, debug=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)