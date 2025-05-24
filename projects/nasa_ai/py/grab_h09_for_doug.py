""" Grab Himawari 09 data for Doug """
import os

from wrangler.datasets import loader 
from wrangler.extract import grab_and_go 


def main():
    aios_ds = loader.load_dataset('H09_L3C')
    t0 = '2023-04-01T00:00:00'
    t1 = '2023-08-01T01:00:00'
    download_dir = os.path.join(os.getenv('OS_SST'), 'H09', 'Doug')
    #
    local_files = grab_and_go.grab(aios_ds, t0, t1, download_dir=download_dir)


# Command line
if __name__ == "__main__":
    main()