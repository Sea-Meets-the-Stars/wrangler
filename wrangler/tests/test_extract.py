
import asyncio

from wrangler.extract import grab_and_go
from wrangler.datasets.loader import load_dataset

# Grab

def test_grab_filelist():
    aios_ds = load_dataset('VIIRS_NPP')
    asyncio.run(grab_and_go.grab(aios_ds, '2024-01-01', '2024-01-02', 
                                 verbose=True, skip_download=True))

# Extract
def test_extract():
    extract_file = 'files/extract_viirs_std.json'
    asyncio.run(grab_and_go.run('VIIRS_NPP', '2024-01-01', 
                            '2024-01-02', extract_file,
                            'test.h5', 'test.parquet', 
                            4, # n_cores
                            verbose=True, debug=False, 
                            save_local_files=True))

    