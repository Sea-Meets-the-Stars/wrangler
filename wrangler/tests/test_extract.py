
import asyncio

from wrangler.extract import grab_and_go
from wrangler.datasets.loader import load_dataset

# Grab

def test_grab_filelist():
    aios_ds = load_dataset('VIIRS_NPP')
    asyncio.run(grab_and_go.grab(aios_ds, '2024-01-01', '2024-01-02', 
                                 verbose=True, skip_download=True))

# Extract
#def test_extract():
extract_options = dict(field_size=(192,192))
asyncio.run(grab_and_go.run('VIIRS_NPP', '2024-01-01', 
                            '2024-01-02', extract_options, 
                            'test.h5', 
                            verbose=True, debug=False, 
                            save_local_files=False))

    