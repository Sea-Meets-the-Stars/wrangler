
import asyncio
from wrangler.extract import grab_and_go

extract_options = dict(field_size=(192,192))
asyncio.run(grab_and_go.run('VIIRS_NPP', '2024-01-01', '2024-01-02', extract_options,
        'test.h5', verbose=True, debug=False, save_local_files=False))

    