""" Scripts to Download the assess H09 data from 2024 """

from wrangler.extract import grab_and_go

def download_h09():
    """ Download the H09 data from 2024 """

    dataset = '`H09-AHI-L3C-ACSPO-v2.90'
    t0 = '2024-01-01:T00:00:00'
    t1 = '2024-12-31:T23:59:59'

    # Download the data
    local_files = grab_and_go(dataset=dataset,
        t0=t0,
        t1=t1, skip_download=True)

