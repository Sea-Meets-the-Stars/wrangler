
import os
from io import BytesIO
import pandas

from wrangler import s3_io as wrangler_io


def load_main_table(tbl_file:str, verbose=True, process_masked:bool=False):
    """Load the table of cutouts 

    Args:
        tbl_file (str): Path to table of cutouts. Local or s3
        verbose (bool, optional): [description]. Defaults to True.
        process_masked (bool, optional): If True, convert masked int columns to pandas nullable Int64 dtype. Defaults to False.

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: table of cutouts
    """
    _, file_extension = os.path.splitext(tbl_file)

    # s3?
    if tbl_file[0:5] == 's3://':
        inp = wrangler_io.load_to_bytes(tbl_file)
    else:
        inp = tbl_file
        
    # Allow for various formats
    if file_extension == '.csv':
        main_table = pandas.read_csv(inp, index_col=0)
        # Set time
        if 'datetime' in main_table.keys():
            main_table.datetime = pandas.to_datetime(main_table.datetime)
    elif file_extension == '.feather':
        # Allow for s3
        main_table = pandas.read_feather(inp)
    elif file_extension == '.parquet':
        # Allow for s3
        main_table = pandas.read_parquet(inp)
    else:
        raise IOError("Bad table extension: ")

    # Deal with masked int columns
    if process_masked:
        for key in ['gradb_Npos', 'FS_Npos', 'UID', 'pp_type']:
            if key in main_table.keys():
                main_table[key] = pandas.array(main_table[key].values, dtype='Int64')

    # Report
    if verbose:
        print("Read main table: {}".format(tbl_file))

    # Decorate
    if 'DT' not in main_table.keys() and 'T90' in main_table.keys():
        main_table['DT'] = main_table.T90 - main_table.T10
        
    return main_table

def write_main_table(main_table:pandas.DataFrame, outfile:str, to_s3=False):
    """Write Main table for ULMO analysis
    Format is determined from the outfile extension.
        Options are ".csv", ".feather", ".parquet"

    Args:
        main_table (pandas.DataFrame): Main table for ULMO analysis
        outfile (str): Output filename.  Its extension sets the format
        to_s3 (bool, optional): If True, write to s3

    Raises:
        IOError: [description]
    """
    _, file_extension = os.path.splitext(outfile)
    if file_extension == '.csv':
        main_table.to_csv(outfile, date_format='%Y-%m-%d %H:%M:%S')
    elif file_extension == '.parquet':
        bytes_ = BytesIO()
        main_table.to_parquet(path=bytes_)
        if to_s3:
            raise NotImplementedError("Not ready for this")
            wrangler_io.write_bytes_to_s3(bytes_, outfile)
        else:
            wrangler_io.write_bytes_to_local(bytes_, outfile)
    else:
        raise IOError("Not ready for this")
    print("Wrote Analysis Table: {}".format(outfile))

