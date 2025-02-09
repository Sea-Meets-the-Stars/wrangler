""" Basic I/O methods including s3 calls """

import os
from urllib.parse import urlparse
from io import BytesIO


# s3
import smart_open
import boto3
import functools

endpoint_url = (os.getenv('ENDPOINT_URL') 
                if os.getenv('ENDPOINT_URL') is not None else 
                    'http://rook-ceph-rgw-nautiluss3.rook')

s3 = boto3.resource('s3', endpoint_url=endpoint_url)
client = boto3.client('s3', endpoint_url=endpoint_url)
tparams = {'client': client}
open = functools.partial(smart_open.open, 
                         transport_params=tparams)


def load_to_bytes(s3_uri:str):
    """Load s3 file into memory as a Bytes object

    Args:
        s3_uri (str): Full s3 path

    Returns:
        BytesIO: object in memory
    """
    parsed_s3 = urlparse(s3_uri)
    f = BytesIO()
    s3.meta.client.download_fileobj(parsed_s3.netloc, 
                                    parsed_s3.path[1:], f)
    f.seek(0)
    return f


def download_file_from_s3(local_file:str, s3_uri:str, 
                          clobber_local=True, verbose=True):
    """ Grab an s3 file

    Args:
        local_file (str): Path+filename for new file on local machine
        s3_uri (str): s3 path+filename
        clobber_local (bool, optional): [description]. Defaults to True.
    """
    parsed_s3 = urlparse(s3_uri)
    # Download
    if not os.path.isfile(local_file) or clobber_local:
        if verbose:
            print("Downloading from s3: {}".format(local_file))
        s3.Bucket(parsed_s3.netloc).download_file(
            parsed_s3.path[1:], local_file)
        if verbose:
            print("Done!")
    
def upload_file_to_s3(local_file:str, s3_uri:str):
    """Upload a single file to s3 storage

    Args:
        local_file (str): path to local file
        s3_uri (str): URL for s3 file 
    """
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
    parsed_s3 = urlparse(s3_uri)
    s3.meta.client.upload_file(local_file,
                             parsed_s3.netloc, 
                             parsed_s3.path[1:])
    print("Uploaded {} to {}".format(local_file, s3_uri))
    
def write_bytes_to_local(bytes_:BytesIO, outfile:str):
    """Write a binary object to disk

    Args:
        bytes_ (BytesIO): contains the binary object
        outfile (str): [description]
    """
    bytes_.seek(0)
    with open(outfile, 'wb') as f:
        f.write(bytes_.getvalue())


def write_bytes_to_s3(bytes_:BytesIO, s3_uri:str):
    """Write bytes to s3 

    Args:
        bytes_ (BytesIO): contains the binary object
        s3_uri (str): Path to s3 bucket including filename
    """
    bytes_.seek(0)
    # Do it
    parsed_s3 = urlparse(s3_uri)
    s3.meta.client.upload_fileobj(Fileobj=bytes_, 
                             Bucket=parsed_s3.netloc, 
                             Key=parsed_s3.path[1:])
