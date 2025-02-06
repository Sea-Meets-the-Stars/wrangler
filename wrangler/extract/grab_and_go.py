""" Routines to grab and extract dataasets in one go """

from remote_sensing.download import podaac


def run(tstart:str, tend:str, dataset:str, outfile:str,
        tdelta:dict={'days':1}, verbose:bool=True,
        debug:bool=False):

    # Instantiate the AIOS_DataSet