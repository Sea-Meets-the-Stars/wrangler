""" SST Datasets """

from .base import AIOS_DataSet


class VIIRS(AIOS_DataSet):
    """ Parent for VIIRS Datasets """
    source = 'PODAAC'
    quality_level = 5

    def __init__(self):
        super().__init__()

class VIIRS_NPP(VIIRS):
    """ VIIRS NPP Dataset """
    name = 'VIIRS_NPP'
    podaac_collection = 'VIIRS_NPP-STAR-L2P-v2.80'
    
    def __init__(self):
        super().__init__()