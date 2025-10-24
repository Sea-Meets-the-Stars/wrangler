""" SST Datasets """

from .base import AIOS_DataSet

class SST(AIOS_DataSet):
    field = 'SST'

    def __init__(self):
        super().__init__()

# VIIRS Datasets

class VIIRS(SST):
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

class VIIRS_N20(VIIRS):
    """ VIIRS NPP Dataset """
    name = 'VIIRS_N20'
    podaac_collection = 'VIIRS_N20-STAR-L2P-v2.80'
    
    def __init__(self):
        super().__init__()

class VIIRS_N21(VIIRS):
    """ VIIRS NPP Dataset """
    name = 'VIIRS_N21'
    podaac_collection = 'N21-VIIRS-L2P-ACSPO-v2.80'
    
    def __init__(self):
        super().__init__()

# Himawari Datasets

class H09(SST):
    """ Parent for Himawari Datasets """
    source = 'PODAAC'
    quality_level = 5

    def __init__(self):
        super().__init__()

class H09_L3C(H09):
    """ Himawari Level 3 Dataset """
    name = 'H09_L3C'
    podaac_collection = 'H09-AHI-L3C-ACSPO-v2.90'
    
    def __init__(self):
        super().__init__()

class LLC_4320(SST):
    """ LLC 4320 Dataset """
    name = 'LLC4320_SST'
    source = 'LLC4320'
    
    def __init__(self):
        super().__init__()