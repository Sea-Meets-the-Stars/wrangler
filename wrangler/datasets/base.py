""" Base Definitions for Datasets """

from abc import ABCMeta

class AIOS_DataSet:
    __metaclass__ = ABCMeta

    # Name of the dataset
    name:str = None

    # Source 
    source:str = None

    """ PODAAC """

    # PODAAC Collection name
    podaac_collection:str = None

    # Quality
    quality_level:int = None

    """ Base class for remote sensing datasets """
    def __init__(self):
        pass

    def __repr__(self):
        return f'<{self.name} Dataset>'

    def __str__(self):
        return f'{self.name} Dataset'