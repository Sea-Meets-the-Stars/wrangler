""" SSH Datasets """

from .base import AIOS_DataSet


class SSH(AIOS_DataSet):
    field = 'SSH'

    def __init__(self):
        super().__init__()

class LLC_4320(SSH):
    """ LLC 4320 Dataset """
    name = 'LLC4320_SSH'
    source = 'LLC4320'
    variable = 'eta'
    
    def __init__(self):
        super().__init__()