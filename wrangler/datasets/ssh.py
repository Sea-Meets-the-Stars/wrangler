""" SSH Datasets """

from .base import AIOS_DataSet


class SSH(AIOS_DataSet):
    field = 'SSH'

    def __init__(self):
        super().__init__()

class LLC4320_SSH(SSH):
    """ LLC 4320 Dataset """
    name = 'LLC4320_SSH'
    source = 'LLC4320'
    variable = 'Theta'
    
    def __init__(self):
        super().__init__()