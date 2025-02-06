""" Tests for Datasets """

import pytest

from wrangler.datasets.loader import load_dataset
from wrangler.datasets.sst import VIIRS_NPP


def test_load_dataset():
    """ Test the load_dataset method (and more) """

    # Test loading a dataset by name
    viirs = load_dataset('VIIRS_NPP')
    assert isinstance(viirs, VIIRS_NPP)

    # Check the items
    assert viirs.source == 'PODAAC'  # Tests inheritance
    assert viirs.quality_level == 5  
    assert viirs.podaac_collection == 'VIIRS_NPP-STAR-L2P-v2.80'

    # Test loading a dataset by instance
    viirs2 = load_dataset(viirs)
    assert isinstance(viirs2, VIIRS_NPP)