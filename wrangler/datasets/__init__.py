from wrangler.utils import all_subclasses

from wrangler.datasets import base 
from wrangler.datasets import sst
from wrangler.datasets import ssh

def dataset_classes():
    import numpy as np
    # Recursively collect all subclasses
    spec_c = np.array(list(all_subclasses(base.AIOS_DataSet)))
    # Select spectrograph classes with a defined name; spectrographs without a
    # name are either undefined or a base class.
    spec_c = spec_c[[c.name is not None for c in spec_c]]
    # Construct a dictionary with the spectrograph name and class
    srt = np.argsort(np.array([c.name for c in spec_c]))
    return dict([ (c.name,c) for c in spec_c[srt]])

available_datasets = list(dataset_classes().keys())