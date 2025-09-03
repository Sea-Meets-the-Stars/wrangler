""" Method to load a named AIOS_DataSet """

from wrangler.datasets import base
from wrangler import datasets

def load_dataset(dataset):
    """
    Instantiate a dataset from the available subclasses of
    :class:`~wrangler.datasets.base.AIOS_DataSet`.

    Args:
        dataset (:obj:`str`, :class:`~wrangler.datasets.base.AIOS_DataSet`):
            The spectrograph to instantiate. If the input object is ``None``
            or has :class:`~wrangler.datasets.base.AIOS_DataSet`
            as a base class, the instance is simply returned. If it is a
            string, the string is used to instantiate the relevant
            spectrograph instance.

    Raises:
        IOError: If the dataset is not a supported dataset.

    Returns:
        :class:`~wrangler.datasets.base.AIOS_DataSet`: 
        The dataset instance

    """
    if dataset is None or isinstance(dataset, base.AIOS_DataSet):
        return dataset

    classes = datasets.dataset_classes()
    if dataset in classes.keys():
        return classes[dataset]()


    raise IOError(f'{dataset} is not a supported dataset.')