# encoding: utf-8
try:
    import copy
    import os

    import torch
    import torchvision.datasets as dset
    import numpy as np
    import scipy.io as scio
    import torchvision.transforms as transforms

    from torch.utils.data import Dataset

    from dsvae.config import DATASETS_DIR

except ImportError as e:
    print(e)
    raise ImportError


DATASET_FN_DICT = {
    'mnist': dset.MNIST,
}


dataset_list = DATASET_FN_DICT.keys()


def _get_dataset(dataset_name='mnist'):

    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


# get the loader of all datas
def get_dataloader(dataset_path='../datasets/mnist',
                   dataset_name='mnist', train=True, batch_size=50):
    dataset = _get_dataset(dataset_name)

    transform = [
        transforms.ToTensor(),
    ]
    loader = torch.utils.data.DataLoader(
        dataset(dataset_path, download=True, train=train, transform=transforms.Compose(transform)),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader
