import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset

from dataset.norb import smallNORBViewPoint, smallNORB
from dataset.slice import Slice
from dataset.oneperson_3d import onePerson_3d


def get_train_test_loader( dataset,
                           batch_size,
                           data_dir = ' ',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == 'mnist':
        trans = [transforms.Pad(2),
                 transforms.RandomCrop(28),
                 transforms.ToTensor(),
                ]
        train_set = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose(trans))

        trans = [transforms.ToTensor()]
        test_set = datasets.MNIST('../data', train=False,
                                  transform=transforms.Compose(trans))
        data_size = {'h':28, 'w':28, 'c':1, 'dataset':dataset}
        n_classes = 10

    elif dataset == "cifar10":
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        train_set = datasets.CIFAR10('../data', train=True, download=True,
                                     transform=transforms.Compose(trans))

        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        test_set = datasets.CIFAR10('../data', train=False,
                                  transform=transforms.Compose(trans))
        data_size = {'h':32, 'w':32, 'c':3, 'dataset':dataset}
        n_classes = 10

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='train', download=True,
                transform=transforms.Compose(trans))

    elif dataset == "slice":
        train_set = Slice('../../prostate_pirads/phase4/npy/slice_org', '../../prostate_pirads/phase4/npy/slice_org_train.txt')
        test_set = Slice('../../prostate_pirads/phase4/npy/slice_org', '../../prostate_pirads/phase4/npy/slice_org_test.txt')
        data_size = {'h':160, 'w':160, 'c':2, 'dataset':dataset}
        n_classes = 5

    elif dataset == "onePerson_3d":
        #data_size = {'n_slice':25, 'h':160, 'w':160, 'c':2, 'dataset':dataset}
        data_size = {'n_slice':32, 'h':160, 'w':160, 'c':2, 'dataset':dataset}
        train_set = onePerson_3d('../../prostate_pirads/phase4/npy/onePerson_3d', '../../prostate_pirads/phase4/npy/onePerson_3d_train.txt', data_size)
        test_set = onePerson_3d('../../prostate_pirads/phase4/npy/onePerson_3d', '../../prostate_pirads/phase4/npy/onePerson_3d_test.txt', data_size)
        n_classes = 4

    elif dataset == "onePerson_3d_ROI":
        data_size = {'n_slice':32, 'h':160, 'w':160, 'c':4, 'dataset':dataset}
        train_set = onePerson_3d('../../prostate_pirads/phase4/npy/onePerson_3d_ROI', '../../prostate_pirads/phase4/npy/onePerson_3d_train.txt', data_size)
        test_set = onePerson_3d('../../prostate_pirads/phase4/npy/onePerson_3d_ROI', '../../prostate_pirads/phase4/npy/onePerson_3d_test.txt', data_size)
        n_classes = 4


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, test_loader, data_size, n_classes


DATASET_CONFIGS = {
    'mnist': {'size': 28, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
}


