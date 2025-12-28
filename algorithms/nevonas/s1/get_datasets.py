import json

import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
import ut

from config_utils import load_config, dict2config

#def get_dataloader(dataset, batch_size, valid_batch_size, cutout, cutout_length, autoaug=False):
def get_dataloader(args):
  if args.dataset == 'cifar10':
    train_transform, valid_transform = ut._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root = args.data_dir, train = True, download = True, transform = train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 96
        num_train = split + 96
    print(f"Training samples: {split}, Validation samples: {num_train - split}")

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True)

  elif args.dataset == 'cifar100':
    train_transform, valid_transform = ut.data_transforms_cifar100(args.cutout, args.cutout_length, args.autoaug)
    train_data = dset.CIFAR100(root=args.data_dir, train=True , transform=train_transform, download=True)
    assert len(train_data) == 50000, "Something wrong with the dataset loading"

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 96
        num_train = split + 96
    print(f"Training samples: {split}, Validation samples: {num_train - split}")

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True)

  return train_transform, valid_transform, train_queue, valid_queue
