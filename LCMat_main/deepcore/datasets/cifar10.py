from torchvision import datasets, transforms
from torch import tensor, long
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import torch
def CIFAR10(data_path,args=None):
    if args != None:
        val = args.val
        val_ratio = args.val_ratio
    else:
        val = False
        val_ratio = 0
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    dst_valid = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes

    if val:
        train_indices, val_indices = train_test_split(list(range(len(dst_train.targets))), test_size=val_ratio,
                                                      stratify=dst_train.targets)
        dst_train.targets = tensor(dst_train.targets, dtype=long)
        dst_valid.targets = tensor(dst_valid.targets, dtype=long)
        dst_test.targets = tensor(dst_test.targets, dtype=long)

        dst_train.data = dst_train.data[train_indices]
        dst_train.targets = dst_train.targets[train_indices]
        # dst_train.semi_targets = dst_train.semi_targets[train_indices]

        dst_valid.data = dst_valid.data[val_indices]
        dst_valid.targets = dst_valid.targets[val_indices]
        # dst_valid.semi_targets = dst_valid.semi_targets[val_indices]


        return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_valid, dst_test

    else:
        dst_train.targets = tensor(dst_train.targets, dtype=long)
        dst_test.targets = tensor(dst_test.targets, dtype=long)
        return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test