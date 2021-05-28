import torch
import torchvision as tv
import torchtext as tt

from typing import *
from logzero import logger
from torch.utils.data import ConcatDataset, Subset


def get_image_data(config: Dict) -> ConcatDataset:
    """
    Load torchvision data, both training and tuning, and return a concatenated Dataset object.
    Splitting occurs further downstream (in holdout class methods)

    Args:
        config: Configuration dictionary

    Returns:
        torch.utils.data.Datasets
    """
    logger.info(f"Downloading {config['data_config']['dataset']} data to {config['data_config']['data_kwargs']['root']}")
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    #If specified, select only a subset for faster running (TAKES DOUBLE THE NUMBER IN CONFIG)
    subset = config["data_config"].get("subset", None)
    if isinstance(subset, int):
        subset_indices = list(range(subset))
        logger.info(f"Using a subset of the dataset sized: {subset}")
    else:
        subset_indices = []

    try:
        if config["data_config"]["dataset"].casefold() == "mnist":
            for x in [True, False]:
                tv_dataset = tv.datasets.MNIST(root=config["data_config"]["data_kwargs"]["root"],
                                               download=config["data_config"]["data_kwargs"]["download"],
                                               train=x,
                                               transform=transform)
        elif config["data_config"]["dataset"].casefold() == "cifar10":
            for x in [True, False]:
                tv_dataset = tv.datasets.CIFAR10(root=config["data_config"]["data_kwargs"]["root"],
                                                 download=config["data_config"]["data_kwargs"]["download"],
                                                 train=x,
                                                 transform=transform)
        elif config["data_config"]["dataset"].casefold() == "imagenet":
            for x in ["train", "val"]:
                tv_dataset = tv.datasets.ImageNet(root=config["data_config"]["data_kwargs"]["root"],
                                                  download=config["data_config"]["data_kwargs"]["download"],
                                                  train=x,
                                                  transform=transform)
        else:
            raise NotImplementedError(f"{config['data_config']['dataset']} is not a dataset")
    except Exception as e:
        raise e

    to_concat = []
    to_concat_targets = []
    if isinstance(subset, int):
        to_concat.append(Subset(tv_dataset, subset_indices))
        to_concat_targets.append(tv_dataset.targets[:subset])
    else:
        to_concat.append(tv_dataset)
        to_concat_targets.append(tv_dataset.targets)

    # In case targets are not a tensor (like in CIFAR10)
    to_concat_targets = [torch.tensor(x) for x in to_concat_targets]

    dataset = ConcatDataset(to_concat)
    dataset.targets = torch.cat(to_concat_targets)

    return dataset

def get_text_data(config: Dict) -> ConcatDataset:

    logger.info(f"Downloading {config['data_config']['dataset']} data to {config['data_config']['data_kwargs']['root']}")
    transform = tt.transforms.Compose([tv.transforms.ToTensor()])


