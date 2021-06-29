import torch
import torchvision as tv
import torchtext as tt

from typing import *
from logzero import logger
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from itertools import tee
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import ConcatDataset, IterableDataset, ChainDataset, Subset

from utils.utilities import initialize_iterable_dataset
from data.text_preprocessing import CustomIterableDataset


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
    subset = config["data_config"].get("data_subset", None)
    if isinstance(subset, int):
        subset_indices = list(range(subset))
        logger.info(f"Using a subset of the dataset sized: {subset}")
    else:
        subset_indices = []

    all_datasets = []
    try:
        if config["data_config"]["dataset"].casefold() == "mnist":
            for x in [True, False]:
                all_datasets.append(tv.datasets.MNIST(root=config["data_config"]["data_kwargs"]["root"],
                                                       download=config["data_config"]["data_kwargs"]["download"],
                                                       train=x,
                                                       transform=transform))
        elif config["data_config"]["dataset"].casefold() == "cifar10":
            for x in [True, False]:
                all_datasets.append(tv.datasets.CIFAR10(root=config["data_config"]["data_kwargs"]["root"],
                                                         download=config["data_config"]["data_kwargs"]["download"],
                                                         train=x,
                                                         transform=transform))
        elif config["data_config"]["dataset"].casefold() == "imagenet":
            for x in ["train", "val"]:
                all_datasets.append(tv.datasets.ImageNet(root=config["data_config"]["data_kwargs"]["root"],
                                                          download=config["data_config"]["data_kwargs"]["download"],
                                                          train=x,
                                                          transform=transform))
        else:
            raise NotImplementedError(f"{config['data_config']['dataset']} is not a dataset")
    except Exception as e:
        raise e

    to_concat = []
    to_concat_targets = []
    for tv_dataset in all_datasets:
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


def get_graph_data(config: Dict) -> ConcatDataset:
    pass


def get_text_data(config: Dict) -> ChainDataset:

    logger.info(f"Downloading {config['data_config']['dataset']} data to {config['data_config']['data_kwargs']['root']}")

    #If specified, select only a subset for faster running (TAKES DOUBLE THE NUMBER IN CONFIG)
    subset = config["data_config"].get("subset", None)
    if isinstance(subset, int):
        subset_indices = list(range(subset))
        logger.info(f"Using a subset of the dataset sized: {subset}")
    else:
        subset_indices = []

    all_datasets: List[torch.utils.data.IterableDataset] = initialize_iterable_dataset(config)

    # to_concat = []
    # for tt_dataset in all_datasets:
    #     if isinstance(subset, int):
    #         to_concat.append(Subset(tt_dataset, subset_indices))
    #     else:
    #         to_concat.append(tt_dataset)

    dataset = ChainDataset(all_datasets)

    # Following the tokenization and vocab building spec in PyTorch tutorial
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, dataset))  # Will automatically use Counter and Vocab objects
    vocab.set_default_index(vocab["<unk>"])

    all_datasets: List[torch.utils.data.IterableDataset] = initialize_iterable_dataset(config)

    # Tokenize and wrap with IterableDataset for Chaining
    datasets = []
    for dataset in all_datasets:
        dataset = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in dataset]
        datasets.append(CustomIterableDataset(torch.cat(tuple(filter(lambda t: t.numel() > 0, dataset)))))

    dataset = ChainDataset(datasets)
    dataset.vocab = vocab

    return dataset


