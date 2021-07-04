import torch


class CustomIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets."""
    def __init__(self, iterator):
        """Initiate the dataset abstraction."""
        super(CustomIterableDataset, self).__init__()
        self._iterator = iterator
        self.num_lines = len(iterator)

    def __iter__(self):
        return iter(self._iterator)

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos





