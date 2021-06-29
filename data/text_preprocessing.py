import torch


class CustomIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets."""
    def __init__(self, iterator):
        """Initiate the dataset abstraction."""
        super(CustomIterableDataset, self).__init__()
        self._iterator = iterator
        self.num_lines = len(iterator)
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos





