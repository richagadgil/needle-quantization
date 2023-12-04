import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.ordering = None


    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        else:
            self.ordering = np.array_split(np.random.permutation(np.arange(len(self.dataset))), 
                                            range(self.batch_size, len(self.dataset), self.batch_size))
        
        self.current_index = 0 
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION

        num_batches = len(self.dataset) // self.batch_size 

        if self.current_index >= num_batches:
            raise StopIteration  # No more batches to fetch

        if self.ordering is not None:
            batch_indices = self.ordering[self.current_index]

        batch = [self.dataset[i] for i in batch_indices]
        dims = len(batch[0])

        
        t = []
        for dim in range(0, dims):
            t.append(Tensor([i[dim] for i in batch]))

        self.current_index += 1  

        return tuple(t)

        ### END YOUR SOLUTION
