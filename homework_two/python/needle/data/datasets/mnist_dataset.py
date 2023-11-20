from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        f = gzip.open(image_filename,'rb')

        # read magic number, image count, size
        magic, image_count, dim1, dim2 = struct.unpack(">IIII", f.read(16))

        image_content = np.frombuffer(f.read(), dtype=np.dtype(np.uint8))
        image_content = image_content.reshape(image_count, dim1 * dim2)
        image_content = image_content / 255
        self.image_content = image_content.astype(np.float32)

        f = gzip.open(label_filename,'rb')
        magic, label_count = struct.unpack(">II", f.read(8))
        self.label_content = np.frombuffer(f.read(), dtype=np.dtype(np.uint8))

        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.image_content[index]
        label = self.label_content[index]


        if self.transforms is not None:
            image = np.reshape(image, (28,28,1))
            for t in self.transforms:
                image = t(image)


        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.image_content)
        ### END YOUR SOLUTION