import numpy as np
import torch

from pathlib import Path
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class BaseQuantizedImageDataset(Dataset):
    @property
    def num_colors(self):
        raise NotImplementedError

    def dequantize(self, img):
        raise NotImplementedError


class QuantizedImageDataset(BaseQuantizedImageDataset):
    def __init__(self, root):
        super().__init__()

        self.root = Path(root)
        self.image_paths = list((self.root / 'images').glob('*.png'))

        self.palette = torch.tensor(np.load(str(self.root / 'palette.npy')))
        self._num_colors = self.palette.size(0)

    @property
    def num_colors(self):
        return self._num_colors

    def __getitem__(self, idx):
        img = io.imread(str(self.image_paths[idx]))
        return torch.tensor(img, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def dequantize(self, img):
        return self.palette[img].permute(0, 3, 1, 2)


class BinaryMNIST(BaseQuantizedImageDataset, MNIST):
    def __init__(self, path, img_size=(32, 32)):
        super().__init__(
            path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                lambda t: (t > 0.5).type(torch.float)
            ]),
        )

    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]

    @property
    def num_colors(self):
        return 2

    def dequantize(self, img):
        return img
