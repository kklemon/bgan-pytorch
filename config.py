from easydict import EasyDict as edict
from bgan.datasets import QuantizedImageDataset, BinaryMNIST


configs = {
    'disc_mnist': edict(
        dataset_factory=lambda data_path: BinaryMNIST(data_path, img_size=(32, 32)),
        G_blocks=[128, 64, 32, 16],
        D_blocks=[16, 32, 64, 128],
        activation='leaky_relu',
    ),
    'disc_celeba': edict(
        dataset_factory=lambda data_path: QuantizedImageDataset(data_path),
        G_blocks=[256, 128, 64, 32],
        D_blocks=[32, 64, 128, 256],
        activation='elu'
    )
}