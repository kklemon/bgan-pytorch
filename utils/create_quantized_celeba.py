import argparse
import numpy as np

from pathlib import Path

from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from skimage import io

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms


def main(args):
    if args.num_colors > 256:
        raise ValueError('--num-colors must be <= 256.')

    celeba = CelebA(args.source_path, split='all', transform=transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ]))

    dataloader = DataLoader(celeba, batch_size=args.batch_size, num_workers=args.num_workers)

    def get_batches():
        yield from map(lambda batch: batch[0].permute(0, 2, 3, 1).numpy(), dataloader)

    k_means = MiniBatchKMeans(args.num_colors, batch_size=args.batch_size, compute_labels=False)

    num_palette_samples = args.num_palette_samples
    if not num_palette_samples:
        num_palette_samples = args.batch_size * len(dataloader)

    print(f'Computing color palette using {num_palette_samples} samples')
    with tqdm(total=num_palette_samples) as pbar:
        for batch_idx, batch in enumerate(get_batches()):
            k_means.partial_fit(batch.reshape(-1, 3))

            pbar.update(args.batch_size)

            if (batch_idx + 1) * args.batch_size > num_palette_samples:
                break

    target_path = Path(args.target_path)
    target_img_dir = target_path / 'images'
    if not target_img_dir.exists():
        target_img_dir.mkdir()

    np.save(str(target_path / 'palette'), k_means.cluster_centers_)

    print('Quantizing images')
    with tqdm(total=len(celeba)) as pbar:
        for batch_idx, batch in enumerate(get_batches()):
            result = k_means.predict(batch.reshape(-1, 3))
            result = result.reshape(batch.shape[:3]).astype('uint8')

            for i, img in enumerate(result):
                img_idx = batch_idx * args.batch_size + i + 1
                save_path = target_img_dir / f'{img_idx:06d}.png'
                io.imsave(str(save_path), img, check_contrast=False)

            pbar.update(args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''
        Creates a quantized version of the CelebA dataset.
        The number of colors in the target palette can be set using the ---num-colors argument.
        
        The resulting images are stored in the PNG file format for convenience and lower space usage.
        Note, that this has the implication, that the number of colors is limited to a maximum of 256.
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('source_path',
                        help='Path to the root folder of the CelebA dataset.')
    parser.add_argument('target_path',
                        help='Where to store the quantized images.')
    parser.add_argument('--size', type=int, default=64,
                        help='To which size to resize the source images.')
    parser.add_argument('--num-colors', type=int, default=16,
                        help='Number of colors in the target palette. Must be <= 256.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for the mini-batch K-Means algorithm used for the quantization.')
    parser.add_argument('--num-palette-samples', type=int, default=10_000,
                        help='Number of samples to use for computing the color palette.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for the data loader.')
    args = parser.parse_args()

    main(args)
