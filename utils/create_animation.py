import argparse
import imageio

from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize


def animation_from_result(result_dir, step_size, out_path):
    samples = (Path(result_dir) / 'samples').glob('fakes_*')
    samples = list(sorted(samples))
    samples = samples[::step_size]

    if args.n_duplicate_last_frame:
        samples += [samples[-1]] * args.n_duplicate_last_frame

    frames = (imageio.imread(sample) for sample in samples)

    if args.resize_to:
        size = int(args.resize_to)
        frames = [resize(frame, (size, size), anti_aliasing=False) for frame in frames]

    if args.progress:
        frames = tqdm(frames, total=len(samples))

    options = {}
    if args.save_options:
        options.update(dict(eval(args.save_options)))

    imageio.mimwrite(out_path, frames, **options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str,
                        help='Path to a result directory..')
    parser.add_argument('--step-size', type=int, default=5,
                        help='Steps between each frame in the animation.')
    parser.add_argument('--out', type=str, default='binary_mnist.gif',
                        help='Path output file. Defaults to binary_mnist.gif')
    parser.add_argument('--progress', '-p', action='store_true',
                        help='Whether to display the progress.')
    parser.add_argument('--save-options', type=str,
                        help='Dictionary of options passed to imageio.mimwrite. '
                             'See https://imageio.readthedocs.io/en/stable/formats.html')
    parser.add_argument('--resize-to')
    parser.add_argument('--n-duplicate-last-frame', '-nlast', type=int, default=20,
                        help='Number of times to duplicate the last frame.')
    args = parser.parse_args()

    animation_from_result(args.result_dir, args.step_size, args.out)
