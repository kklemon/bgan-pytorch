import argparse
import torch

from bgan.dcgan import DCGANGenerator, DCGANDiscriminator
from bgan.utils import get_default_device, create_result_dir, get_activation_by_name, apply_spectral_norm, init_weights
from bgan.losses import binary_bgan_loss, multinomial_bgan_loss
from bgan.model import Model
from config import configs


parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['disc_mnist', 'disc_celeba'])
parser.add_argument('--run-name')
parser.add_argument('--data-path', default='data')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--latent-dim', type=int, default=64)
parser.add_argument('--g-lr', type=float, default=0.0001)
parser.add_argument('--d-lr', type=float, default=0.0001)
parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'elu'])
parser.add_argument('--spectral-norm', action='store_true')
parser.add_argument('--log-every', type=int, default=100)
parser.add_argument('--sample-every', type=int, default=100)
parser.add_argument('--n-sample', type=int, default=16)
parser.add_argument('--n-mc-samples', type=int, default=20)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--device')
args = parser.parse_args()


device = get_default_device(args.device)

config = configs[args.dataset]

dataset = config.dataset_factory(args.data_path)

result_dir, sample_dir, checkpoint_dir = create_result_dir(args.run_name or args.dataset)

activation = get_activation_by_name(args.activation or config.activation)

if dataset.num_colors <= 2:
    dim = 1
    loss_f = binary_bgan_loss
else:
    dim = dataset.num_colors
    loss_f = multinomial_bgan_loss

G = DCGANGenerator(args.latent_dim, dim, config.G_blocks, activation=activation)
D = DCGANDiscriminator(dim, config.D_blocks, activation=activation)

init_weights(G)
init_weights(D)

if args.spectral_norm:
    apply_spectral_norm(G)
    apply_spectral_norm(D)

G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

model = Model(
    G, D,
    G_opt, D_opt,
    loss_f=loss_f,
    dataset=dataset,
    batch_size=args.batch_size,
    device=device,
    sample_folder=sample_dir,
    checkpoint_folder=checkpoint_dir,
    n_sample=args.n_sample,
    n_mc_samples=args.n_mc_samples,
    num_workers=args.num_workers
)
model.train(args.epochs, args.log_every, args.sample_every)
