import time
import torch
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader


class Model:
    def __init__(self,
                 G, D,
                 G_opt, D_opt,
                 loss_f,
                 dataset,
                 batch_size,
                 device,
                 sample_folder,
                 checkpoint_folder,
                 n_sample=16,
                 n_mc_samples=20,
                 num_workers=4):
        self.G = G.to(device)
        self.D = D.to(device)

        self.G_opt = G_opt
        self.D_opt = D_opt

        assert dataset.num_colors > 1, 'num_colors property of dataset must be > 1'

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size,
                                     shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

        self.sample_folder = sample_folder
        self.checkpoint_folder = checkpoint_folder

        self.loss_f = loss_f
        self.device = device
        self.n_mc_samples = n_mc_samples
        self.n_sample = n_sample

    def train(self, epochs, log_every=10, sample_every=10):
        z_sample = torch.randn(self.n_sample, self.G.latent_dim, 1, 1).to(self.device)

        for epoch in range(epochs):
            g_loss_sum = 0
            d_loss_sum = 0

            p_fake_sum = 0
            p_real_sum = 0

            start_time = time.time()

            for step, reals in enumerate(self.dataloader):
                global_step = epoch * len(self.dataloader) + step

                reals = reals_input = reals.to(self.device)
                if self.dataset.num_colors > 2:
                    reals_input = F.one_hot(reals_input, num_classes=self.dataset.num_colors)
                    reals_input = reals_input.permute(0, 3, 1, 2).type(torch.float)

                z = torch.randn(reals.size(0), self.G.latent_dim, 1, 1).to(self.device)

                fake_logits = self.G(z)

                d_loss, g_loss, p_fake, p_real = self.loss_f(self.D, fake_logits, reals_input,
                                                             n_samples=self.n_mc_samples)

                self.D_opt.zero_grad()
                self.G_opt.zero_grad()

                torch.autograd.backward([d_loss, g_loss])

                self.D_opt.step()
                self.G_opt.step()

                g_loss_sum += float(g_loss)
                d_loss_sum += float(d_loss)

                p_fake_sum += float(p_fake)
                p_real_sum += float(p_real)

                if step % log_every == 0:
                    cur_step = min(step + 1, log_every)
                    batches_per_sec = cur_step / (time.time() - start_time)

                    print(f'[EPOCH {epoch + 1:03d}] [{step:04d} / {len(self.dataloader):04d}] '
                          f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, '
                          f'p_fake: {p_fake_sum / cur_step:.5f}, p_real: {p_real_sum / cur_step:.5f}, '
                          f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0
                    p_fake_sum = p_real_sum = 0

                    start_time = time.time()

                if step % sample_every == 0:
                    logits = self.G(z_sample)
                    if logits.size(1) == 1:
                        samples = torch.sigmoid(logits)
                    else:
                        samples = logits.argmax(1)

                    samples = self.dataset.dequantize(samples)
                    reals = self.dataset.dequantize(reals[:self.n_sample])

                    samples_path = str(self.sample_folder / f'fakes_{global_step:06d}.png')
                    reals_path = str(self.sample_folder / f'reals_{global_step:06d}.png')

                    torchvision.utils.save_image(samples, samples_path, nrow=4, normalize=True)
                    torchvision.utils.save_image(reals, reals_path, nrow=4, normalize=True)

            torch.save(self.G, str(self.checkpoint_folder / f'G_{global_step:06d}.pth'))
            torch.save(self.D, str(self.checkpoint_folder / f'D_{global_step:06d}.pth'))
