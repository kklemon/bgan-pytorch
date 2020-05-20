import torch
import torch.nn.functional as F


def compute_norm_weights(log_w):
    log_n = torch.log(torch.tensor(float(log_w.shape[0])))
    log_z_est = torch.logsumexp(log_w - log_n, dim=0)
    log_w_tilde = log_w - log_z_est - log_n

    w_tilde = torch.exp(log_w_tilde)

    return w_tilde


def binary_bgan_loss(D, fake_logits, reals, n_samples=8):
    batch_size, num_channels = fake_logits.shape[:2]
    spatial_dims = fake_logits.shape[2:]

    # Draw samples according to G's output densities
    fake_logits = fake_logits.unsqueeze(0)
    samples = torch.rand(n_samples, batch_size, num_channels, *spatial_dims, device=fake_logits.device)
    samples = (samples <= torch.sigmoid(fake_logits)).type(torch.float)

    real_out = D(reals)
    fake_out = D(samples.view(-1, 1, *spatial_dims))

    log_w = fake_out.view(n_samples, batch_size)
    log_g = -((1.0 - samples) * fake_logits + F.softplus(-fake_logits)).mean(dim=(2, 3, 4))

    w_tilde = compute_norm_weights(log_w).detach()

    d_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out)) + \
             F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
    g_loss = -(w_tilde * log_g).sum(0).mean()

    p_fake = (fake_out < 0).type(torch.float).mean().detach()
    p_real = (real_out > 0).type(torch.float).mean().detach()

    return d_loss, g_loss, p_fake, p_real


def multinomial_bgan_loss(D, fake_logits, reals, n_samples=8):
    batch_size = reals.size(0)
    n_channels = fake_logits.size(1)
    spatial_dims = fake_logits.shape[2:]

    fake_p = torch.softmax(fake_logits, dim=1).view(batch_size, n_channels, -1).transpose(1, 2)
    fake_p = fake_p.repeat(n_samples, 1, 1).view(-1, n_channels)

    samples = torch.multinomial(fake_p, num_samples=1)
    samples = samples.view(n_samples, batch_size, *spatial_dims)

    samples_one_hot = F.one_hot(samples, num_classes=n_channels).type(torch.float)
    samples_one_hot = samples_one_hot.permute(0, 1, 4, 2, 3)

    real_out = D(reals)
    fake_out = D(samples_one_hot.view(-1, *samples_one_hot.shape[2:]))

    log_w = fake_out.view(n_samples, batch_size)
    log_g = -(samples_one_hot * (fake_logits - torch.logsumexp(fake_logits, dim=1, keepdim=True)).unsqueeze(0)).mean(dim=(2, 3, 4))

    w_tilde = compute_norm_weights(log_w).detach()

    d_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out)) + \
             F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
    g_loss = (w_tilde * log_g).sum(0).mean()

    p_fake = (fake_out < 0).type(torch.float).mean().detach()
    p_real = (real_out > 0).type(torch.float).mean().detach()

    return d_loss, g_loss, p_fake, p_real
