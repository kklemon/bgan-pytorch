import torch.nn as nn
import torch.nn.functional as F


class DCGANBase(nn.Module):
    def __init__(self, input_dim, output_dim, block_config, block_builder):
        super().__init__()

        channels = [input_dim] + block_config

        blocks = []
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            blocks += block_builder(in_channels, out_channels, is_first=not i)

        self.blocks = nn.Sequential(*blocks)
        if output_dim:
            self.final = nn.Conv2d(channels[-1], output_dim, kernel_size=1)
        else:
            self.final = None

    def forward(self, input):
        out = self.blocks(input)
        if self.final is not None:
            out = self.final(out)
        return out


class DCGANGenerator(DCGANBase):
    def __init__(self, latent_dim, output_dim, block_config, activation=nn.ReLU, norm=nn.BatchNorm2d):
        def block(in_channels, out_channels, is_first=False, **kwargs):
            blocks = [
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=4,
                    stride=1 if is_first else 2,
                    padding=0 if is_first else 1
                ),
                activation()
            ]
            if norm:
                blocks.append(norm(out_channels))
            return blocks

        super().__init__(latent_dim, output_dim, block_config, block)

        self.latent_dim = latent_dim


class DCGANDiscriminator(DCGANBase):
    def __init__(self, input_dim, block_config, activation=nn.ReLU, norm=None):
        def block(in_channels, out_channels, **kwargs):
            blocks = [
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                activation()
            ]
            if norm:
                blocks.append(norm(out_channels))
            return blocks

        super().__init__(input_dim, None, block_config, block)

        self.final = nn.Linear(4 * block_config[-1], 1)

    def forward(self, input):
        out = self.blocks(input)
        out = out.reshape(out.size(0), -1)
        out = self.final(out).squeeze()
        return out
