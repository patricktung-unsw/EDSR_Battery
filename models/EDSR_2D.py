import torch.nn as nn
import torch


class EDSRResBlock(nn.Module):
    def __init__(self, filters=64, kernel_size=3, padding='same', res_scaling=1):
        super(EDSRResBlock, self).__init__()

        self.res_scaling = res_scaling

        self.conv1 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        out = self.conv1(x)   # Conv
        out = self.relu(out)  # ReLU
        out = self.conv2(out) # Conv
        out = out * self.res_scaling
        out = out + x         # Addition/Skip connection

        return out

class Upsample(nn.Module):
    def __init__(self, filters=64, kernel_size=3, padding='same'):
        super(Upsample, self).__init__()

        scaling_factor = 2
        self.conv = nn.Conv2d(filters, filters*scaling_factor**2, kernel_size, padding=kernel_size//2)
        self.pixel_shuffle = nn.PixelShuffle(scaling_factor)

    def forward(self, x):
        out = self.conv(x) # Channels: 64 -> 256
        out = self.pixel_shuffle(out) # Channels: 256 -> 64

        return out

class MeanShift(nn.Module):
    def __init__(self, rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040), sign=-1):
        super(MeanShift, self).__init__()

        rgb_mean = sign * rgb_range * torch.Tensor(rgb_mean)
        self.register_buffer('rgb_mean', rgb_mean.view(1,3,1,1))

    def forward(self, x):
        return x + self.rgb_mean

class EDSR(nn.Module):
    def __init__(self, num_channels=3, n_resblock=16, filters=64, res_scaling=1, scale=4):
        super(EDSR, self).__init__()

        self.scale = scale

        # Define convolutional layers
        self.conv_in = nn.Conv2d(num_channels, filters, kernel_size=3, padding=3//2)
        self.conv_mid = nn.Conv2d(filters, filters, kernel_size=3, padding=3//2)
        self.conv_out = nn.Conv2d(filters, num_channels, kernel_size=3, padding=3//2)

        # Define ResBlock layers
        res_blocks = [EDSRResBlock(filters=filters, res_scaling=res_scaling) for _ in range(n_resblock)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Define Upscale layers
        self.upsample_x2 = Upsample(filters=filters)
        if scale == 4:
            self.upsample_x4 = Upsample(filters=filters)

        # Define normalization layers
        self.mean_sub = MeanShift(sign=-1)
        self.mean_add = MeanShift(sign=1)


    def forward(self, x):
        x = self.mean_sub(x) # Subtract DIV2k mean
        x = self.conv_in(x) # Channels: 3 -> 64

        out = self.res_blocks(x)
        out = self.conv_mid(out)
        out = out + x # Addition/Skip connection

        out = self.upsample_x2(out) # Upsample to X2
        if self.scale == 4:
            out = self.upsample_x4(out) # Upsample to X4

        out = self.conv_out(out) # Channels: 64 -> 3
        out = self.mean_add(out) # Add DIV2k mean

        return out