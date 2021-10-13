import math

import torch


class Generator(torch.nn.Module):
    def __init__(self, z_dim, hidden_dim, image_dim=3, resolution=128):
        super(Generator, self).__init__()
        hd_multiplier = 8
        n_layers = int(math.log2(resolution)) - 1
        layers = list()
        for i in range(n_layers):
            stride = 2
            padding = 1
            is_last = False
            coeff_input = 2 ** (i - 1)
            coeff_output = 2 ** (i)
            if i == 0:
                stride = 1
                padding = 0
                coeff_input = (hd_multiplier * hidden_dim) / z_dim
            elif i == n_layers - 1:
                is_last = True
                coeff_output = (hd_multiplier * hidden_dim) / image_dim
            layers.append(self._create_layer(int(hidden_dim * hd_multiplier / coeff_input),
                                             int(hidden_dim * hd_multiplier / coeff_output), stride=stride,
                                             padding=padding, is_last=is_last))

        self.main = torch.nn.Sequential(*layers)

    def forward(self, noise_vector):
        return self.main(noise_vector)

    def _create_layer(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, is_last=False):
        ops = [torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)]
        if not is_last:
            ops += [torch.nn.BatchNorm2d(out_dim), torch.nn.ReLU(True)]
        else:
            ops += [torch.nn.Tanh()]

        return torch.nn.Sequential(*ops)


class Discriminator(torch.nn.Module):
    def __init__(self, image_dim, hidden_dim, resolution=128, is_wgan=False):
        super(Discriminator, self).__init__()
        # Previous version had 12 after 8
        layers = list()
        n_layers = int(math.log2(resolution)) - 1
        for i in range(n_layers):
            coeff_input = 2 ** (i - 1)
            coeff_output = 2 ** i
            stride = 2
            padding = 1
            is_first = False
            is_last = False
            if i == 0:
                is_first = True
                coeff_input = image_dim / hidden_dim
            elif i == n_layers - 1:
                stride = 1
                padding = 0
                is_last = True
                coeff_output = 1 / hidden_dim

            layers.append(self._create_layer(int(hidden_dim * coeff_input),
                                             int(hidden_dim * coeff_output), stride=stride,
                                             padding=padding, is_last=is_last, is_first=is_first,
                                             is_wgan=is_wgan))

        self.main = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    def _create_layer(self, in_dim, out_dim, stride=2, padding=1, is_last=False, is_first=False, is_wgan=False):
        ops = [torch.nn.Conv2d(in_dim, out_dim, 4, stride, padding, bias=False)]
        if not is_last and not is_first:
            norm_op = torch.nn.BatchNorm2d if not is_wgan else torch.nn.InstanceNorm2d
            ops += [norm_op(out_dim), torch.nn.LeakyReLU(0.2, inplace=True)]
        elif is_last and not is_wgan:
            ops += [torch.nn.Sigmoid()]
        elif is_first:
            ops += [torch.nn.LeakyReLU(0.2, inplace=True)]

        return torch.nn.Sequential(*ops)
