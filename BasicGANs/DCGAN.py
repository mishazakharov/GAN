import os
import math
import random

import glob
import torch

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from loss_functions import get_minimax_loss


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
    def __init__(self, image_dim, hidden_dim, resolution=128):
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
                                             padding=padding, is_last=is_last, is_first=is_first))

        self.main = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    def _create_layer(self, in_dim, out_dim, stride=2, padding=1, is_last=False, is_first=False):
        ops = [torch.nn.Conv2d(in_dim, out_dim, 4, stride, padding, bias=False)]
        if not is_last and not is_first:
            ops += [torch.nn.BatchNorm2d(out_dim), torch.nn.LeakyReLU(0.2, inplace=True)]
        elif is_last:
            ops += [torch.nn.Sigmoid()]
        elif is_first:
            ops += [torch.nn.LeakyReLU(0.2, inplace=True)]

        return torch.nn.Sequential(*ops)


if __name__ == "__main__":
    data_path = glob.glob("/home/misha/datasets/20_WIDE_NUMAKT/*/*/images/*.jpg")
    log_folder = "second_experiment"
    # Batch size during training
    batch_size = 128
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = (128, 128)
    # Number of channels in the training images. For color images this is 3
    image_dim = 3
    # Size of z latent vector (i.e. size of generator input)
    z_dim = 100
    # Size of feature maps in GAN
    hidden_dim = 64
    # Learning rate for optimizers
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    num_images = 64
    num_workers = 0
    device = "cuda:0"
    seed = 5000

    # For reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, "SavedModels"), exist_ok=True)

    dataset = CustomDataset(data_path, image_size,
                            transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    net_G = Generator(z_dim, hidden_dim, image_dim, image_size[0]).to(device)
    net_G.apply(weights_init)

    net_D = Discriminator(image_dim, hidden_dim, image_size[0]).to(device)
    net_D.apply(weights_init)

    disc_params = sum([p.numel() for p in net_D.parameters() if p.requires_grad])
    gen_params = sum([p.numel() for p in net_G.parameters() if p.requires_grad])
    print("Generator parameters: {0} \nDiscriminator parameters: {1}".format(gen_params, disc_params))

    # Initialize BCELoss function
    criterion = torch.nn.BCELoss()

    fixed_noise = torch.randn(num_images, z_dim, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr, betas=(beta_1, beta_2))

    tensorboard = SummaryWriter(
        os.path.join(log_folder, "tensorboard_logs"))

    global_step = 1

    print("Starting Training Loop...")
    while True:
        for i, batch in enumerate(data_loader, 0):
            # Update discriminator's weights
            net_D.zero_grad()
            real_batch = batch["image"].to(device)
            b_size = real_batch.shape[0]
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            noise_batch = torch.randn(b_size, z_dim, 1, 1, device=device)
            D_error, D_G_z1, D_x, fake_batch = get_minimax_loss(net_G, net_D, criterion, label, real_batch,
                                                                noise_batch, fake_label)
            optimizer_D.step()

            # Update generator's weights
            net_G.zero_grad()
            label.fill_(real_label)
            G_error, D_G_z2 = get_minimax_loss(net_G, net_D, criterion, label, fake_batch, discriminator=False)
            optimizer_G.step()

            # Output training stats
            if global_step % 100 == 0:
                save_checkpoint(
                    os.path.join(log_folder, "checkpoints", "checkpoints.tar"), net_G, net_D, optimizer_G, optimizer_D)
                write_logs({"Global Step": global_step, "G_Loss": G_error.item(), "D_Loss": D_error.item(),
                            "D(X)": D_x, "D(G(Z))": [D_G_z1, D_G_z2]}, tensorboard, global_step)

            # Check how the generator is doing by saving G's output on fixed_noise
            if global_step % 500 == 0:
                generate_with_fixed_noise(net_G, fixed_noise, tensorboard, global_step)
                save_weight(os.path.join(log_folder, "SavedModels", str(global_step) + ".pth"), net_G)

            global_step += 1
