import os
import random

import cv2
import glob
import torch

import numpy as np
import torchvision.transforms as transforms

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import CustomDataset, save_checkpoint, save_weight, weights_init, get_noise
from loss_functions import get_wasserstein_loss


class Generator(nn.Module):

    def __init__(self, z_dim, hidden_dim, image_dim=3):
        super(Generator, self).__init__()
        self.first_block = self.generate_upsample_block(z_dim, hidden_dim * 8, 4, 1, 0)
        self.second_block = self.generate_upsample_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1)
        self.third_block = self.generate_upsample_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1)
        self.fourth_block = self.generate_upsample_block(hidden_dim * 2, hidden_dim * 1, 4, 2, 1)
        self.fifth_block = self.generate_upsample_block(hidden_dim * 1, hidden_dim // 2, 4, 2, 1)
        self.sixth_block = self.generate_upsample_block(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.seventh_block = self.generate_upsample_block(hidden_dim // 4, hidden_dim // 8, 4, (1, 2), 1)
        self.eighth_block = self.generate_upsample_block(hidden_dim // 8, image_dim, 4, (1, 2), 1, last=True)
        self.pool = torch.nn.AdaptiveAvgPool2d((128, 512))

    def forward(self, noise_vector):
        output = self.first_block(noise_vector)
        output = self.second_block(output)
        output = self.third_block(output)
        output = self.fourth_block(output)
        output = self.fifth_block(output)
        output = self.sixth_block(output)
        output = self.seventh_block(output)
        output = self.eighth_block(output)
        output = self.pool(output)

        return output

    def generate_upsample_block(self, input_dim, output_dim, *args, last=False):
        if last:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(input_dim, output_dim, *args, bias=False),
                torch.nn.Tanh()
            )

        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_dim, output_dim, *args, bias=False),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(True)
        )


class Discriminator(torch.nn.Module):

    def __init__(self, image_dim, hidden_dim):
        super().__init__()
        self.first_block = self.generate_downsample_block(image_dim, hidden_dim, 4, 2, 1, first=True)
        self.second_block = self.generate_downsample_block(hidden_dim, hidden_dim * 2, 4, 2, 1)
        self.third_block = self.generate_downsample_block(hidden_dim * 2, hidden_dim * 4, 4, 2, 1)
        self.fourth_block = self.generate_downsample_block(hidden_dim * 4, hidden_dim * 8, 4, 2, 1)
        self.fifth_block = self.generate_downsample_block(hidden_dim * 8, hidden_dim * 12, 4, 2, 1)
        self.sixth_block = self.generate_downsample_block(hidden_dim * 12, hidden_dim * 8, 4, 2, 1)
        self.seventh_block = self.generate_downsample_block(hidden_dim * 8, hidden_dim * 6, 4, 2, 1)
        self.eighth_block = self.generate_downsample_block(hidden_dim * 6, 1, (1, 4), 1, 0, last=True)

    def forward(self, x):
        output = self.first_block(x)
        output = self.second_block(output)
        output = self.third_block(output)
        output = self.fourth_block(output)
        output = self.fifth_block(output)
        output = self.sixth_block(output)
        output = self.seventh_block(output)
        output = self.eighth_block(output)

        return output

    def generate_downsample_block(self, input_dim, output_dim, *args, first=False, last=False):
        # No BatchNorm2d!
        if not first and not last:
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_dim, output_dim, *args, bias=False),
                torch.nn.InstanceNorm2d(output_dim, affine=True),
                torch.nn.LeakyReLU(0.2, inplace=True)
            )
        elif first:
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_dim, output_dim, *args, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True)
            )
        elif last:
            # No SIGMOID!
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_dim, output_dim, *args, bias=False),
            )


if __name__ == "__main__":
    data_path = glob.glob(
        "/home/misha/datasets/passports_word_recognition/images/*.jpg")
    log_folder = "test"
    # Batch size during training
    batch_size = 32
    # Spatial size of training images
    image_size = (512, 128)
    # Number of channels in the training images. For color images this is 3
    image_dim = 3
    # Size of z latent vector (i.e. size of generator input)
    z_dim = 100
    # Size of feature maps in GAN
    hidden_dim = 64
    # Learning rate for optimizers
    lr = 0.0001
    # Beta1 hyperparam for Adam optimizers
    beta_1 = 0
    beta_2 = 0.9
    c_lambda = 10
    disc_repeats = 1  # 5
    device = "cuda:0"
    seed = 5000

    # For reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, "SavedModels"), exist_ok=True)

    # Data preparation
    dataset = CustomDataset(data_path, image_size,
                            transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    net_G = Generator(z_dim, hidden_dim, image_dim).to(device)
    net_G.apply(weights_init)

    net_D = Discriminator(image_dim, hidden_dim).to(device)
    net_D.apply(weights_init)

    disc_params = sum([p.numel() for p in net_D.parameters() if p.requires_grad])
    gen_params = sum([p.numel() for p in net_G.parameters() if p.requires_grad])
    print("Generator parameters: {0} \nDiscriminator parameters: {1}".format(gen_params, disc_params))

    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=(beta_1, beta_2))

    tensorboard = SummaryWriter(
        os.path.join(log_folder, "tensorboard_logs"))

    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    global_step = 1
    while True:
        for i, batch in enumerate(data_loader):
            real_batch = batch["image"].to(device)
            batch_size = real_batch.shape[0]

            D_error_mean = 0
            for _ in range(disc_repeats):
                # Update discriminator to optimality!
                optimizer_D.zero_grad()
                fake_noise = get_noise(batch_size, z_dim, device=device).unsqueeze(2).unsqueeze(2)
                D_x, D_G_z1, gradient_penalty = get_wasserstein_loss(net_G, net_D, real_batch, fake_noise)
                D_error = -D_x + D_G_z1
                D_error_mean += D_error.item() / disc_repeats
                # Update optimizer
                optimizer_D.step()

            # Update generator with optimal gradients!
            optimizer_G.zero_grad()
            fake_noise_1 = get_noise(batch_size, z_dim, device=device).unsqueeze(2).unsqueeze(2)
            G_error, D_G_z2 = get_wasserstein_loss(net_G, net_D, noise_vector=fake_noise_1, discriminator=False)

            # Update the weights
            optimizer_G.step()

            if global_step % 100 == 0:
                D_x = D_x.mean().item()
                D_G_z1 = D_G_z1.mean().item()
                D_G_z2 = D_G_z2.mean().item()
                tensorboard.add_scalar("G_LOSS", G_error.item(), global_step)
                tensorboard.add_scalar("D_LOSS", D_error.item(), global_step)
                tensorboard.add_scalar("D(X)", D_x, global_step)
                tensorboard.add_scalar("D(G(Z))", D_G_z1, global_step)
                save_checkpoint(
                    os.path.join(log_folder, "checkpoints", "checkpoints.tar"), net_G, net_D, optimizer_G, optimizer_D)
                print("GLOBAL STEP: {0} , G_LOSS: {1} , D_LOSS: {2} , D(X): {3} , D(G(Z)): {4} / {5}".format(
                    global_step, G_error.item(), D_error.item(), D_x, D_G_z1, D_G_z2))

            # Check how the generator is doing by saving G's output on fixed_noise
            if global_step % 500 == 0:
                with torch.no_grad():
                    fake_images = net_G(fixed_noise).detach().cpu()

                tensorboard.add_images("Generator state images", fake_images, global_step)
                save_weight(os.path.join(log_folder, "SavedModels", str(global_step) + ".pth"), net_G)

            global_step += 1
