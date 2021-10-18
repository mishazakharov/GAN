import os
import random

import glob
import torch

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from models import *
from loss_functions import *  # get_minimax_loss


if __name__ == "__main__":
    data_path = glob.glob("*.jpg")
    log_folder = "second_experiment"
    is_wgan = True
    # Batch size during training
    batch_size = 128
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = (64, 64)
    # Number of channels in the training images. For color images this is 3
    image_dim = 3
    # Size of z latent vector (i.e. size of generator input)
    z_dim = 100
    # Size of feature maps in GAN
    hidden_dim = 64
    # Learning rate for optimizers
    lr = 0.0001  # 0.0002
    beta_1 = 0.0  # 0.5
    beta_2 = 0.9  # 0.999
    c_lambda = 10
    disc_repeats = 5
    num_images = 64
    num_workers = 0
    device = "cpu"
    seed = 5000
    log_step = 100
    val_step = 500
    save_step = 1_000

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

    net_G = Generator(z_dim, hidden_dim, image_dim, resolution=image_size[0]).to(device)
    net_G.apply(weights_init)

    net_D = Discriminator(image_dim, hidden_dim, resolution=image_size[0], is_wgan=is_wgan).to(device)
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
            real_batch = batch["image"].to(device)
            b_size = real_batch.shape[0]

            # Update discriminator's weights
            D_error_mean = 0.0
            for _ in range(disc_repeats):
                optimizer_D.zero_grad()
                fake_noise = get_noise(b_size, z_dim, device=device).unsqueeze(2).unsqueeze(2)
                if not is_wgan:
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    D_error, D_G_z1, D_x, fake_batch = get_minimax_loss(net_G, net_D, criterion, label, real_batch,
                                                                        fake_noise, fake_label)
                else:
                    D_x, D_G_z1, gradient_penalty = get_wasserstein_loss(net_G, net_D, real_batch, fake_noise)
                    D_error = -D_x + D_G_z1

                D_error_mean += D_error.item() / disc_repeats
                optimizer_D.step()

            # Update generator's weights
            optimizer_G.zero_grad()
            if not is_wgan:
                label.fill_(real_label)
                G_error, D_G_z2 = get_minimax_loss(net_G, net_D, criterion, label, fake_batch, discriminator=False)
            else:
                fake_noise_1 = get_noise(b_size, z_dim, device=device).unsqueeze(2).unsqueeze(2)
                G_error, D_G_z2 = get_wasserstein_loss(net_G, net_D, noise_vector=fake_noise_1, discriminator=False,
                                                       c_lambda=c_lambda)

            optimizer_G.step()

            # Output training stats
            if global_step % log_step == 0:
                save_checkpoint(
                    os.path.join(log_folder, "checkpoints", "checkpoints.tar"), net_G, net_D, optimizer_G, optimizer_D)
                write_logs({"Global Step": global_step, "G_Loss": G_error.item(), "D_Loss": D_error_mean,
                            "D(X)": D_x, "D(G(Z))": [D_G_z1, D_G_z2]}, tensorboard, global_step)

            # Check how the generator is doing by saving G's output on fixed_noise
            if global_step % val_step == 0:
                generate_with_fixed_noise(net_G, fixed_noise, tensorboard, global_step)

            if global_step % save_step == 0:
                save_weight(os.path.join(log_folder, "SavedModels", str(global_step) + ".pth"), net_G)

            global_step += 1
