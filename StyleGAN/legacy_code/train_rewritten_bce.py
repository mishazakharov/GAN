import os
import math
import random

import cv2
import torch
import torchvision

import numpy as np

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model import StyleBasedGenerator, Discriminator
from new import save_weight, save_checkpoint
from wgan_gp import get_gradient, get_gen_loss, get_crit_loss, gradient_penalty
from cfg import *


class UnconditionalDataset(torch.utils.data.Dataset):
    def __init__(self, paths, image_size=4, transform=None):
        self.data = paths
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, idx):
        path = self.data[idx]
        image = cv2.imread(path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).float().permute(2, 0, 1).contiguous() / 255

        if self.transform:
            image = self.transform(image)

        return {"image": image}

    def __len__(self):
        return self.data.__len__()


class ResizeToTensor(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        image = cv2.resize(image, self.image_size)
        image = torch.from_numpy(image).float().permute(2, 0, 1).contiguous() / 255

        return image


def change_dataloader(dataset, image_size, batch_size, num_workers):
    dataset.image_size = image_size
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


def change_dataloader_MNIST(dataset, image_size, batch_size, num_workers):
    dataset.transform.transforms[0].size = image_size
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


def change_lr(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# For reproducibility
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

os.makedirs(log_folder, exist_ok=True)
os.makedirs(os.path.join(log_folder, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(log_folder, "SavedModels"), exist_ok=True)

tensorboard = SummaryWriter(
    os.path.join(log_folder, "tensorboard_logs"))

# dataset = UnconditionalDataset(data_path, initial_image_size,
#                                transform=transforms.Compose(
#                                    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataset = torchvision.datasets.MNIST(root="/home/m_zakharov/.", train=True, download=True,
                                     transform=transforms.Compose(
                                         [transforms.Resize(size=initial_image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))]))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_sizes[initial_image_size], shuffle=True, num_workers=num_workers)

# Create the generator
net_G = StyleBasedGenerator(
    z_dim=z_dim, w_dim=w_dim, image_dim=image_dim, n_layers=n_layers, normalize=normalize, fused=fused).to(device)

# Create the Discriminator
net_D = Discriminator().to(device)

criterion = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr[initial_image_size], betas=(beta_1, beta_2))
optimizer_D = torch.optim.Adam(net_G.parameters(), lr=lr[initial_image_size], betas=(beta_1, beta_2))

fixed_noise = torch.randn(64, z_dim, device=device)

real_label = 1.
fake_label = 0.

# Progressive growing hyper-parameters!
pg_step = int(math.log2(initial_image_size) - 2)
max_pg_step = int(math.log2(maximum_image_size) - 2)
image_size = initial_image_size
passed_samples = 0
final_layer = False

D_error_mean = 0
global_step = 1
while True:
    requires_grad(net_G, False)
    requires_grad(net_D, True)
    for i, batch in enumerate(data_loader):
        real_batch = batch[0].to(device)
        batch_size = real_batch.shape[0]
        real_batch = real_batch.expand(batch_size, 3, real_batch.shape[2], real_batch.shape[3])
        passed_samples += batch_size

        alpha = min(1, 1 / phase_samples * (passed_samples + 1)) if not final_layer else 1
        # Initial size means training only first block
        if image_size == initial_image_size:
            alpha = 1
        # Point where we should go to the next progressive growing phase
        if passed_samples > phase_samples * 2 and not final_layer:
            passed_samples = 0
            pg_step += 1
            if pg_step > max_pg_step:
                pg_step = max_pg_step
                final_layer = True
            # Save checkpoint for a particular resolution's final state except last obviously!
            ckpt_saving_path = os.path.join(
                log_folder, "checkpoints", "final_ckpt_" + str(image_size) + "x" + str(image_size) + "_resolution.tar")
            save_checkpoint(ckpt_saving_path, net_G, net_D, optimizer_G, optimizer_D)
            # Adjust data preparation step
            image_size = 4 * 2 ** pg_step
            batch_size = batch_sizes[image_size]
            data_loader = change_dataloader_MNIST(dataset, image_size, batch_size, num_workers)
            # Adjust learning rate in optimizers
            learning_rate = lr[image_size]
            change_lr(optimizer_G, learning_rate)
            change_lr(optimizer_D, learning_rate)
            # Reinitialize for cycle!
            break

        # Style mixing regularization technique!
        if style_mixing and random.random() < style_mixing_rate and pg_step > 0:
            a, b, c, d = torch.randn(4, batch_size, z_dim, device=device).chunk(4, dim=0)
            a, b, c, d = list(map(lambda x: x.squeeze(0), [a, b, c, d]))
            noise_vectors_1 = [a, b]
            noise_vectors_2 = [c, d]
        else:
            noise_vectors_1, noise_vectors_2 = torch.randn(2, batch_size, z_dim, device=device).chunk(2, dim=0)
            noise_vectors_1, noise_vectors_2 = list(map(lambda x: x.squeeze(0), [noise_vectors_1, noise_vectors_2]))

        # Update discriminator to optimality!
        optimizer_D.zero_grad()
        fake_batch = net_G(noise_vectors_1, progressive_growing_phase=pg_step, alpha=alpha)
        D_G_z1 = net_D(fake_batch.detach(), progressive_growing_phase=pg_step, alpha=alpha).view(-1)
        D_x = net_D(real_batch, progressive_growing_phase=pg_step, alpha=alpha).view(-1)
        label = torch.torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        D_error_real = criterion(D_x, label)
        D_error_real.backward()

        label.fill_(fake_label)
        D_error_fake = criterion(D_G_z1, label)
        D_error_fake.backward()
        D_error_mean = (D_error_real + D_error_fake) / 2

        # Update optimizer
        optimizer_D.step()

        if global_step % disc_repeats == 0:
            # Update generator with optimal gradients!
            optimizer_G.zero_grad()

            requires_grad(net_G, True)
            requires_grad(net_D, False)

            label.fill_(real_label)

            fake_batch_1 = net_G(noise_vectors_2, progressive_growing_phase=pg_step, alpha=alpha)
            D_G_z2 = net_D(fake_batch_1, progressive_growing_phase=pg_step, alpha=alpha).view(-1)

            G_error = criterion(D_G_z2, label)
            # G_error = get_gen_loss(D_G_z2)
            G_error.backward()

            # Update the weights
            optimizer_G.step()

            requires_grad(net_G, False)
            requires_grad(net_D, True)

        if global_step % 100 == 0:
            D_x = D_x.mean().item()
            D_G_z1 = D_G_z1.mean().item()
            D_G_z2 = D_G_z2.mean().item()
            tensorboard.add_scalar("G_LOSS", G_error.item(), global_step)
            tensorboard.add_scalar("D_LOSS", D_error_mean, global_step)
            tensorboard.add_scalar("D(X)", D_x, global_step)
            tensorboard.add_scalar("D(G(Z))", D_G_z1, global_step)
            save_checkpoint(
                os.path.join(log_folder, "checkpoints", "checkpoints.tar"), net_G, net_D, optimizer_G, optimizer_D)
            print("GLOBAL STEP: {0} , G_LOSS: {1} , D_LOSS: {2} , WD: {6} , D(X): {3} , D(G(Z)): {4} / {5}".format(
                global_step, G_error.item(), D_error_mean, D_x, D_G_z1, D_G_z2, 1))

        # Check how the generator is doing by saving G's output on fixed_noise
        if global_step % 500 == 0:
            with torch.no_grad():
                fake_images = net_G(fixed_noise, progressive_growing_phase=pg_step, alpha=alpha).detach().cpu()
                # # De-normalization
                # fake_images = fake_images * 0.5 + 0.5

            tensorboard.add_images("Generator state images", fake_images, global_step)
            save_weight(os.path.join(log_folder, "SavedModels", "Weights_" + str(image_size) + "x" + str(image_size) +
                                     "_" + str(global_step) + ".pth"), net_G)

        # TODO: drop D_error_mean each time we encounter generator update!
        D_error_mean = 0
        global_step += 1
