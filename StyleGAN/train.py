import os
import math
import random

import cv2
import tqdm
import torch
import torchvision

import numpy as np

import __init__

from typing import Type
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from models.StyleGAN import StyleBasedGenerator, Discriminator
from cfg import *
from BasicGANs.loss_functions import get_wasserstein_loss
from BasicGANs.utils import write_logs, generate_with_fixed_noise


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

        return image

    def __len__(self):
        return self.data.__len__()


def change_dataloader(dataset: Type[torch.utils.data.Dataset], image_size: int,
                      batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    dataset.image_size = image_size
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


def change_dataloader_MNIST(dataset, image_size, batch_size, num_workers):
    dataset.transform.transforms[0].size = image_size
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


def change_lr(optimizer: Type[torch.optim.Optimizer], learning_rate: float):
    for param_group in optimizer.param_groups:
        multiplier = param_group.get("multipllier", 1)
        param_group["lr"] = learning_rate * multiplier


def requires_grad(model: torch.nn.Module, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag


def moving_average(model_1: torch.nn.Module, model_2: torch.nn.Module, weight: float = 0.999):
    parameters_1 = dict(model_1.named_parameters())
    parameters_2 = dict(model_2.named_parameters())

    for key in parameters_1.keys():
        parameters_1[key].data.mul_(weight).add_(1 - weight, parameters_2[key].data)


def save_checkpoint(path: str, net_G: torch.nn.Module, net_D: torch.nn.Module,
                    optimizer_G: Type[torch.optim.Optimizer], optimizer_D: Type[torch.optim.Optimizer],
                    averaged_G: torch.nn.Module = None):
    checkpoint = {
        "net_G": net_G.state_dict(),
        "net_D": net_D.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
    }
    if averaged_G:
        checkpoint["averaged_G"] = averaged_G.state_dict()

    torch.save(checkpoint, path)


def save_weight(path: str, net_G: torch.nn.Module):
    torch.save(net_G.state_dict(), path)


# import models
#
# model_types = {
#     "StyleGAN": {
#         "Generator": models.StyleGAN.StyleBasedGenerator,
#         "Discriminator": models.StyleGAN.Discriminator
#     },
#     "DCGAN": {
#         "Generator": models.DCGAN.Generator,
#         "Discriminator": models.DCGAN.Discriminator
#     },
#     "WGAN": {
#         "Generator": models.DCGAN.Generator,
#         "Discriminator": models.DCGAN.Discriminator
#     },
#     "BigGAN": {}
# }


if __name__ == "__main__":
    # For reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_folder, "SavedModels"), exist_ok=True)

    tensorboard = SummaryWriter(
        os.path.join(log_folder, "tensorboard_logs"))

    # dataset = torchvision.datasets.MNIST(root="/home/misha/datasets/.", train=True, download=True,
    #                                      transform=transforms.Compose(
    #                                          [transforms.Resize(size=initial_image_size),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize((0.5,), (0.5,))]))
    dataset = UnconditionalDataset(data_path, image_size=initial_image_size,
                                   transform=transforms.Compose([
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_sizes[initial_image_size], shuffle=True, num_workers=num_workers)

    # Create the generator
    net_G = torch.nn.DataParallel(StyleBasedGenerator(
        z_dim=z_dim, w_dim=w_dim, image_dim=image_dim, n_layers=n_layers, normalize=normalize, fused=fused)).cuda()

    # Create the Discriminator
    net_D = torch.nn.DataParallel(Discriminator(fused=True, use_activations=True)).cuda()

    averaged_G = None
    if weights_averaging:
        averaged_G = StyleBasedGenerator(
            z_dim=z_dim, w_dim=w_dim, image_dim=image_dim, n_layers=n_layers, normalize=normalize, fused=fused).cuda()
        averaged_G.train(mode=False)

        # Equalize parameters for running generator
        moving_average(averaged_G, net_G.module, weight=0)

    optimizer_G = torch.optim.Adam(
        net_G.module.synthesis_network.parameters(), lr=lr[initial_image_size], betas=(beta_1, beta_2))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr[initial_image_size], betas=(beta_1, beta_2))

    # Mapping network learning rate is 10x slower
    optimizer_G.add_param_group({
        "params": net_G.module.mapping_network.parameters(),
        "lr": lr[initial_image_size] * 0.01,
        "multiplier": 0.01
    })

    if checkpoints_path:
        checkpoint = torch.load(checkpoints_path)
        net_G.load_state_dict(checkpoint["net_G"])
        net_D.load_state_dict(checkpoint["net_D"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        if weights_averaging:
            if checkpoint.get("averaged_G"):
                averaged_G.load_state_dict(checkpoint["averaged_G"])
            else:
                moving_average(averaged_G, net_G.module, weight=0)

    fixed_noise = torch.randn(num_images, z_dim, device=device)

    # Progressive growing hyper-parameters!
    pg_step = int(math.log2(initial_image_size) - 2)
    max_pg_step = int(math.log2(maximum_image_size) - 2)
    image_size = initial_image_size
    passed_samples = 0
    final_layer = False

    gp_loss_value = 0
    disc_loss_value = 0
    generator_loss_value = 0
    global_step = 1
    while True:
        stdout = tqdm.tqdm(data_loader)
        for real_batch in stdout:
            real_batch = real_batch.cuda()
            batch_size = real_batch.shape[0]
            # real_batch = real_batch.expand(batch_size, 3, real_batch.shape[2], real_batch.shape[3])

            passed_samples += batch_size
            alpha = min(1, 1 / phase_samples * (passed_samples + 1)) if not final_layer else 1
            # Initial size means training only first block
            if image_size == initial_image_size:
                alpha = 1
            # Point where we should go to the next progressive growing phase
            if passed_samples > phase_samples * 2 and not final_layer and progressive_growing:
                passed_samples = 0
                pg_step += 1
                if pg_step > max_pg_step:
                    pg_step = max_pg_step
                    final_layer = True
                # Save checkpoint for a particular resolution's final state except last obviously!
                ckpt_saving_path = os.path.join(
                    log_folder, "checkpoints",
                    "final_ckpt_" + str(image_size) + "x" + str(image_size) + "_resolution.tar")
                save_checkpoint(ckpt_saving_path, net_G, net_D, optimizer_G, optimizer_D, averaged_G)
                # Adjust data preparation step
                image_size = 4 * 2 ** pg_step
                batch_size = batch_sizes[image_size]
                data_loader = change_dataloader(dataset, image_size, batch_size, num_workers)
                # Adjust learning rate in optimizers
                learning_rate = lr[image_size]
                change_lr(optimizer_G, learning_rate)
                change_lr(optimizer_D, learning_rate)
                # Reinitialize 'for' cycle!
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

            requires_grad(net_G, False)
            requires_grad(net_D, True)
            optimizer_D.zero_grad()

            D_x, D_G_z1, gradient_penalty = get_wasserstein_loss(net_G, net_D, real_batch, noise_vectors_1,
                                                                 progressive_growing_phase=pg_step, alpha=alpha)

            # Logging purposes
            if global_step % 10 == 0:
                disc_loss_value = (-D_x + D_G_z1).item()
                gp_loss_value = gradient_penalty.item()

            # Update optimizer
            optimizer_D.step()

            if global_step % disc_repeats == 0:
                # Update generator with optimal gradients!
                optimizer_G.zero_grad()

                requires_grad(net_G, True)
                requires_grad(net_D, False)

                G_error, D_G_z2 = get_wasserstein_loss(net_G, net_D, noise_vector=noise_vectors_2, discriminator=False,
                                                       progressive_growing_phase=pg_step, alpha=alpha)

                if global_step % 10 == 0:
                    generator_loss_value = G_error.item()

                # G_error.backward()

                # Update the weights
                optimizer_G.step()

                if weights_averaging:
                    # Update averaged generator
                    moving_average(averaged_G, net_G.module)

                requires_grad(net_G, False)
                requires_grad(net_D, True)

            if global_step % log_step == 0:
                write_logs({"G_Loss": generator_loss_value, "D_Loss": disc_loss_value, "GP_Loss": gp_loss_value,
                            "D(X)": D_x.mean().item(), "D(G(Z))": D_G_z1}, tensorboard, global_step)
                save_checkpoint(
                    os.path.join(log_folder, "checkpoints", "checkpoints.tar"),
                    net_G, net_D, optimizer_G, optimizer_D, averaged_G)

            eval_save_model = averaged_G if weights_averaging else net_G
            # Check how the generator is doing by saving averaged G's output on fixed_noise
            if global_step % val_step == 0:
                generate_with_fixed_noise(eval_save_model, fixed_noise, tensorboard, global_step,
                                          progressive_growing_phase=pg_step, alpha=alpha)

            if global_step % save_step == 0:
                save_weight(os.path.join(
                    log_folder, "SavedModels",
                    "Weights_" + str(image_size) + "x" + str(image_size) + "_" + str(global_step) + ".pth"),
                    eval_save_model)

            log_info = (
                f"Resolution: {image_size}; GLoss: {generator_loss_value:.3f}; DLoss: {disc_loss_value:.3f};"
                f" GPLoss: {gp_loss_value:.3f}; Alpha: {alpha:.6f}"
            )
            stdout.set_description(log_info)
            global_step += 1
