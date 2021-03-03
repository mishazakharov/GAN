import os
import random

import cv2
import tqdm
import torch

import pandas as pd
import numpy as np

import __init__

from typing import Type
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Discriminator
from cfg import *
from BasicGANs.loss_functions import get_hinge_loss
from BasicGANs.utils import write_logs


class UnconditionalDataset(torch.utils.data.Dataset):

    base_dir = "images/"

    def __init__(self, paths, image_size=4, transform=None):
        self.data = self.concatenate_csv(paths)
        self.image_size = image_size
        self.transform = transform
        self.data_dir = "/".join(paths[0].split("/")[:-3])

    def __getitem__(self, idx):
        label = len(str(self.data[idx, 1])) - 1
        path = os.path.join(self.data_dir, self.data[idx, 2],
                            self.data[idx, 3], UnconditionalDataset.base_dir)
        image = cv2.imread(os.path.join(path, self.data[idx, 0]))

        if label > 3 or image is None:
            return self.__getitem__(idx + 1)

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = torch.from_numpy(image).float().permute(2, 0, 1).contiguous() / 255

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'label': torch.LongTensor([label]),
                  }

        return sample

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def concatenate_csv(global_string) -> np.ndarray:
        frames = list()
        for csv in global_string:
            path = csv.split('/')
            company = path[-3]
            template = path[-2]
            try:
                dataframe = pd.read_csv(csv).values
            except pd.errors.EmptyDataError:
                print('I met an empty csv file!')
                continue
            company_array = np.asarray([company for _ in range(dataframe.shape[0])])
            template_array = np.asarray([template for _ in range(dataframe.shape[0])])
            new_dataframe = np.c_[dataframe, company_array]
            new_dataframe = np.c_[new_dataframe, template_array]
            frames.append(new_dataframe)

        return np.concatenate(frames, axis=0)


def requires_grad(model: torch.nn.Module, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag


def moving_average(model_1: torch.nn.Module, model_2: torch.nn.Module, weight: float = 0.9999):
    parameters_1 = dict(model_1.named_parameters())
    parameters_2 = dict(model_2.named_parameters())

    for key in parameters_1.keys():
        parameters_1[key].data.mul_(weight).add_(1 - weight, parameters_2[key].data)


def save_checkpoint(path: str, net_G: torch.nn.Module, net_D: torch.nn.Module,
                    optimizer_G: Type[torch.optim.Optimizer], optimizer_D: Type[torch.optim.Optimizer],
                    averaged_G: torch.nn.Module):
    checkpoint = {
        "net_G": net_G.state_dict(),
        "net_D": net_D.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "averaged_G": averaged_G.state_dict()
    }
    torch.save(checkpoint, path)


def save_weight(path: str, net_G: torch.nn.Module):
    torch.save(net_G.state_dict(), path)


def orthogonal_init(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) \
                or isinstance(module, torch.nn.Embedding):
            torch.nn.init.orthogonal_(module.weight)


def orthogonal_regularization(module, beta=1e-4):
    """ Directly computes gradients

    References:
        https://github.com/ajbrock/BigGAN-PyTorch
    """
    with torch.no_grad():
        for parameter in module.parameters():
            if len(parameter.shape) > 2:
                weight = parameter.view(parameter.shape[0], -1)
                gradient = (2 * torch.mm(torch.mm(weight, weight.t()) * (1. - torch.eye(weight.shape[0],
                                                                                        device=weight.device)), weight))
                parameter.grad.data += beta * gradient.view(parameter.shape)


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

    dataset = UnconditionalDataset(data_path, image_size=image_size,
                                   transform=transforms.Compose([
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create the generator
    net_G = torch.nn.DataParallel(Generator(n_classes=n_classes, ch=64)).cuda()
    orthogonal_init(net_G.module)

    # Create the Discriminator
    net_D = torch.nn.DataParallel(Discriminator(n_classes=n_classes, ch=64)).cuda()
    orthogonal_init(net_D.module)

    averaged_G = Generator(n_classes=n_classes, ch=64).cuda()
    averaged_G.train(mode=False)

    # Equalize parameters for running generator
    moving_average(averaged_G, net_G.module, weight=0)

    optimizer_G = torch.optim.Adam(
        net_G.parameters(), lr=lr_G, betas=(beta_1, beta_2))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D, betas=(beta_1, beta_2))

    fixed_noise = torch.randn(num_images, z_dim, device=device).to(device)
    fixed_label = torch.randint(0, n_classes-1, size=(num_images,)).to(device)

    disc_loss_value = 0
    generator_loss_value = 0
    global_step = 1
    while True:
        stdout = tqdm.tqdm(data_loader)
        for sample in stdout:
            real_batch = sample["image"]
            real_label = sample["label"]
            real_batch = real_batch.cuda()
            real_label = real_label.cuda().squeeze()
            batch_size = real_batch.shape[0]

            noise_vector_1 = torch.rand(batch_size, z_dim, device=device)
            fake_label_1 = torch.randint(0, n_classes-1, size=(batch_size,))
            noise_vector_2 = torch.rand(batch_size, z_dim, device=device)
            fake_label_2 = torch.randint(0, n_classes-1, size=(batch_size,))

            requires_grad(net_G, False)
            requires_grad(net_D, True)

            optimizer_D.zero_grad()

            D_x, D_G_z1 = get_hinge_loss(net_G, net_D, real_batch, real_label, noise_vector_1, fake_label_1)

            # Logging purposes
            if global_step % 10 == 0:
                disc_loss_value = (D_x + D_G_z1).item()

            if orth_reg:
                orthogonal_regularization(net_D.module)

            # Update optimizer
            optimizer_D.step()

            if global_step % disc_repeats == 0:
                # Update generator with optimal gradients!
                optimizer_G.zero_grad()

                requires_grad(net_G, True)
                requires_grad(net_D, False)

                G_error = get_hinge_loss(net_G, net_D, noise_vector=noise_vector_2, fake_label=fake_label_2,
                                         discriminator=False)
                if global_step % 10 == 0:
                    generator_loss_value = G_error.item()

                if orth_reg:
                    orthogonal_regularization(net_G.module)

                # Update the weights
                optimizer_G.step()

                # Update averaged generator
                moving_average(averaged_G, net_G.module)

                requires_grad(net_G, False)
                requires_grad(net_D, True)

            if global_step % 100 == 0:
                write_logs({"G_Loss": generator_loss_value, "D_Loss": disc_loss_value, "D(X)": D_x.mean().item(),
                            "D(G(Z))": D_G_z1.mean().item()}, tensorboard, global_step)
                save_checkpoint(
                    os.path.join(log_folder, "checkpoints", "checkpoints.tar"),
                    net_G, net_D, optimizer_G, optimizer_D, averaged_G)

            # Check how the generator is doing by saving averaged G's output on fixed_noise
            if global_step % 500 == 0:
                with torch.no_grad():
                    fake_images = averaged_G(
                        fixed_noise, fixed_label).detach().cpu()

                tensorboard.add_images("Generator state images", fake_images, global_step)
                if global_step % 10_000 == 0:
                    save_weight(os.path.join(
                        log_folder, "SavedModels",
                        "Weights_" + str(image_size) + "x" + str(image_size) + "_" + str(global_step) + ".pth"),
                        averaged_G)

            log_info = f"GLoss: {generator_loss_value:.3f}; DLoss: {disc_loss_value:.3f};"
            stdout.set_description(log_info)
            global_step += 1
