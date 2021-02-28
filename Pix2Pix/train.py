import os
import random

import tqdm
import torch

import numpy as np

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model_text_generation import Generator, Discriminator
from dataset import TextGenerationDataset, CityScapesDataset
from utils import save_weight, save_checkpoint, requires_grad
from cfg import *


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

    # dataset = PairedDataset(root_path, "train",
    #                         transform=transforms.Compose([
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    # eval_dataset = PairedDataset(root_path, "val")
    dataset = TextGenerationDataset(root_path,
                                    transform=transforms.Compose([
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    eval_dataset = TextGenerationDataset("/home/misha/datasets/passports_word_annotations/test_campaign/test_template/eval.csv",
                                    transform=transforms.Compose([
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=num_images, shuffle=True, num_workers=0)

    # Create the generator
    net_G = Generator().cuda()

    # Create the Discriminator
    net_D = Discriminator().cuda()

    optimizer_G = torch.optim.Adam(
        net_G.parameters(), lr=lr_G, betas=(beta_1, beta_2))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D, betas=(beta_1, beta_2))

    binary_CE = torch.nn.BCELoss()
    loss_L1 = torch.nn.L1Loss()


    fixed_sample = eval_data_loader.__iter__().__next__()
    fixed_images = fixed_sample["image"]
    fixed_labels = fixed_sample["label_mask"].cuda()

    disc_loss_value = 0
    generator_loss_value = 0
    global_step = 1
    while True:
        stdout = tqdm.tqdm(data_loader)
        for sample in stdout:
            real_batch = sample["image"]
            real_label = sample["label_mask"]
            real_batch = real_batch.cuda()
            real_label = real_label.cuda().squeeze()
            batch_size = real_batch.shape[0]

            requires_grad(net_G, False)
            requires_grad(net_D, True)

            optimizer_D.zero_grad()

            D_x = net_D(real_batch, real_label)
            all_ones_label = torch.ones_like(D_x)
            D_x = binary_CE(D_x, all_ones_label)

            fake_batch = net_G(real_label)
            D_G_z1 = net_D(fake_batch.detach(), real_label)
            all_zeros_label = torch.zeros_like(D_G_z1)
            D_G_z1 = binary_CE(D_G_z1, all_zeros_label)

            disc_loss_value = (D_x + D_G_z1) * 0.5
            disc_loss_value.backward()
            optimizer_D.step()

            # Logging purposes
            if global_step % 10 == 0:
                disc_loss_value = disc_loss_value.item()

            if global_step % disc_repeats == 0:
                # Update generator with optimal gradients!
                optimizer_G.zero_grad()

                requires_grad(net_G, True)
                requires_grad(net_D, False)

                fake_batch_1 = net_G(real_label)
                D_G_z2 = net_D(fake_batch_1, real_label)

                G_error_1 = binary_CE(D_G_z2, all_ones_label)
                G_error_2 = loss_L1(fake_batch_1, real_batch)
                G_error = G_error_1 + lmbd * G_error_2

                if global_step % 10 == 0:
                    generator_loss_value = G_error.item()

                G_error.backward()

                # Update the weights
                optimizer_G.step()

                requires_grad(net_G, False)
                requires_grad(net_D, True)

            if global_step % 100 == 0:
                tensorboard.add_scalar("G_LOSS", generator_loss_value, global_step)
                tensorboard.add_scalar("D_LOSS", disc_loss_value, global_step)
                save_checkpoint(
                    os.path.join(log_folder, "checkpoints", "checkpoints.tar"), net_G, net_D, optimizer_G, optimizer_D)

            # Check how the generator is doing by saving averaged G's output on fixed_noise
            if global_step % 500 == 0:
                with torch.no_grad():
                    fake_images = net_G(fixed_labels).detach().cpu()

                tensorboard.add_images("Generator state images", fake_images, global_step)
                if global_step % 10_000 == 0:
                    save_weight(os.path.join(
                        log_folder, "SavedModels",
                        "Weights_"+ str(global_step) + ".pth"), net_G)

            log_info = f"GLoss: {generator_loss_value:.3f}; DLoss: {disc_loss_value:.3f};"
            stdout.set_description(log_info)
            global_step += 1
