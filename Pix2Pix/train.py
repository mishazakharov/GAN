import os
import random

import tqdm
import torch

import numpy as np

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# from model_text_generation import Generator, Discriminator
from models import *
from dataset import *
from utils import *
from cfg import *


dataset_types = {
    "Pix2Pix": Pix2PixDataset,
    "CityScapes": CityScapesDataset,
    "CMPFacade": CMPFacadeDataset,
}

generator_types = {
    "Pix2Pix": Generator,
    "Pix2PixHD": GeneratorHD
}


if __name__ == "__main__":
    # For reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(LOG_FOLDER, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(LOG_FOLDER, "SavedModels"), exist_ok=True)

    tensorboard = SummaryWriter(
        os.path.join(LOG_FOLDER, "tensorboard_logs"))

    dataset_type = dataset_types.get(DATASET_TYPE, None)
    if dataset_type is None:
        raise Exception

    dataset = dataset_type(ROOT_PATH,
                           transform=transforms.Compose([
                               Resize(SIZE),
                               Augmentations(AUGS),
                               ToTensor(),
                               Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    eval_dataset = dataset_type(ROOT_PATH, mode="val",
                                transform=transforms.Compose([Resize(SIZE),
                                                              ToTensor()
                                                              ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=NUM_IMAGES, shuffle=False, num_workers=0)

    generator_type = generator_types.get(MODEL_TYPE, None)
    if generator_type is None:
        raise Exception
    net_G = generator_type().to(DEVICE)
    # Create the Discriminator
    net_D = Discriminator().to(DEVICE)

    optimizer_G = torch.optim.Adam(
        net_G.parameters(), lr=LR_G, betas=(BETA_1, BETA_2))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=LR_D, betas=(BETA_1, BETA_2))

    binary_CE = torch.nn.BCELoss()
    loss_L1 = torch.nn.L1Loss()

    fixed_sample = eval_data_loader.__iter__().__next__()
    fixed_images = fixed_sample["image"]
    fixed_labels = fixed_sample["label_mask"].to(DEVICE)

    disc_loss_value = 0
    generator_loss_value = 0
    global_step = 1
    while True:
        stdout = tqdm.tqdm(data_loader)
        for sample in stdout:
            real_batch = sample["image"]
            real_label = sample["label_mask"]
            real_batch = real_batch.to(DEVICE)
            real_label = real_label.to(DEVICE)
            batch_size = real_batch.shape[0]

            requires_grad(net_G, False)
            requires_grad(net_D, True)

            optimizer_D.zero_grad()

            D_x, real_features = net_D(real_batch, real_label, return_features=FEATURE_MATCHING_LOSS)
            all_ones_label = torch.ones_like(D_x)
            D_x = binary_CE(D_x, all_ones_label)

            fake_batch = net_G(real_label)
            D_G_z1, _ = net_D(fake_batch.detach(), real_label)
            all_zeros_label = torch.zeros_like(D_G_z1)
            D_G_z1 = binary_CE(D_G_z1, all_zeros_label)

            disc_loss_value = (D_x + D_G_z1) * 0.5
            disc_loss_value.backward()
            optimizer_D.step()

            # Logging purposes
            if global_step % 10 == 0:
                disc_loss_value = disc_loss_value.item()

            if global_step % DISC_REPEATS == 0:
                # Update generator with optimal gradients!
                optimizer_G.zero_grad()

                requires_grad(net_G, True)
                requires_grad(net_D, False)

                fake_batch_1 = net_G(real_label)
                D_G_z2, fake_features = net_D(fake_batch_1, real_label, return_features=FEATURE_MATCHING_LOSS)

                G_error_1 = binary_CE(D_G_z2, all_ones_label)
                if FEATURE_MATCHING_LOSS:
                    G_error_2 = 0
                    coeff = 1.0 / len(fake_features)
                    for i in range(len(fake_features)):
                        G_error_2 += loss_L1(fake_features[i], real_features[i].detach()) * coeff
                else:
                    G_error_2 = loss_L1(fake_batch_1, real_batch)
                G_error = G_error_1 + LMBD * G_error_2

                if global_step % 10 == 0:
                    generator_loss_value = G_error.item()

                G_error.backward()

                # Update the weights
                optimizer_G.step()

                requires_grad(net_G, False)
                requires_grad(net_D, True)

            if global_step % LOG_STEP == 0:
                tensorboard.add_scalar("G_LOSS", generator_loss_value, global_step)
                tensorboard.add_scalar("D_LOSS", disc_loss_value, global_step)
                save_checkpoint(
                    os.path.join(LOG_FOLDER, "checkpoints", "checkpoints.tar"), net_G, net_D, optimizer_G, optimizer_D)

            # Check how the generator is doing by saving averaged G's output on fixed_noise
            if global_step % VAL_STEP == 0:
                with torch.no_grad():
                    fake_images = net_G(fixed_labels).detach().cpu()
                    # Shift value range from [-1, 1] to [0, 1]
                    fake_images = fake_images * 0.5 + 0.5
                    # Tensorboard expects RGB images
                    fake_images = torch.flip(fake_images, [1])

                tensorboard.add_images("Generator state images", fake_images, global_step)

            if global_step % SAVE_STEP == 0:
                save_weight(os.path.join(
                    LOG_FOLDER, "SavedModels",
                    "Weights_"+ str(global_step) + ".pth"), net_G)

            log_info = f"GLoss: {generator_loss_value:.3f}; DLoss: {disc_loss_value:.3f};"
            stdout.set_description(log_info)
            global_step += 1
