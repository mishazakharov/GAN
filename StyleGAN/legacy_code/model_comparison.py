import torch

import time

import numpy as np

from model_not_mine import StyledGenerator
from model_not_mine import Discriminator as DiscriminatorOld
from model import StyleBasedGenerator, Discriminator
from utils import form_state_dict, tensor_to_image, show_image


def count_params(module):

    return sum([p.numel() for p in module.parameters() if p.requires_grad])


def compare_outputs_gen(module_1, module_2, visualize=False):
    dummy_input = torch.randn(1, 512).cuda()
    module_1.eval()
    module_2.eval()
    output_1 = module_1(dummy_input, step=8)
    output_2 = module_2(dummy_input)
    image_1 = tensor_to_image(output_1)
    image_2 = tensor_to_image(output_2)
    if visualize:
        mixed_image = np.zeros((1024, 2048, 3), dtype=np.uint8)
        mixed_image[:, :1024, :] = image_1
        mixed_image[:, 1024:, :] = image_2
        show_image(mixed_image)
    else:
        diff = np.abs(image_1 - image_2).mean()

        return diff


def compare_outputs_disc(module_1, module_2):
    dummy_input = torch.rand(1, 3, 1024, 1024).cuda()
    output_1 = module_1(dummy_input, step=8)
    output_2 = module_2(dummy_input)
    diff = torch.abs(output_1 - output_2).mean()

    return diff


if __name__ == "__main__":
    device = "cuda:0"

    old_gen = StyledGenerator().to(device)
    new_gen = StyleBasedGenerator().to(device)
    old_disc = DiscriminatorOld(fused=True).to(device)
    new_disc = Discriminator(fused=True).to(device)

    ckpt = torch.load("stylegan-1024px-new.model", map_location=device)
    old_gen.load_state_dict(ckpt["g_running"])
    new_gen.load_state_dict(form_state_dict(ckpt["g_running"], new_gen.state_dict()))


    old_gen_p = count_params(old_gen)
    new_gen_p = count_params(new_gen)

    old_disc_p = count_params(old_disc)
    new_disc_p = count_params(new_disc)
    dummy_input = torch.rand(1, 3, 1024, 1024).to(device)
    t0 = time.time()
    for _ in range(1000):
        with torch.no_grad():
            # diff_gen = compare_outputs_gen(old_gen, new_gen)
            # diff_disc = compare_outputs_disc(old_disc, new_disc)
            output = old_disc(dummy_input, step=8)
    t1 = time.time()
    t2 = time.time()
    for _ in range(1000):
        with torch.no_grad():
            output = new_disc(dummy_input)
    t3 = time.time()

    old_disc_time = t1 - t0
    new_disc_time = t3 - t2


    a = 1
