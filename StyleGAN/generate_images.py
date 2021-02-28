import math

import torch

from model import StyleBasedGenerator
from utils import tensor_to_image, show_image, remove_module_from_keys, binary_style_mixing


generation_size = 256
style_mixing = True
device = "cuda:0"
weights_path = "/home/misha/GANCourse/StyleGAN/train_logs/NUMAKT_StyleGAN_training/SavedModels/Weights_256x256_120000.pth"

pg_step = int(math.log2(generation_size) - 2)
weights = torch.load(weights_path, map_location=device)
# Just in case weights have been saved in a DP regime!
weights_removed = remove_module_from_keys(weights)
generator = StyleBasedGenerator(fused=True).to(device)
try:
    generator.load_state_dict(weights_removed)
except RuntimeError:
    generator.load_state_dict(weights)

generator.eval()
with torch.no_grad():
    while True:
        if style_mixing:
            image = binary_style_mixing(
                generator, pg_phase=pg_step, alpha=1, device=device, image_size=generation_size)
        else:
            noise_vector = torch.rand(1, 512).to(device)
            output = generator(noise_vector, progressive_growing_phase=pg_step, alpha=1)
            image = tensor_to_image(output)

        show_image(image)
