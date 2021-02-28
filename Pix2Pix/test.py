import cv2
import torch

import numpy as np

from torchvision import transforms

from dataset import TextGenerationDataset, CityScapesDataset
from pix2pixhd import Generator
from utils import tensor_to_image, show_image
from cfg import *


# dataset = PairedDataset(root_path, mode="val", segmentation_mask=True)
dataset = TextGenerationDataset("/home/misha/datasets/passports_word_annotations/test_campaign/test_template/eval.csv",
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

model_path = "/home/misha/GANCourse/Pix2Pix/train_logs/text_generation_pix2pixHD/Weights_150000.pth"
weights = torch.load(model_path, map_location="cpu")
net_G = Generator()
net_G.load_state_dict(weights)
net_G.eval()

with torch.no_grad():
    for sample in dataset:
        image, label_mask = sample["image"], sample["label_mask"]
        output = net_G(label_mask.unsqueeze(0))
        generated_image = tensor_to_image(output)
        image = tensor_to_image(image)
        label_mask = tensor_to_image(label_mask)
        # generated_image = cv2.resize(generated_image, (512, 512))
        # segmentation_mask = cv2.resize(segmentation_mask, (512, 512))
        result = np.concatenate((label_mask, generated_image), axis=1)

        show_image(result)
