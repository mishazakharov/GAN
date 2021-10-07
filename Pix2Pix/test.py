import cv2
import torch

import numpy as np

from torchvision import transforms

from train import *


if __name__ == "__main__":
    dataset_type = dataset_types.get(DATASET_TYPE, None)
    if dataset_type is None:
        raise Exception

    eval_dataset = dataset_type(ROOT_PATH, mode="val",
                                transform=transforms.Compose([Resize(SIZE)
                                                              ]))

net_G = Generator()
load_weights(TEST_WEIGHTS, net_G)
net_G.to(DEVICE)

with torch.no_grad():
    for sample in dataset:
        image, label_mask = sample["image"], sample["label_mask"]
        output = net_G(label_mask.unsqueeze(0))
        generated_image = tensor_to_image(output)
        image = tensor_to_image(image)
        label_mask = tensor_to_image(label_mask, is_label=True)
        # generated_image = cv2.resize(generated_image, (512, 512))
        # segmentation_mask = cv2.resize(segmentation_mask, (512, 512))
        generated_result = np.concatenate((label_mask, generated_image), axis=1)
        real_result = np.concatenate((label_mask, image), axis=1)

        result = np.concatenate((generated_result, real_result), axis=0)

        show_image(result)
