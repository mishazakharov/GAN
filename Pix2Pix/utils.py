import cv2

import torch

import numpy as np

from typing import Type


def tensor_to_image(tensor, is_label=False):
    """ Performs tensor to image mapping where tensor is an output from style-based generator
    """
    tensor = tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if not is_label:
        tensor = tensor * 0.5 + 0.5
        tensor *= 255
        tensor = np.clip(tensor, 0, 255)
    tensor = tensor.astype(np.uint8)

    return tensor


def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def save_checkpoint(path: str, net_G: torch.nn.Module, net_D: torch.nn.Module,
                    optimizer_G: Type[torch.optim.Optimizer], optimizer_D: Type[torch.optim.Optimizer]):
    checkpoint = {
        "net_G": net_G.state_dict(),
        "net_D": net_D.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict()
    }
    torch.save(checkpoint, path)


def save_weight(path: str, net_G: torch.nn.Module):
    torch.save(net_G.state_dict(), path)


def requires_grad(model: torch.nn.Module, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag


def load_weights(path, model):
    weights = torch.load(path, map_locatioin="cpu")
    model.load_state_dict(weights)
