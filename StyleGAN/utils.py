import copy

import cv2
import torch

import numpy as np


def parse_weights(pretrained_generator, my_generator):
    mapping_network_old = list()
    to_rgb_old = list()
    synthesis_network_adain_old = list()
    synthesis_network_noise_old = list()
    synthesis_network_conv1_old = list()
    synthesis_network_conv2_old = list()
    for key in pretrained_generator.keys():
        if key.startswith("style"):
            mapping_network_old.append(key)
        if key.startswith("generator.to_rgb"):
            to_rgb_old.append(key)
        if key.startswith("generator.progression.0.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.1.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.2.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.3.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.4.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.5.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.6.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.7.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.8.adain"):
            synthesis_network_adain_old.append(key)
        if key.startswith("generator.progression.0.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.1.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.2.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.3.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.4.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.5.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.6.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.7.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.8.noise"):
            synthesis_network_noise_old.append(key)
        if key.startswith("generator.progression.0.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.1.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.2.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.3.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.4.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.5.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.6.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.7.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.8.conv1"):
            synthesis_network_conv1_old.append(key)
        if key.startswith("generator.progression.0.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.1.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.2.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.3.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.4.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.5.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.6.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.7.conv2"):
            synthesis_network_conv2_old.append(key)
        if key.startswith("generator.progression.8.conv2"):
            synthesis_network_conv2_old.append(key)

    mapping_network_new = list()
    to_rgb_new = list()
    synthesis_network_adain_new = list()
    synthesis_network_noise_new = list()
    synthesis_network_conv1_new = list()
    synthesis_network_conv2_new = list()
    for key in my_generator.keys():
        if key.startswith("mapping_network"):
            mapping_network_new.append(key)
        if key.startswith("synthesis_network.to_RGB"):
            to_rgb_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.0.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.1.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.2.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.3.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.4.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.5.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.6.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.7.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.8.adaIN"):
            synthesis_network_adain_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.0.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.1.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.2.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.3.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.4.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.5.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.6.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.7.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.8.noise"):
            synthesis_network_noise_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.0.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.1.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.2.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.3.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.4.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.5.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.6.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.7.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.8.convolution_1"):
            synthesis_network_conv1_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.0.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.1.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.2.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.3.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.4.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.5.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.6.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.7.convolution_2"):
            synthesis_network_conv2_new.append(key)
        if key.startswith("synthesis_network.synthesis_network.8.convolution_2"):
            synthesis_network_conv2_new.append(key)

    return (mapping_network_old, to_rgb_old, synthesis_network_adain_old,
            synthesis_network_noise_old, synthesis_network_conv1_old, synthesis_network_conv2_old), \
           (mapping_network_new, to_rgb_new, synthesis_network_adain_new,
            synthesis_network_noise_new, synthesis_network_conv1_new, synthesis_network_conv2_new)


def form_state_dict(pretrained_generator_state_dict, my_generator_state_dict):
    """ From parsed weights

    References:
        https://github.com/rosinality/style-based-gan-pytorch (I took their weights!)
    """
    my_generator_state_dict = copy.deepcopy(my_generator_state_dict)
    old, new = parse_weights(pretrained_generator_state_dict, my_generator_state_dict)

    for first_index, old_el in enumerate(old):
        for second_index, old_el_el in enumerate(old_el):
            my_generator_state_dict[new[first_index][second_index]] = pretrained_generator_state_dict[old_el_el].clone()

    return my_generator_state_dict


def tensor_to_image(tensor):
    """ Performs tensor to image mapping where tensor is an output from style-based generator
    """
    tensor = tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    tensor = tensor * 0.5 + 0.5
    tensor *= 255
    tensor = np.clip(tensor, 0, 255)
    tensor = tensor.astype(np.uint8)
    image = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)

    return image


def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def binary_style_mixing(generator, pg_phase, device, image_size, z_dim=512, alpha=1):
    dummy_code = torch.randn(1, z_dim).to(device)
    dummy_code1 = torch.randn(1, z_dim).to(device)
    output_mix_1 = generator([dummy_code, dummy_code1], progressive_growing_phase=pg_phase, alpha=alpha)
    output_mix_2 = generator([dummy_code1, dummy_code], progressive_growing_phase=pg_phase, alpha=alpha)
    output_1 = generator(dummy_code, progressive_growing_phase=pg_phase, alpha=alpha)
    output_2 = generator(dummy_code1, progressive_growing_phase=pg_phase, alpha=alpha)
    image_mix_1 = tensor_to_image(output_mix_1)
    image_mix_2 = tensor_to_image(output_mix_2)
    image_1 = tensor_to_image(output_1)
    image_2 = tensor_to_image(output_2)
    if image_size > 512:
        image_mix_1 = cv2.resize(image_mix_1, (512, 512))
        image_mix_2 = cv2.resize(image_mix_2, (512, 512))
        image_1 = cv2.resize(image_1, (512, 512))
        image_2 = cv2.resize(image_2, (512, 512))
        image_size = 512

    image = np.zeros((image_size * 2, image_size * 2, 3), dtype=np.uint8)
    image[:image_size, image_size:, :] = image_1
    image[image_size:, :image_size] = image_2
    image[image_size:, image_size:] = image_mix_1
    image[:image_size, :image_size] = image_mix_2

    return image


def remove_module_from_keys(weights):
    """ When you save weights in data parallel training it automatically adds prefix 'module.'
    This function removes this prefix from each key name!
    """
    placeholder = dict()
    for key in weights.keys():
        new_key = key.lstrip("module").lstrip(".")
        placeholder[new_key] = weights[key]

    return placeholder
