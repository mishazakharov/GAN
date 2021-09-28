import os
import csv
import random

import cv2
import glob
import torch

import numpy as np


def show(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)


class TextGenerationDataset(torch.utils.data.Dataset):

    """ Class for general domain specific text generation via Pix2Pix framework!
    """
    image_folder = "images"
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_COMPLEX
    size = (64, 512)

    def __init__(self, csv_path, transform=None):
        super().__init__()
        self.data = self._parse_csv(csv_path)
        self.root_path, _ = os.path.split(csv_path)
        self.transform = transform

    def __getitem__(self, idx):
        image_name, text_label = self.data[idx, 0], self.data[idx, 1]
        label_image = self._generate_text_from_image(text_label)
        image_path = os.path.join(self.root_path, self.image_folder, image_name)
        image = cv2.imread(image_path)
        image = self._resize_image(image)
        # image = cv2.resize(image, (256, 256))
        # label_image = cv2.resize(label_image, (256, 256))
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        label_image = torch.from_numpy(label_image).float().permute(2, 0, 1) / 255

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label_mask": label_image}

        return sample

    def __len__(self):
        return self.data.shape[0]

    def _parse_csv(self, csv_path):
        data = list()
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)

        return np.array(data)

    def _generate_text_from_image(self, text_string):
        rectangle_bgr = (255, 255, 255)
        img = np.ones((self.size[0], self.size[1], 3)) * 255
        (text_width, text_height) = cv2.getTextSize(text_string, self.font, fontScale=self.font_scale, thickness=1)[0]
        text_offset_x = 8
        text_offset_y = img.shape[0] - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, text_string, (text_offset_x, text_offset_y), self.font, fontScale=self.font_scale,
                    color=(0, 0, 0), thickness=2)

        return img

    def _resize_image(self, image):
        coefficient = self.size[0] / image.shape[0]
        image = cv2.resize(image, None, fx=coefficient, fy=coefficient)
        width_delta = self.size[1] - image.shape[1]
        if width_delta < 1:
            image = cv2.resize(image, (self.size[1], self.size[0]))
        else:
            white_addition = np.ones((self.size[0], width_delta, 3)) * 255
            image = np.concatenate((image, white_addition), axis=1).astype(np.uint8)

        return image


class CityScapesDataset(torch.utils.data.Dataset):

    """ CityScapes specific dataset!
    """
    annotations_path = "gtFine_trainvaltest/gtFine"
    images_path = "leftImg8bit_trainvaltest/leftImg8bit"
    label_mask_postfix = "_gtFine_color"

    def __init__(self, root_path, mode="train", transform=None):
        super().__init__()
        self.root_path = root_path
        self.mode = mode
        self.transform = transform
        self.name_vector = self._get_name_vector()

    def __getitem__(self, index):
        file_name = self.name_vector[index]
        label_name = '_'.join(file_name.split('_')[:-1]) + self.label_mask_postfix
        folder = file_name.split('_')[0]
        image_path = os.path.join(
            self.root_path, self.images_path, self.mode, folder, file_name) + ".png"
        label_path = os.path.join(
            self.root_path, self.annotations_path, self.mode, folder, label_name) + ".png"
        image = cv2.imread(image_path)
        label_mask = cv2.imread(label_path)

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        label_mask = cv2.resize(label_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image, label_mask = random_jitter(image, label_mask)

        # random mirroring
        if random.randint(0, 1):
            image, label_mask = flip(image, label_mask)

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        label_mask = torch.from_numpy(label_mask).float().permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label_mask": label_mask}

        return sample

    def __len__(self):
        return self.name_vector.shape[0]

    def _get_name_vector(self):
        base_path = os.path.join(self.root_path, self.images_path, self.mode, "*/*.png")
        images_pathes = glob.glob(base_path)
        images_names = list(map(lambda string: os.path.split(string)[-1].split('.')[0], images_pathes))
        name_vector = np.array(images_names)

        return name_vector


class CMPFacadeDataset(torch.utils.data.Dataset):
    """ CMP Facade Dataset reader
    https://cmp.felk.cvut.cz/~tylecr1/facade/
    """
    image_extension = ".jpg"
    label_extension = ".png"

    def __init__(self, base_dir, mode="base", transform=None, segmentation_mask=False):
        super().__init__()
        self.base_dir = base_dir
        self.mode = mode
        self.transform = transform
        self.segmentation_mask = segmentation_mask
        self.name_vector = self._get_name_vector()

    def __getitem__(self, index):
        file_name = self.name_vector[index]
        image_path = os.path.join(
            self.base_dir, self.mode, file_name) + self.image_extension
        label_path = os.path.join(
            self.base_dir, self.mode, file_name) + self.label_extension
        image = cv2.imread(image_path)
        label_mask = cv2.imread(label_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        label_mask = cv2.resize(label_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image, label_mask = random_jitter(image, label_mask)

        # random mirroring
        if random.randint(0, 1):
            image, label_mask = flip(image, label_mask)

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        label_mask = torch.from_numpy(label_mask).float().permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label_mask": label_mask}

        return sample

    def __len__(self):
        return self.name_vector.shape[0]

    def _get_name_vector(self):
        base_path = os.path.join(self.base_dir, self.mode, "*.png")
        image_paths = glob.glob(base_path)
        image_names = list(map(lambda string: os.path.basename(string).replace(".png", ""), image_paths))

        return np.array(image_names)


def random_jitter(image: np.ndarray,
                 mask: np.ndarray,
                 patch_h: int = 256,
                 patch_w: int = 256):
    image, mask = resize(image, mask, (286, 286))

    max_x = image.shape[1] - patch_w
    max_y = image.shape[0] - patch_h
    if max_x == 0 or max_y == 0:
        start_x = np.random.randint(0, 1)
        start_y = np.random.randint(0, 1)
    else:
        start_x = np.random.randint(0, max_x)
        start_y = np.random.randint(0, max_y)

    cropped_image = image[start_y:start_y + patch_h, start_x:start_x + patch_w, :]
    cropped_mask = mask[start_y:start_y + patch_h, start_x:start_x + patch_w]

    return cropped_image, cropped_mask


def flip(image: np.ndarray, mask: np.ndarray):
    return cv2.flip(image, 1), cv2.flip(mask, 1)


def resize(image, mask, size):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    return image, mask


if __name__ == "__main__":

    dataset = CMPFacadeDataset("../datasets/CMP_facade_DB_base")
    for sample in dataset:
        a = 1
