import os
import csv

import cv2
import glob
import torch

import numpy as np


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
    segmentation_mask_postfix = "_gtFine_color"
    label_mask_postfix = "_gtFine_labelIds"

    def __init__(self, root_path, mode="train", transform=None, segmentation_mask=False):
        super().__init__()
        self.root_path = root_path
        self.mode = mode
        self.transform = transform
        self.segmentation_mask = segmentation_mask
        self.name_vector = self.__get_name_vector()

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

        if self.segmentation_mask:
            segmentation_name = '_'.join(file_name.split('_')[:-1]) + self.segmentation_mask_postfix
            segmentation_path = os.path.join(
                self.root_path, self.annotations_path, self.mode, folder, segmentation_name) + ".png"
            segmentation_mask = cv2.imread(segmentation_path)
            segmentation_mask = cv2.resize(segmentation_mask, (256, 256))

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        label_mask = cv2.resize(label_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        label_mask = torch.from_numpy(label_mask).float().permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "label_mask": label_mask}
        if self.segmentation_mask:
            sample["segmentation_mask"] = segmentation_mask

        return sample

    def __len__(self):
        return self.name_vector.shape[0]

    def __get_name_vector(self):
        base_path = os.path.join(self.root_path, self.images_path, self.mode, "*/*.png")
        images_pathes = glob.glob(base_path)
        images_names = list(map(lambda string: os.path.split(string)[-1].split('.')[0], images_pathes))
        name_vector = np.array(images_names)

        return name_vector


if __name__ == "__main__":
    a = 1
    dataset = TextGenerationDataset(
        "/home/misha/datasets/passports_word_annotations/test_campaign/test_template/train.csv")

    for sample in dataset:
        pass
