import cv2
import tqdm
import glob
import scipy
import torch
import torchvision

import numpy as np


class FidDataset(torch.utils.data.Dataset):

    def __init__(self, glob_string, image_size):
        self.data = glob_string
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.data[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = torch.from_numpy(image).float().permute(2, 0, 1).contiguous() / 255

        return image

    def __len__(self):
        return self.data.__len__()


def get_inceptionv3_model():
    model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
    model.fc = torch.nn.Identity()

    return model


def frechet_inception_distance(real_stats, fake_stats):
    expected_value_real, covariance_matrix_real = real_stats
    expected_value_fake, covariance_matrix_fake = fake_stats
    cm_multiplication, _ = scipy.linalg.sqrtm(np.dot(covariance_matrix_real, covariance_matrix_fake), disp=False)
    fid = np.linalg.norm(expected_value_real - expected_value_fake,
                         ord=2) + np.trace(covariance_matrix_real + covariance_matrix_fake - 2 * cm_multiplication)

    return fid


def calculate_statistics(paths, model, batch_size=10, device="cuda:0", num_workers=0):
    dataset = FidDataset(paths, image_size=(299, 299))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    features = list()
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            batch = batch.to(device)
            output = model(batch)
            features.append(output.squeeze())

    features = torch.cat(features, dim=0).cpu().numpy()

    expected_value = np.mean(features, axis=0)
    covariance_matrix = np.cov(features, rowvar=False)

    return expected_value, covariance_matrix


if __name__ == "__main__":
    real_images_path = glob.glob("/home/misha/datasets/20_WIDE_NUMAKT/20ZTV02_19_WIDE_NUMAKT/*/images/*.jpg")
    fake_images_path = glob.glob("/home/misha/GANCourse/gen1/*.jpg")
    device = "cuda:0"
    batch_size = 256
    num_workers = 0
    model = get_inceptionv3_model().to(device)
    real_stats = calculate_statistics(
        real_images_path, model, device=device, batch_size=batch_size, num_workers=num_workers)
    fake_stats = calculate_statistics(
        fake_images_path, model, device=device, batch_size=batch_size, num_workers=num_workers)
    fid = frechet_inception_distance(real_stats, fake_stats)
    print(fid)
