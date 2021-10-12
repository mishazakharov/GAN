import cv2
import torch


def save_checkpoint(path, net_G, net_D, optimizer_G, optimizer_D):
    checkpoint = {
        "net_G": net_G.state_dict(),
        "net_D": net_D.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict()
    }
    torch.save(checkpoint, path)


def save_weight(path, net_G):
    torch.save(net_G.state_dict(), path)


def get_noise(n_samples, z_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, glob_string, image_size, transform=None):
        self.data = glob_string
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index):
        path = self.data[index]
        image = cv2.imread(path)
        image = cv2.resize(image, self.image_size)
        image = torch.from_numpy(image).float().permute(2, 0, 1).contiguous() / 255

        if self.transform:
            image = self.transform(image)

        return {"image": image}

    def __len__(self):
        return self.data.__len__()


def write_logs(data, tensorboard, global_step):
    log_info = []
    for key, value in data.items():
        if isinstance(value, list):
            logging_value = value[0]
        else:
            logging_value = value

        tensorboard.add_scalar(key, logging_value, global_step)

        log_info.append(f"{key.upper()}: {value}")

    print(", ".join(log_info))


def generate_with_fixed_noise(net_G, fixed_noise, tensorboard, global_step, **kwargs):
    with torch.no_grad():
        fake_images = net_G(fixed_noise, **kwargs).detach().cpu()
        # Shift value range from [-1, 1] to [0, 1]
        fake_images = fake_images * 0.5 + 0.5
        # Tensorboard expects RGB images
        fake_images = torch.flip(fake_images, [1])

    tensorboard.add_images("Generator state images", fake_images, global_step)


def load_weights(path, model):
    weights = torch.load(path, map_location="cpu")
    model.load_state_dict(weights)
