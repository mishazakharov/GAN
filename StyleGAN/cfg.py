import glob


# General settings:
data_path = glob.glob(
    "/home/misha/datasets/first_pages_passport_BIG/*.jpg")
checkpoints_path = False  # "/home/misha/GAN/StyleGAN/train_logs/PASSPORT_StyleGAN_training/checkpoints/final_ckpt_64x64_resolution.tar"
initial_image_size = 4
maximum_image_size = 256
log_folder = "train_logs/test"
num_workers = 0
device = "cuda:0"
num_images = 64

# Training procedure hyper-parameters:
batch_sizes = {4: 512, 8: 256, 16: 128, 32: 128, 64: 32, 128: 32, 256: 32, 512: 16, 1024: 8}
lr = {4: 0.001, 8: 0.001, 16: 0.001, 32: 0.001, 64: 0.001, 128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

phase_samples = 200_000
style_mixing = True
style_mixing_rate = 0.9

beta_1 = 0
beta_2 = 0.9

c_lambda = 10
disc_repeats = 1

seed = 5000

# Neural network hyper-parameters:
image_dim = 3

z_dim = 512
w_dim = 512

n_layers = 8

normalize = True
fused = True

# GAN training tricks:
weights_averaging = True
progressive_growing = True

log_step = 1  # 100
val_step = 1  # 500
save_step = 1  # 10_000
