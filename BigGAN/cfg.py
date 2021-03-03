import glob

device = "cuda:0"
z_dim = 120
image_size = 128
batch_size = 8
beta_1 = 0.0
beta_2 = 0.999
disc_repeats = 2
orth_reg = True
lr_G = 0.0002
lr_D = 0.00005
num_workers = 0
num_images = 16
n_classes = 4
seed = 5000
log_folder = "train_logs/test"
data_path = glob.glob("/home/misha/datasets/20_WIDE_NUMAKT/*/*/*_train.csv")
