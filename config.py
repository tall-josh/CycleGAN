import os

sim_test_lap = "/home/users/josh/repos/mine/CycleGAN/donkey_sim_test_lap/*.jpg"
pattern_A = sim_test_lap
#pattern_A = "../_CycleGAN/data/donkeycar/donkey_warehouse_sim/*.jpg"
pattern_B = "../_CycleGAN/data/donkeycar/make_create_warehouse/*.jpg"
pattern_B = pattern_A


img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width
batch_size = 1
pool_size = 50
ngf = 32
ndf = 64

mode = "test"
#to_train = True
#to_test = False
to_restore = False
output_dir = "./donkey_004"
check_dir = os.path.join(output_dir, "checkpoints")

temp_check = 0

max_epoch = 1
max_images = 819

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
save_training_images = True

