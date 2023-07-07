# CONFIGURATION FILE

########################################################
# Paths
########################################################
# TODO add later if time left

########################################################
# Pre-Processing
########################################################

time_steps_per_cond = 500
target_resolution = (256, 128)
target_tensor_shape = (256, 128, 500)


test_keys = ['ma0.84_alpha3.00', 'ma0.84_alpha5.00']
num_train_flow_conds = 5
train_split = 450
val_split = 50

mini_datset = True
mini_train_per_cond = 10
mini_val_per_cond = 1
mini_test_per_cond = 2

########################################################
# Model parameters
########################################################

epochs = 500
batch_size = 8
learning_rate = 1e-4
min_loss = 1.05e5


# CNN-VAE
input_channels = (1, 64, 128, 256, 512)
output_channels = (512, 256, 128, 64, 1)
latent_size = 256

