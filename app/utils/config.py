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

# Pre-Processing with whole dataset
test_keys = ['ma0.84_alpha3.00', 'ma0.84_alpha5.00']
num_train_flow_conds = 5
train_split = 450
val_split = 50

mini_dataset = False
mini_train_per_cond = 25
mini_val_per_cond = 4
mini_test_per_cond = 4

# Pre-Processing for single flow cond pipeline
single_flow_cond_train_share = .8   # 80% training data -> 20% test data

########################################################
# SVD/ VAE Evaluation
########################################################

timestep_reconstruction = 100

########################################################
# Model parameters
########################################################

# CNN-VAE
input_channels = (1, 64, 128, 256, 512)
output_channels = (512, 256, 128, 64, 1)
latent_size = 256
epochs = 500
batch_size = 32
learning_rate = 1e-4
lr_factor = 0.1
patience = 5
min_loss = 1.05e5

# Fully-Connected



########################################################
# Plots
########################################################

standard_figsize_1 = (6, 3)
standard_figsize_2 = (6, 4)