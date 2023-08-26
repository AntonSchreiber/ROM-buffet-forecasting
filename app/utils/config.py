# CONFIGURATION FILE

import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Pre-Processing for SVD and VAE
test_keys_all = ['ma0.84_alpha3.00', 'ma0.84_alpha5.00']
num_train_flow_conds = 5
train_split_all = 450
val_split_all = 50

mini_dataset = False
mini_train_per_cond = 25
mini_val_per_cond = 4
mini_test_per_cond = 4

# Pre-Processing for single flow cond pipeline
single_flow_cond = 'ma0.84_alpha4.00'
single_flow_cond_train_share = .8   # 90% training data -> 10% test data

# Pre-Processing for single flow cond pipeline
train_keys_multi = ['ma0.84_alpha3.00', 'ma0.84_alpha3.50', 'ma0.84_alpha4.50', 'ma0.84_alpha5.00']
test_keys_multi = ['ma0.84_alpha4.00']
train_split_multi = 420
val_split_multi = 80

########################################################
# Model Optimization
########################################################

timestep_reconstruction = 100
timestep_prediction_single = (40, 50)
timestep_prediction_multi = (150, 200)
timestep_prediction = 250

########################################################
# Model parameters
########################################################

#### Dim Reduction ####
# SVD
SVD_rank = 300

# CNN-VAE
VAE_input_channels = (1, 64, 128, 256, 512)
VAE_output_channels = (512, 256, 128, 64, 1)
VAE_epochs = 500
VAE_batch_size = 32
VAE_learning_rate = 1e-4
VAE_lr_factor = 0.1
VAE_patience_scheduler = 5
VAE_patience_earlystop = 50
VAE_latent_size = 128

VAE_model = "128/4_128"

#### Time Evolution ####

input_width = 32

# Fully-Connected (FC)
FC_learning_rate = 1e-4
FC_lr_factor = 0.5
FC_single_epochs = 6000
FC_multi_epochs = 1000
FC_patience_scheduler = 5
FC_patience_earlystop = 50

FC_SVD_single_batch_size = 128
FC_VAE_single_batch_size = 512
FC_SVD_multi_batch_size = 512
FC_VAE_multi_batch_size = 1024

FC_SVD_single_model = "2_32_256_4"
FC_VAE_single_model = "7_32_256_6"

# Long Short-Term Memory (LSTM)
LSTM_learning_rate = 1e-4
LSTM_lr_factor = 0.1
LSTM_single_epochs = 2000
LSTM_multi_epochs = 2000
LSTM_patience_scheduler = 5
LSTM_patience_earlystop = 50

LSTM_SVD_single_batch_size = 128
LSTM_VAE_single_batch_size = 128
LSTM_SVD_multi_batch_size = 512
LSTM_VAE_multi_batch_size = 1024


########################################################
# Plots
########################################################

U_inf = 211             # m/s
a = 295                 # m/s
c_mean = 0.1965         # m
timesteps_per_second = 2000

standard_figsize_1 = (6, 3)
standard_figsize_2 = (6, 4)
power_sepctra_figsize = (10, 4)
plot_lims_MSE_general = [2e-3, 1.3e-2]
plot_lims_R_squarred = [0.93, 1]
plot_lims_MSE_spatial = (0, 0.018)
plot_lims_MSE_temporal = [1.5e-3, 1.5e-2]
plot_lims_cp = (-1, 1)
plot_lims_MSE_reconstruction = [0, 0.05]
plot_lims_MSE_FC_single_heatmap = [3e-1, 1e1]