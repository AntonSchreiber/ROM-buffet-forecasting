
#######################################################################################################################
# settings file for iPSP data preprocessing and neural network prediction
#######################################################################################################################
# import modules
import os

#######################################################################################################################
# Path definition
#######################################################################################################################
data_path = '/local/disk1/rzahn/iPSP/'
file_path = os.path.join(data_path,'iPSP_Data/')
preprocessed_data_path = os.path.join(data_path, 'Preprocessed_Data/')

train_path = os.path.join(preprocessed_data_path, 'Training_Data/')
validation_path = os.path.join(preprocessed_data_path, 'Validation_Data/')
test_path = os.path.join(preprocessed_data_path, 'Test_Data/')

result_path = os.path.join(data_path, 'Results/')
results_AE = os.path.join(result_path, 'Results_AE/')
results_LSTM = os.path.join(result_path, 'Results_LSTM/')
results_hybrid_ROM = os.path.join(result_path, 'Results_Hybrid_ROM/')

#######################################################################################################################
# Folder definition
#******************************************************************************************************************#
# data set definition
# define training, validation and test data for pre-processing
train_valid_file = '0410.hdf5'
test_file = '0408.hdf5'

# files of data sets after preprocessing (depending on the included number of data points)
train_data = 'train_data_two_flow_new.npy'
validation_data = 'validation_data_two_flow_new.npy'
test_data = 'test_0409_500.npy'

# chose preprocessing option
preprocess_single_data_set = 1
preprocess_several_data_sets = 0
save_datasets_separately = 0
#******************************************************************************************************************#
# parameters for pre-processing script

# define number of snapshots applied for training, validation and test
n_snapshots_train = 10
n_snapshots_validation = 500
n_snapshots_test = 10

n_snapshots_several_train = 600
n_snapshots_several_validation = 100

# define target surface resolution
surf_resolution = (256, 128)
# define target scale limits
scale_range = [-1, 1]

select_train_valid_data = 0
select_test_data = 0

#******************************************************************************************************************#
# select training or test modus of model
train_model = 1
test_model = 0
call_decoder = 0

# select if training, validation or test data is encoded 
decode_train_valid_data = 1
decode_test_data = 0

# select if results from one-step or multi-step predictions are encoded 
encode_multiStep = 0
encode_oneStep = 0

#******************************************************************************************************************#
# Hyperparameter Definition

epochs = 500
batch_size = 32
learning_rate = 0.001
min_loss = 1.0e5

# CNN-AE
input_channels = (1, 64, 128, 256, 512)
output_channels = (512, 256, 128, 64, 1)
latent_size = 256

# LSTM
hidden_size = 512
num_layers = 4
init_seq_steps = 50
num_predictions = 'all'

# folder definition
save_folder_CNN_AE = 'results_two_flow_%i_%i_%i' % (batch_size, latent_size, epochs)
save_folder_LSTM = 'results_%i_%i_%i' % (batch_size, num_layers, hidden_size)

#******************************************************************************************************************#
# plot settings
textsize = 16

#******************************************************************************************************************#
# select options for pre- and postprocessing
plot_resolution_study = 0
plot_losses_train_valid = 1
plot_losses_test = 0
load_single_timestep_train_valid = 0
load_single_timestep_test = 0
load_all_timesteps_train_valid = 0
load_all_timesteps_test = 0
timestep = 50


