import sys
import os
from os.path import join
from pathlib import Path

# include app directory into sys.path
parent_dir = Path(os.path.abspath(''))
app_dir = join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

from itertools import product
from collections import defaultdict
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
pt.manual_seed(0)

from utils.DataWindow import DataWindow_end_to_end
from autoencoder.CNN_VAE import ConvDecoder, ConvEncoder
from LSTM.LSTM_model import LSTM
from CNN_VAE_LSTM import autoencoder_LSTM
from utils.training_funcs import train_end_to_end
from utils.helper_funcs import delete_directory_contents, load_datasets_end_to_end
from utils import config

# use GPU if possible
device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
print("Computing device:        ", device)

# define parameters
PRED_HORIZON = config.E2E_pred_horizon
N_LATENT = config.E2E_latent_size
BATCH_SIZE = config.E2E_batch_size

# define paths
DATA_PATH = join(parent_dir, "data", "end_to_end")
OUTPUT_PATH = join(parent_dir, "output", "end_to_end", "single", f"pred_horizon_{PRED_HORIZON}")

# define study parameters of model
INPUT_WIDTHS = [32]
HIDDEN_SIZES = [64]
N_HIDDEN_LAYERS = [2]

def start_study_repeat(n_repeat):
    ''' run param study for different architectures with a given number of training iterations'''
    
    print("Training E2E models with varying model parameters: ")
    print("     input width:                ", INPUT_WIDTHS)
    print("     neurons in hidden layers:   ", HIDDEN_SIZES)
    print("     number of hidden layers:    ", N_HIDDEN_LAYERS)

    delete_directory_contents(OUTPUT_PATH)
    train, test = load_datasets_end_to_end(DATA_PATH)

    print("Starting study...")
    study_results = defaultdict(list)
    param_combinations = list(product(INPUT_WIDTHS, HIDDEN_SIZES, N_HIDDEN_LAYERS))
    print(f"'---> {len(param_combinations) * n_repeat} trainings in total")

    # train each model architecture n_repeat times
    for i in range(n_repeat):
        print("---Iteration %d--------" %(i+1))
        pt.manual_seed(i)
        for param_set in param_combinations:
            input_width, hidden_size, n_hidden_layers = param_set
            set_key = f"{input_width}_{hidden_size}_{n_hidden_layers}"
            print("--input_width={}, hidden_size={} and n_hidden={}".format(input_width, hidden_size, n_hidden_layers))

            # create DataWindow object to create windows of data, feed into DataLoaders
            data_window = DataWindow_end_to_end(train=train, test=test, input_width=input_width, pred_horizon=PRED_HORIZON)
            train_loader = DataLoader(data_window.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(data_window.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # initialize models
            encoder = ConvEncoder(
                in_size=config.target_resolution,
                n_channels=config.VAE_input_channels,
                n_latent=N_LATENT,
                variational=True,
                layernorm=True
            )
            decoder = ConvDecoder(
                in_size=config.target_resolution,
                n_channels=config.VAE_output_channels,
                n_latent=N_LATENT,
                layernorm=True,
                squash_output=True
            )
            lstm = LSTM(
                latent_size=N_LATENT, 
                hidden_size=hidden_size, 
                num_layers=n_hidden_layers)
            
            model = autoencoder_LSTM(encoder=encoder, LSTM=lstm, decoder=decoder)

            loss_func_latent = nn.MSELoss()
            optimizer = pt.optim.AdamW(model.parameters(), lr=config.E2E_learning_rate)

            # start training and append resoults to defaultdict
            study_results[f"{input_width}_{hidden_size}_{n_hidden_layers}"].append(train_end_to_end(
                model=model,
                loss_func=loss_func_latent,
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                epochs=config.E2E_epochs,
                device=device
            ))
            # create directory to save model state
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            model.save((join(OUTPUT_PATH, str(i + 1) + "_" + set_key)))
            print("\n")
        
    # save results of training metrics
    print("========== Study finished, saving results")
    pt.save(study_results, join(OUTPUT_PATH, "study_results.pt"))


if __name__ == '__main__':
    start_study_repeat(n_repeat=1)