# implement and test a training pipeline for a single flow condition and a single network architecture
# test the incremental increase of the prediction horizon to improve long-term stability

# 1. Load pre-processed data
# 2. Encode into reduced space
# 3. Train Fully-Connected model in reduced space
# 4. Compute loss in reduced space
# 5. Decode into full-space
# 6. Compute loss in full-space

import os
from os.path import join
from pathlib import Path
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
pt.manual_seed(0)

from utils.Scaler import MinMaxScaler_1_1
from utils.DataWindow import DataWindow
from utils.FullyConnected import FullyConnected
from utils.CNN_VAE import make_VAE_model
from utils.EarlyStopper import EarlyStopper
from utils.training_funcs import train_AR_pred
from utils.helper_funcs import delete_directory_contents
import utils.config as config

# use GPU if possible
device = pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

VAE_PATH = join(Path(os.path.abspath('')), "output", "VAE", "latent_study", config.VAE_model)
DATA_PATH = join(Path(os.path.abspath('')), "data", "single_flow_cond")
OUTPUT_PATH = join(Path(os.path.abspath('')), "output", "VAE_FC", "param_study", "pred_horizon_5")

N_LATENT = 32
PRED_HORIZON = 5

INPUT_WIDTHS = [35, 40, 45, 50, 55]
HIDDEN_SIZES = [8, 16, 32, 64, 128, 256, 512]
N_HIDDEN_LAYERS = [1, 2, 3, 4]

def start_study():
    print("Training Fully-Connected models with varying model parameters: ")
    print("     input width:                ", INPUT_WIDTHS)
    print("     neurons in hidden layers:   ", HIDDEN_SIZES)
    print("     number of hidden layers:    ", N_HIDDEN_LAYERS)

    # delete_directory_contents(OUTPUT_PATH)

    # start encoding
    train_enc, test_enc = reduce_datasets()

    # start study
    print("Starting study...")
    study_results = defaultdict(list)
    param_combinations = list(product(INPUT_WIDTHS, HIDDEN_SIZES, N_HIDDEN_LAYERS))

    for param_set in param_combinations:
        input_width, hidden_size, n_hidden_layers = param_set
        set_key = f"{input_width}_{hidden_size}_{n_hidden_layers}"
        print("--input_width={}, hidden_size={} and n_hidden={}".format(input_width, hidden_size, n_hidden_layers))

        # create DataWindow object to create windows of data, feed into DataLoaders
        data_window = DataWindow(train=train_enc, test=test_enc, input_width=input_width, pred_horizon=PRED_HORIZON)
        train_loader = DataLoader(data_window.train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(data_window.test_dataset, batch_size=32, shuffle=True)
        
        # initialize model and utilities
        model = FullyConnected(
            latent_size=N_LATENT,
            input_width=input_width,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers
    )

        loss_func_latent = nn.MSELoss()
        optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.FC_patience_scheduler, factor=config.FC_lr_factor)
        earlystopper = EarlyStopper(patience=config.FC_patience_earlystop)

        # start training and append resoults to defaultdict
        study_results[f"{input_width}_{hidden_size}_{n_hidden_layers}"].append(train_AR_pred(
            model=model,
            loss_func=loss_func_latent,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            lr_schedule=scheduler,
            epochs=config.FC_epochs,
            device=device,
            early_stopper=earlystopper,
            pred_horizon=PRED_HORIZON
        ))
        pt.save(model.state_dict(), join(OUTPUT_PATH, set_key + ".pt"))
        print("\n")
    
    # save results of training metrics
    print("========== Study finished, saving results")
    pt.save(study_results, join(OUTPUT_PATH, "study_results.pt"))


def reduce_datasets():
    # load data
    print("Loading datasets ... ")
    train_data = pt.load(join(DATA_PATH, "VAE_train.pt"))
    test_data = pt.load(join(DATA_PATH, "VAE_test.pt"))
    print("     min and max train cp prior encoding:     ", train_data.min().item(), train_data.max().item())

    # load pre-trained autoencoder model
    autoencoder = make_VAE_model(n_latent=N_LATENT, device=device)
    autoencoder.load(VAE_PATH)
    autoencoder.eval()

    # encode datasets 
    print("Encoding datasets ...")
    train_enc = autoencoder.encode_dataset(train_data)
    test_enc = autoencoder.encode_dataset(test_data)
    print("     Shape of encoded train data:     ", train_enc.shape)
    print("     Shape of encoded test data:      ", test_enc.shape, "\n")
    print("     min and max train cp after encoding:     ", train_enc.min().item(), train_enc.max().item())

    # scale data
    print("Scaling encoded data to [-1, 1] ... ")
    scaler = MinMaxScaler_1_1().fit(train_enc)
    train_enc, test_enc = scaler.scale(train_enc), scaler.scale(test_enc)
    print("     min and max train cp after scaling:     ", train_enc.min().item(), train_enc.max().item(), "\n")    

    print("Saving scaler for inference")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    pt.save(scaler, join(OUTPUT_PATH, "scaler.pt"))

    return train_enc, test_enc


if __name__ == '__main__':
    start_study()

    

