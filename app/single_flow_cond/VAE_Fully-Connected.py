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
from utils.TimeSeriesDataset import TimeSeriesDataset
from utils.FullyConnected import FullyConnected
from utils.CNN_VAE import make_encoder_model, make_decoder_model
from utils.EarlyStopper import EarlyStopper
from utils.training_funcs import train_AR_pred
import utils.config as config

# use GPU if possible
device = pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

VAE_PATH = join(Path(os.path.abspath('')), "output", "VAE", "latent_study", config.VAE_model)
# VAE_PATH = join(Path(os.path.abspath('')), "output", "VAE", "latent_study", "128", "3_128")
DATA_PATH = join(Path(os.path.abspath('')), "data", "pipeline_single")
OUTPUT_PATH = join(Path(os.path.abspath('')), "output", "single_flow_cond", "parameter_study")

N_LATENT = 16
PRED_HORIZON = 1

INPUT_WIDTHS = [30, 40, 50, 60, 70]
HIDDEN_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]
N_HIDDEN_LAYERS = [1, 2, 3, 4, 5, 6]

def start_study():
    print("Training Fully-Connected models with varying model parameters: ")
    print("     input width:                ", INPUT_WIDTHS)
    print("     neurons in hidden layers:   ", HIDDEN_SIZES)
    print("     number of hidden layers:    ", N_HIDDEN_LAYERS)

    # start encoding
    train_enc, test_enc = encode_datasets()

    # start study
    print("Starting study...")
    study_results = defaultdict(list)
    param_combinations = list(product(INPUT_WIDTHS, HIDDEN_SIZES, N_HIDDEN_LAYERS))

    for param_set in param_combinations:
        input_width, hidden_size, n_hidden = param_set
        set_key = f"{input_width}_{hidden_size}_{n_hidden}"
        print("--input_width={}, hidden_size={} and n_hidden={}".format(input_width, hidden_size, n_hidden))

        # create TimeSeriesDataset object to create windows of data, feed into DataLoaders
        timeseriesdataset = TimeSeriesDataset(train=train_enc, test=test_enc, input_width=input_width, pred_horizon=PRED_HORIZON)
        train_loader = DataLoader(timeseriesdataset.train_dataset, batch_size=config.FC_batch_size, shuffle=True)
        test_loader = DataLoader(timeseriesdataset.test_dataset, batch_size=config.FC_batch_size, shuffle=True)
        
        # initialize model and utilities
        model = FullyConnected(
            input_size=N_LATENT * input_width,
            output_size=N_LATENT,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden
        )

        loss_func_latent = nn.MSELoss()
        optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.FC_patience_scheduler, factor=config.FC_lr_factor)
        earlystopper = EarlyStopper(patience=config.FC_patience_earlystop)

        # start training and append resoults to defaultdict
        study_results[f"{input_width}_{hidden_size}_{n_hidden}"].append(train_AR_pred(
            model=model,
            loss_func=loss_func_latent,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            lr_schedule=scheduler,
            epochs=config.FC_epochs,
            device=device,
            early_stopper=earlystopper
        ))
        # create directory to save model state
        subfolder = join(OUTPUT_PATH, "pred_horizon_1")
        os.makedirs(subfolder, exist_ok=True)
        pt.save(model.state_dict(), join(subfolder, set_key + ".pt"))
        print("\n")
    
    # save results of training metrics
    print("========== Study finished, saving results")
    pt.save(study_results, join(OUTPUT_PATH, "study_results.pt"))


def encode_datasets():
    # load data
    print("Loading datasets ... ")
    train_data = pt.load(join(DATA_PATH, "train_dataset.pt"))
    test_data = pt.load(join(DATA_PATH, "test_dataset.pt"))

    # load encoder and decoder model
    print("Loading pre-trained encoder and decoder model ...")
    encoder = make_encoder_model(n_latent=N_LATENT, device=device)
    encoder.load_state_dict(pt.load(VAE_PATH + "_encoder.pt", map_location=pt.device("cpu")))
    decoder = make_decoder_model(n_latent=N_LATENT, device=device)
    decoder.load_state_dict(pt.load(VAE_PATH + "_decoder.pt", map_location=pt.device("cpu")))

    # encode datasets 
    print("Encoding datasets ...")
    with pt.no_grad():
        train_enc = pt.stack([encoder(train_data[:, :, n].unsqueeze(0).unsqueeze(0)).squeeze(0).detach() for n in range(train_data.shape[2])], dim=1)
        test_enc = pt.stack([encoder(test_data[:, :, n].unsqueeze(0).unsqueeze(0)).squeeze(0).detach() for n in range(test_data.shape[2])], dim=1)
        print("     Shape of encoded train data:     ", train_enc.shape)
        print("     Shape of encoded test data:      ", test_enc.shape, "\n")

    # scale data
    print("Scaling encoded data to [-1, 1] ... ")
    print("     min and max train cp prior scaling:     ", train_enc.min().item(), train_enc.max().item())
    scaler = MinMaxScaler_1_1().fit(train_enc)
    print("     scaling ...")
    train_enc, test_enc = scaler.scale(train_enc), scaler.scale(test_enc)
    print("     min and max train cp after scaling:     ", train_enc.min().item(), train_enc.max().item(), "\n")    

    return train_enc, test_enc


if __name__ == '__main__':
    start_study()

    

