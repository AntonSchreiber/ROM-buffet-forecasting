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
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
pt.manual_seed(0)

from utils.TimeSeriesDataset import TimeSeriesDataset
from utils.FullyConnected import FullyConnected
from utils.CNN_VAE import make_encoder_model, make_decoder_model
from utils.training_funcs import train_AR_with_VAE
import utils.config as config

# use GPU if possible
device = pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

VAE_PATH = join(Path(os.path.abspath('')), "output", "VAE", "latent_study", "128", "3_128")
DATA_PATH = join(Path(os.path.abspath('')), "data", "pipeline_single")
OUTPUT_PATH = join(Path(os.path.abspath('')), "output", "single_flow_cond")

N_LATENT = 128
INPUT_WIDTH = 50
PRED_HORIZON = 1

if __name__ == '__main__':
    # load data
    train_data = pt.load(join(DATA_PATH, "train_dataset.pt"))
    test_data = pt.load(join(DATA_PATH, "test_dataset.pt"))

    # load encoder and decoder model
    encoder = make_encoder_model(n_latent=128, device=device)
    encoder.load_state_dict(pt.load(VAE_PATH + "_encoder.pt", map_location=pt.device("cpu")))
    decoder = make_decoder_model(n_latent=128, device=device)
    decoder.load_state_dict(pt.load(VAE_PATH + "_decoder.pt", map_location=pt.device("cpu")))

    # encode datasets 
    with pt.no_grad():
        train_enc = pt.stack([encoder(train_data[:, :, n].unsqueeze(0).unsqueeze(0)).squeeze(0).detach() for n in range(train_data.shape[2])], dim=1)
        test_enc = pt.stack([encoder(test_data[:, :, n].unsqueeze(0).unsqueeze(0)).squeeze(0).detach() for n in range(test_data.shape[2])], dim=1)
        print("Shape of encoded train data:     ", train_enc.shape)
        print("Shape of encoded test data:      ", test_enc.shape)

    # create TimeSeriesDataset object to create windows of data, feed into DataLoaders
    timeseriesdataset = TimeSeriesDataset(train=train_enc, test=test_enc, input_width=INPUT_WIDTH, pred_horizon=PRED_HORIZON)
    # print(timeseriesdataset.rolling_window(100))
    train_loader = DataLoader(timeseriesdataset.train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(timeseriesdataset.test_dataset, batch_size=32, shuffle=True)

    # initialize prediction model
    hidden_size = 1024
    n_hidden_layers = 100

    model = FullyConnected(
        input_size=N_LATENT * INPUT_WIDTH,
        output_size=N_LATENT * PRED_HORIZON,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers
    )

    # define loss functions
    loss_func_latent = nn.MSELoss()
    loss_func_orig = nn.MSELoss()

    optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.FC_patience, factor=config.FC_lr_factor)

    results = train_AR_with_VAE(
        model=model,
        loss_func_latent=loss_func_latent,
        loss_func_orig=loss_func_orig,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        lr_schedule=scheduler,
        epochs=config.FC_epochs,
        device=device,
        decoder=decoder
    )

    