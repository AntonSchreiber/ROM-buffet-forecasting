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
import matplotlib.pyplot as plt
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
pt.manual_seed(0)

from utils.DataWindow import DataWindow
from FC.FullyConnected import FullyConnected
from utils.training_funcs import train_AR_pred
import utils.config as config

# use GPU if possible
device = pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

SVD_PATH = join(Path(os.path.abspath('')), "output", "SVD")
DATA_PATH = join(Path(os.path.abspath('')), "data", "pipeline_single")
OUTPUT_PATH = join(Path(os.path.abspath('')), "output", "single_flow_cond")

N_LATENT = config.SVD_rank
INPUT_WIDTH = 90
PRED_HORIZON = 1

if __name__ == '__main__':
    # load data
    train_data = pt.load(join(DATA_PATH, "train_dataset.pt")).flatten(0, 1)
    test_data = pt.load(join(DATA_PATH, "test_dataset.pt")).flatten(0, 1)

    # load left singular vectors
    U = pt.load(join(SVD_PATH, "U.pt"))

    # reduce datasets 
    train_red = pt.transpose(U[:,:config.SVD_rank], 0, 1) @ train_data
    test_red = pt.transpose(U[:,:config.SVD_rank], 0, 1) @ test_data
    print("Shape of reduced train data:     ", train_red.shape)
    print("Shape of reduced test data:      ", test_red.shape)

    # create DataWindow object to create windows of data, feed into DataLoaders
    data_window = DataWindow(train=train_red, test=test_red, input_width=INPUT_WIDTH, pred_horizon=PRED_HORIZON)
    # print(data_window.rolling_window(100))
    train_loader = DataLoader(data_window.train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(data_window.test_dataset, batch_size=32, shuffle=True)

    # initialize prediction model
    hidden_size = 1024
    n_hidden_layers = 200

    model = FullyConnected(
        input_size=N_LATENT * INPUT_WIDTH,
        output_size=N_LATENT * PRED_HORIZON,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers
    )

    # define loss functions
    loss_func_latent = nn.MSELoss()
    loss_func_orig = nn.MSELoss()

    # define utilities
    optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.FC_patience, factor=config.FC_lr_factor)

    results = train_AR_pred(
        model=model,
        U=U[:,:config.SVD_rank],
        loss_func_latent=loss_func_latent,
        loss_func_orig=loss_func_orig,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        lr_schedule=scheduler,
        epochs=config.FC_epochs,
        device=device,
    )

    plt.plot(results["epoch"], results["train_loss"], lw=1, label="training")
    plt.plot(results["epoch"], results["val_loss"], lw=1, label="validation")
    plt.yscale("log")
    plt.xlim(0, 500)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(join(OUTPUT_PATH))
    plt.show()