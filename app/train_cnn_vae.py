import sys
import os
from os.path import join
parent_dir = os.path.abspath(join(os.getcwd(), os.pardir))
app_dir = join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

from pathlib import Path
import torch as pt
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from CNN_VAE import ConvDecoder, ConvEncoder, Autoencoder
from utils.training_loop import train_cnn_vae
import utils.config as config
from sklearn import metrics

# use GPU if possible
device = pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

DATA_PATH = Path(os.path.abspath('')) / "data"
OUTPUT_PATH = Path(os.path.abspath('')) / "output" / "VAE"

if __name__ == "__main__":
    # initialize CNN-VAE classes
    encoder = ConvEncoder(
        in_size=config.target_resolution,
        n_channels=config.input_channels,
        n_latent=config.latent_size,
        batchnorm=True,
        variational=True
    )

    decoder = ConvDecoder(
        in_size=config.target_resolution,
        n_channels=config.output_channels,
        n_latent=config.latent_size,
        batchnorm=True,
        squash_output=True
    )

    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)

    train_dataset = pt.load(join(DATA_PATH, "train_dataset.pt"))
    val_dataset = pt.load(join(DATA_PATH, "val_dataset.pt"))
    test_dataset = pt.load(join(DATA_PATH, "test_dataset.pt"))

    # train_dataset = train_dataset.astype(pt.float32)
    # val_dataset = val_dataset.astype(pt.float32)
    # test_dataset = test_dataset.astype(pt.float32)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # optimizer
    optimizer = pt.optim.Adam(autoencoder.parameters(), lr=config.learning_rate)

    # define score functions
    score_funcs = {
        "Lmax"  : metrics.max_error,
        "L1"    : metrics.mean_absolute_error,
        "R2"    : metrics.r2_score
    }

    test_result = train_cnn_vae(
        model=autoencoder,
        loss_func=nn.MSELoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        score_funcs=score_funcs,
        epochs=config.epochs,
        optimizer=optimizer
)