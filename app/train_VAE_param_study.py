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
from torch.utils.data import DataLoader
from CNN_VAE import ConvDecoder, ConvEncoder, Autoencoder
from utils.training_loop import train_cnn_vae
import utils.config as config
from utils.EarlyStopper import EarlyStopper

pt.manual_seed(0)

# use GPU if possible
device = pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")
print(device)

DATA_PATH = Path(os.path.abspath('')).parent / "data"
OUTPUT_PATH = Path(os.path.abspath('')).parent / "output" / "VAE" /"parameter_study"
print(DATA_PATH)
print(OUTPUT_PATH)

test_case_name = "256_batchnorm_lr1e-4_Plateau_f0.8nosquash"

latent_size = 256

# function to create VAE model
def make_VAE_model(n_latent: int = 256) -> pt.nn.Module:
    encoder = ConvEncoder(
        in_size=config.target_resolution,
        n_channels=config.input_channels,
        n_latent=n_latent,
        variational=True,
        batchnorm=True
    )

    decoder = ConvDecoder(
        in_size=config.target_resolution,
        n_channels=config.output_channels,
        n_latent=n_latent,
        batchnorm=True,
        squash_output=False
    )

    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)
    return autoencoder



if __name__ == "__main__":
    print("Starting parameter study:")
    print("-------------------------------")
    print("    Loading Data")
    train_dataset = pt.load(join(DATA_PATH, "train_dataset.pt"))
    val_dataset = pt.load(join(DATA_PATH, "val_dataset.pt"))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    print("-------------------------------")
    print("    Creating model")
    autoencoder = make_VAE_model(n_latent=256)

    # optimizer
    optimizer = pt.optim.Adam(autoencoder.parameters(), lr=1e-4)

    # learning rate scheduler
    scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.8, patience=5)
    # mode="min" means that the lr will be reduced when the MSE has stopped decreasing
    # factor states by which factor the lr will be reduced on stagnation

    # early stopper
    early_stopper = EarlyStopper(patience=50, mode='min')

    print("-------------------------------")
    print("    Training model")
    test_result = train_cnn_vae(
        model=autoencoder,
        loss_func=nn.MSELoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        optimizer=optimizer,
        lr_schedule=scheduler,
        device=device,
        early_stopper=early_stopper
    )

    print("-------------------------------")
    print("    Saving results")
    # save the test model
    autoencoder.save(str(join(OUTPUT_PATH, "test")))
    pt.save(test_result, join(OUTPUT_PATH, "test_results.pt"))