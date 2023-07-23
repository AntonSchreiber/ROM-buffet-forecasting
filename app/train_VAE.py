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
import matplotlib.pyplot as plt

pt.manual_seed(711)
plt.rcParams["figure.dpi"] = 180

# use GPU if possible
device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

# remote device
DATA_PATH = join(Path(os.path.abspath('')).parent, "data")
OUTPUT_PATH = join(Path(os.path.abspath('')).parent, "output", "VAE", "latent_study")

# local
# DATA_PATH = join(Path(os.path.abspath('')), "data")
# OUTPUT_PATH = join(Path(os.path.abspath('')), "output", "VAE", "latent_study")

# latent_sizes = list(range(10, 311, 20))
latent_sizes = [16, 32, 64, 128, 256, 512]

# function to create VAE model
def make_VAE_model(n_latent: int = 256) -> pt.nn.Module:
    encoder = ConvEncoder(
        in_size=config.target_resolution,
        n_channels=config.input_channels,
        n_latent=n_latent,
        variational=True,
        layernorm=True
    )

    decoder = ConvDecoder(
        in_size=config.target_resolution,
        n_channels=config.output_channels,
        n_latent=n_latent,
        layernorm=True,
        squash_output=True
    )

    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)
    return autoencoder


def start_latent_study(train_loader, val_loader):
     # start study
    print("Running study...")
    results = []
    for latent_size in latent_sizes:
        print("Training autoencoder with {} bottleneck neurons ...".format(latent_size))
        model = make_VAE_model(latent_size)
        print(model)
        optimizer = pt.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.patience, factor=config.lr_factor, verbose=True)

        results.append(train_cnn_vae(
            model=model,
            loss_func=nn.MSELoss(),
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            optimizer=optimizer,
            lr_schedule=scheduler,
            device=device
        ))
        # create directory to save model state
        subfolder = join(OUTPUT_PATH, str(latent_size))
        os.makedirs(subfolder, exist_ok=True)
        model.save(str(join(OUTPUT_PATH, str(latent_size), str(latent_size))))
        print("\n")
    pt.save(results, join(OUTPUT_PATH, "training_results.pt"))

    # plot_results(results)


def plot_results(results):
    # results = pt.load(join(OUTPUT_PATH, "training_results.pt"))
    # plot study results, this will be in the "Training the Autoencoder" chapter
    for i, latent_size in enumerate(latent_sizes):
        plt.plot(results[i]["epoch"], results[i]["val_loss"], lw=1, label="{} bottleneck neurons".format(latent_size))

    plt.yscale("log")
    plt.xlim(0, config.epochs)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(Path(OUTPUT_PATH).parent, "Val_loss_results.png"))

if __name__ == "__main__":
    # print("DATA PATH:   ", DATA_PATH)
    # print("OUTPUT PATH: ", OUTPUT_PATH)
    print("Training CNN VAE models with latent sizes: \n", latent_sizes)
    # load data
    train_dataset = pt.load(join(DATA_PATH, "train_dataset.pt"))
    val_dataset = pt.load(join(DATA_PATH, "val_dataset.pt"))

    # fed to dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    # start training
    start_latent_study(train_loader, val_loader)