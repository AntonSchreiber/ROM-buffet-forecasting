import sys
import os
from os.path import join
parent_dir = os.path.abspath(join(os.getcwd(), os.pardir))
app_dir = join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

import shutil
from pathlib import Path
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
from CNN_VAE import ConvDecoder, ConvEncoder, Autoencoder
from utils.training_loop import train_cnn_vae
import utils.config as config
from utils.EarlyStopper import EarlyStopper
import matplotlib.pyplot as plt
from collections import defaultdict

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

latent_sizes = [8, 16, 64, 128, 256, 512]

# function to create VAE model
def make_VAE_model(n_latent: int) -> pt.nn.Module:
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


def start_latent_study(train_loader, val_loader, test_loader):
    # start study
    print("Running study...")
    results = []
    for latent_size in latent_sizes:
        print("Training autoencoder with {} bottleneck neurons ...".format(latent_size))
        model = make_VAE_model(latent_size)
        print(model)
        optimizer = pt.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.patience, factor=config.lr_factor)

        results.append(train_cnn_vae(
            model=model,
            loss_func=nn.MSELoss(),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
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


def start_latent_study_repeat(n_repeat, train_loader, val_loader, test_loader):
    # start study
    print("Starting study...")

    delete_directory_contents(OUTPUT_PATH)
    study_results = defaultdict(list)

    for i in range(n_repeat):
        print("---Iteration %d--------" %(i+1))
        pt.manual_seed(i)
        for latent_size in latent_sizes:
            print("   Training autoencoder with {} bottleneck neurons ...".format(latent_size))

            # initialize model and utilities
            model = make_VAE_model(latent_size)
            optimizer = pt.optim.Adam(model.parameters(), lr=config.learning_rate)
            scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.patience, factor=config.lr_factor)
            early_stopper = EarlyStopper(patience=50, mode='min')

            # start training and add results to defaultdict
            study_results[str(latent_size)].append(
                train_cnn_vae(
                    model=model,
                    loss_func=nn.MSELoss(),
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    epochs=config.epochs,
                    optimizer=optimizer,
                    lr_schedule=scheduler,
                    device=device,
                    early_stopper=early_stopper
                ))
            # create directory to save model state
            subfolder = join(OUTPUT_PATH, str(latent_size))
            os.makedirs(subfolder, exist_ok=True)
            model.save((join(subfolder, str(i + 1) + "_" + str(latent_size))))
            print("\n")
    
    # save results of training metrics
    print("========== Study finished, saving results")
    pt.save(study_results, join(OUTPUT_PATH, "study_results.pt"))


def delete_directory_contents(directory_path):
    """ Delete directory contents with given path """
    try:
        # Get a list of all files and subdirectories in the directory
        file_list = os.listdir(directory_path)

        # Loop through the list and remove each file and subdirectory
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_directory_contents(file_path)  # Recursively delete subdirectories
                os.rmdir(file_path)  # Remove the empty subdirectory after its contents are deleted

        print(f"Successfully deleted all contents in {directory_path}.")
    except Exception as e:
        print(f"Error occurred while deleting contents in {directory_path}: {e}")

    

if __name__ == "__main__":
    print("Training CNN VAE models with latent sizes: ", latent_sizes)
    # load data
    train_dataset = pt.load(join(DATA_PATH, "train_dataset.pt"))
    val_dataset = pt.load(join(DATA_PATH, "val_dataset.pt"))
    test_dataset = pt.load(join(DATA_PATH, "test_dataset.pt"))

    # fed to dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # start training
    start_latent_study_repeat(10, train_loader, val_loader, test_loader)