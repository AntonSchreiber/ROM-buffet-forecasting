import sys
import os
from os.path import join
from pathlib import Path

# include app directory into sys.path
parent_dir = Path(os.path.abspath('')).parent.parent
app_dir = join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
from CNN_VAE import make_VAE_model
from utils.training_funcs import train_VAE
from utils.helper_funcs import delete_directory_contents
import utils.config as config
from collections import defaultdict

# set torch seed for reproducibility
pt.manual_seed(0)

# use GPU if possible
device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

# define paths
DATA_PATH = join(parent_dir ,"data", "VAE")
OUTPUT_PATH = join(parent_dir, "output", "VAE", "latent_study")

# define bottleneck sizes that should be part of the parameter study
latent_sizes = [8, 16, 32, 64, 128, 256, 512]


def start_latent_study_repeat(n_repeat, train_loader, val_loader, test_loader):
    ''' run param study for different architectures with a given number of training iterations'''
    print("Starting study...")

    delete_directory_contents(OUTPUT_PATH)
    study_results = defaultdict(list)

    # train each model architecture n_repeat times
    for i in range(n_repeat):
        print("---Iteration %d--------" %(i+1))
        pt.manual_seed(i)
        for latent_size in latent_sizes:
            print("   Training autoencoder with {} bottleneck neurons on {}".format(latent_size, device))

            # initialize model and utilities
            model = make_VAE_model(n_latent=latent_size, device=device)
            optimizer = pt.optim.Adam(model.parameters(), lr=config.VAE_learning_rate)
            scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=config.VAE_patience_scheduler, factor=config.VAE_lr_factor)

            # start training and add results to defaultdict
            study_results[str(latent_size)].append(
                train_VAE(
                    model=model,
                    loss_func=nn.MSELoss(),
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    epochs=config.VAE_epochs,
                    optimizer=optimizer,
                    lr_schedule=scheduler,
                    device=device
                ))
            # create directory to save model state
            subfolder = join(OUTPUT_PATH, str(latent_size))
            os.makedirs(subfolder, exist_ok=True)
            model.save((join(subfolder, str(i + 1) + "_" + str(latent_size))))
            print("\n")
    
    # save results of training metrics
    print("========== Study finished, saving results")
    pt.save(study_results, join(OUTPUT_PATH, "study_results.pt"))
    

if __name__ == "__main__":
    print("Training CNN VAE models with latent sizes: ", latent_sizes)

    # load datasets (which are already TensorDataset objects in this case)
    train_dataset = pt.load(join(DATA_PATH, "train_dataset.pt"))
    val_dataset = pt.load(join(DATA_PATH, "val_dataset.pt"))
    test_dataset = pt.load(join(DATA_PATH, "test_dataset.pt"))

    # feed to dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.VAE_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.VAE_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.VAE_batch_size, shuffle=True)

    # start study with 10 trainings for each latent size
    start_latent_study_repeat(10, train_loader, val_loader, test_loader)