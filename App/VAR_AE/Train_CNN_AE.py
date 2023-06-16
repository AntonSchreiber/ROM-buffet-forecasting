#################################################################################################################
# import modules
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from CNN_AE import ConvDecoder, ConvEncoder, Autoencoder

# import settings file 
import Settings_iPSP as settings_file

#################################################################################################################
# use GPU if possible
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#################################################################################################################
def train_CNN_AE(train_data,
                 train_path,
                 model,
                 batch_size,
                 optimizer,
                 criterion,
                 device
                 ):

    train_dataset = np.load(os.path.join(train_path, train_data))
    train_dataset = train_dataset.astype(np.float32)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)


    model.train()
    train_batch_loss = 0

    for surface_data in train_loader:
        surface_data = surface_data.to(device)

        optimizer.zero_grad()

        surface_prediction = model(surface_data)

        loss = criterion(surface_data, surface_prediction)

        loss.backward()
        optimizer.step()
        train_batch_loss += loss.item()

    train_batch_loss = train_batch_loss / len(train_dataset)

    return train_batch_loss

#################################################################################################################
def validate_CNN_AE(validation_data,
                    validation_path,
                    model,
                    batch_size,
                    criterion,
                    ):

    model.eval()
    validation_loss = 0.0

    with torch.no_grad():

        validation_dataset = np.load(os.path.join(validation_path, validation_data))
        validation_dataset = validation_dataset.astype(np.float32)
        validation_loader = DataLoader(validation_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)

        prediction_data_all = []
        for surface_data in validation_loader:

            surface_data = surface_data.to(device)
            prediction = model(surface_data)

            prediction_data_all.append(prediction)
            prediction_all = torch.cat(prediction_data_all, dim=0)

            loss = criterion(prediction, surface_data)
            validation_loss += loss.item()

        # save all predicted data to file
        prediction_all = prediction_all.detach().cpu().numpy()

        # calculate validation loss
        validation_loss = validation_loss / len(validation_dataset)

        return validation_loss, prediction_all

#################################################################################################################
# Train Model 
#################################################################################################################

if settings_file.train_model == True:

    # initialize autoencoder
    # call autoencoder
    encoder = ConvEncoder(settings_file.surf_resolution, settings_file.input_channels, settings_file.latent_size, batchnorm=True, variational=True)
    decoder = ConvDecoder(settings_file.surf_resolution, settings_file.output_channels, settings_file.latent_size, batchnorm=True, squash_output=False)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)

    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=settings_file.learning_rate)
    
    # define lists for training and validation losses 
    train_losses = []
    valid_losses = []

    folder_path = os.path.join(settings_file.results_AE, settings_file.save_folder_CNN_AE)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    for epoch in range(settings_file.epochs):
        print(f"Epoch {epoch+1} of {settings_file.epochs}")

        # train model
        train_batch_loss = train_CNN_AE(train_data=settings_file.train_data,
                                        train_path=settings_file.train_path,
                                        model=autoencoder,
                                        optimizer=optimizer,
                                        batch_size=settings_file.batch_size,
                                        criterion=criterion,
                                        device=device)

        print(f"Training Loss: {train_batch_loss}")

        # validate model during training
        validation_batch_loss, prediction = validate_CNN_AE(validation_data=settings_file.validation_data,
                                                validation_path=settings_file.validation_path,
                                                model=autoencoder,
                                                batch_size=settings_file.batch_size,
                                                criterion=criterion)

        print(f"Validation Loss: {validation_batch_loss}")

        train_losses.append(train_batch_loss)
        valid_losses.append(validation_batch_loss)

        # saving best model
        if train_losses[-1] < settings_file.min_loss:
            autoencoder.save(os.path.join(folder_path,'trained_CNN_AE'))

    # save prediction to file
    #prediction = prediction.detach().cpu().numpy()
    np.save(os.path.join(folder_path, 'predictions_validation.npy'), prediction)

    # save training and validation losses
    with open(os.path.join(folder_path, 'train_loss.txt'), 'w') as f:
        for item in train_losses:
            f.write("%s\n" % item)

    with open(os.path.join(folder_path, 'valid_loss.txt'), 'w') as f:
        for item in valid_losses:
            f.write("%s\n" % item)

#################################################################################################################
if settings_file.test_model == True:

    folder_path = os.path.join(settings_file.results_AE, settings_file.save_folder_CNN_AE)

    # load trained CNN-AE
    encoder = ConvEncoder(settings_file.surf_resolution, settings_file.input_channels, settings_file.latent_size, 
                          batchnorm=True)
    decoder = ConvDecoder(settings_file.surf_resolution, settings_file.output_channels, settings_file.latent_size,
                         batchnorm=True, squash_output=True)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.load(os.path.join(folder_path, 'trained_CNN_AE'))
    autoencoder.to(device)

    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=settings_file.learning_rate)


    # list for test losses
    test_loss = []
    for epoch in range(settings_file.epochs):
        print(f"Epoch {epoch+1} of {settings_file.epochs}")

        test_batch_loss, prediction = validate_CNN_AE(validation_data=settings_file.test_data,
                                                            validation_path=settings_file.test_path,
                                                            model=autoencoder,
                                                            batch_size=settings_file.batch_size,
                                                            criterion=criterion)

        test_loss.append(test_batch_loss)

        print(f"Test Loss: {test_batch_loss}")

    np.save(os.path.join(folder_path, 'predictions_test.npy'), prediction)

    with open(os.path.join(folder_path,'test_loss.txt' ), 'w') as f:
        for item in test_loss:
            f.write("%s\n" % item)

#################################################################################################################







