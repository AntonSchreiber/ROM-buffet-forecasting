# python file with often used helper functions
import os
from os.path import join
from pathlib import Path
import torch as pt
from CNN_VAE.CNN_VAE import make_VAE_model
import utils.config as config
from utils.Scaler import MinMaxScaler_1_1


def delete_directory_contents(directory_path):
    """ Delete directory contents with given path """
    try:
        # Get a list of all files and subdirectories in the directory
        file_list = os.listdir(directory_path)

        # Loop through the list and remove each file and subdirectory
        for file_name in file_list:
            file_path = join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_directory_contents(file_path)  # Recursively delete subdirectories
                os.rmdir(file_path)  # Remove the empty subdirectory after its contents are deleted

        print(f"Successfully deleted all contents in {directory_path}.")
    except Exception as e:
        print(f"Error occurred while deleting contents in {directory_path}: {e}")


def find_target_index_in_dataset(nested_list, target_id):
    """ Inside the rolling data window, find the index of the inputs-targets-pair that is used to predict the snapshot with target_id"""
    assert nested_list[0][-1] <= target_id and nested_list[-1][-1] >= target_id, f"Target snapshot index {target_id} out of range, must be in range({nested_list[0][-1]}, {nested_list[-1][-1] + 1})"
    for index, inner_list in enumerate(nested_list):
        if inner_list[-1] == target_id:
            return index
    return -1


def shift_input_sequence(orig_seq, new_pred):
    """
    Perform sliding window operation on a given tensor.

    Parameters:
        orig_seq (torch.Tensor): The initial tensor of shape [batch_size, input_timesteps * latent_size].
        new_pred (torch.Tensor): The new tensor with latent_size elements to be appended.

    Returns:
        torch.Tensor: The tensor with the first timestep removed and the new_pred appended.
                     The resulting shape will be [batch_size, input_timesteps * latent_size].
    """
    # Get the number of latent dimensions
    latent_size = new_pred.shape[1]

    # Make a copy of the original tensor before performing in-place operations
    orig_seq_copy = orig_seq.clone()

    # Remove the first timestep (latent_size elements) and append the new prediction to the original sequence
    orig_seq_copy[:, :-latent_size] = orig_seq[:, latent_size:]
    orig_seq_copy[:, -latent_size:] = new_pred

    return orig_seq_copy


def reduce_datasets_SVD_multi(DATA_PATH: str, SVD_PATH: str, OUTPUT_PATH: str):
    # load datasets
    train_data, val_data, test_data = load_datasets_multi(DATA_PATH=DATA_PATH, DIM_REDUCTION="SVD")

    # load left singular vectors U
    U = pt.load(SVD_PATH)

    # reduce datasets
    print("Reducing datasets with Left Singular Vectors ...")
    train_red = pt.transpose(U[:,:config.SVD_rank], 0, 1) @ (train_data - train_data.mean(dim=1).unsqueeze(-1))
    val_red = pt.transpose(U[:,:config.SVD_rank], 0, 1) @ (val_data - val_data.mean(dim=1).unsqueeze(-1))
    test_red = pt.transpose(U[:,:config.SVD_rank], 0, 1) @ (test_data - test_data.mean(dim=1).unsqueeze(-1))

    return scale_datasets_multi(train_red, val_red, test_red, OUTPUT_PATH), U[:,:config.SVD_rank]


def reduce_datasets_VAE_multi(DATA_PATH: str, VAE_PATH: str, OUTPUT_PATH: str, device: str):
    # load datasets
    train_data, val_data, test_data = load_datasets_multi(DATA_PATH=DATA_PATH, DIM_REDUCTION="VAE")

    # load pre-trained autoencoder model
    autoencoder = make_VAE_model(n_latent=config.VAE_latent_size, device=device)
    autoencoder.load(VAE_PATH)
    autoencoder.eval()

    # encode datasets 
    print("Reducing datasets with Autoencoder ...")
    train_red = autoencoder.encode_dataset(train_data, device)
    val_red = autoencoder.encode_dataset(val_data, device)
    test_red = autoencoder.encode_dataset(test_data, device)

    return scale_datasets_multi(train_red, val_red, test_red, OUTPUT_PATH), autoencoder._decoder


def load_datasets_multi(DATA_PATH: str, DIM_REDUCTION: str):
    """ Load the pipeline datasets from the given directory """
    print("Loading datasets ... ")
    train_data = pt.load(join(DATA_PATH, f"{DIM_REDUCTION}_train.pt"))
    val_data = pt.load(join(DATA_PATH, f"{DIM_REDUCTION}_val.pt")) 
    test_data = pt.load(join(DATA_PATH, f"{DIM_REDUCTION}_test.pt"))
    print("     min and max train cp prior reduction:       ", train_data.min().item(), train_data.max().item(), "\n")

    return train_data, val_data, test_data


def scale_datasets_multi(train_red: pt.Tensor, val_red: pt.Tensor, test_red: pt.Tensor, OUTPUT_PATH: str):
    print("     Shape of reduced train data:                ", train_red.shape)
    print("     Shape of reduced val data:                  ", val_red.shape)
    print("     Shape of reduced test data:                 ", test_red.shape)
    print("     min and max train cp after reduction:       ", train_red.min().item(), train_red.max().item(), "\n")

    # scale data
    print("Scaling reduced data to [-1, 1] ... ")
    scaler = MinMaxScaler_1_1().fit(train_red)
    train_red, val_red, test_red = scaler.scale(train_red), scaler.scale(val_red), scaler.scale(test_red)
    print("     min and max train cp after scaling:         ", train_red.min().item(), train_red.max().item(), "\n")    

    print("Saving scaler for inference ... ")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    pt.save(scaler, join(OUTPUT_PATH, "scaler.pt"))

    return train_red, val_red, test_red


def reduce_datasets_SVD_single(DATA_PATH: str, SVD_PATH: str, OUTPUT_PATH: str):
    # load datasets
    train_data, test_data = load_datasets_single(DATA_PATH=DATA_PATH, DIM_REDUCTION="SVD")

    # load left singular vectors U
    U = pt.load(SVD_PATH)

    # reduce datasets
    print("Reducing datasets with Left Singular Vectors ...")
    train_red = pt.transpose(U[:,:config.SVD_rank], 0, 1) @ (train_data - train_data.mean(dim=1).unsqueeze(-1))
    test_red = pt.transpose(U[:,:config.SVD_rank], 0, 1) @ (test_data - test_data.mean(dim=1).unsqueeze(-1))

    return scale_datasets_single(train_red, test_red, OUTPUT_PATH), U[:,:config.SVD_rank]


def reduce_datasets_VAE_single(DATA_PATH: str, VAE_PATH: str, OUTPUT_PATH: str, device: str):
    # load datasets
    train_data, test_data = load_datasets_single(DATA_PATH=DATA_PATH, DIM_REDUCTION="VAE")

    # load pre-trained autoencoder model
    autoencoder = make_VAE_model(n_latent=config.VAE_latent_size, device=device)
    autoencoder.load(VAE_PATH)
    autoencoder.eval()

    # encode datasets 
    print("Reducing datasets with Autoencoder ...")
    train_red = autoencoder.encode_dataset(train_data, device)
    test_red = autoencoder.encode_dataset(test_data, device)

    return scale_datasets_single(train_red, test_red, OUTPUT_PATH), autoencoder._decoder


def load_datasets_single(DATA_PATH: str, DIM_REDUCTION: str):
    """ Load the pipeline datasets from the given directory """
    print("Loading datasets ... ")
    train_data = pt.load(join(DATA_PATH, f"{DIM_REDUCTION}_train.pt"))
    test_data = pt.load(join(DATA_PATH, f"{DIM_REDUCTION}_test.pt"))
    print("     min and max train cp prior reduction:       ", train_data.min().item(), train_data.max().item(), "\n")

    return train_data, test_data


def scale_datasets_single(train_red: pt.Tensor, test_red: pt.Tensor, OUTPUT_PATH: str):
    print("     Shape of reduced train data:                ", train_red.shape)
    print("     Shape of reduced test data:                 ", test_red.shape)
    print("     min and max train cp after reduction:       ", train_red.min().item(), train_red.max().item(), "\n")

    # scale data
    print("Scaling reduced data to [-1, 1] ... ")
    scaler = MinMaxScaler_1_1().fit(train_red)
    train_red, test_red = scaler.scale(train_red), scaler.scale(test_red)
    print("     min and max train cp after scaling:         ", train_red.min().item(), train_red.max().item(), "\n")    

    print("Saving scaler for inference ... ")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    pt.save(scaler, join(OUTPUT_PATH, "scaler.pt"))

    return train_red, test_red


if __name__ == '__main__':
    orig_seq = pt.rand(2, 4)
    print("Input Sequence:      \n", orig_seq)

    for step in range(4):
        # shift input sequence by one: add last prediction while discarding first input
        if step != 0:
            orig_seq = shift_input_sequence(orig_seq=orig_seq, new_pred=new_pred)
            print("Input Sequence:      \n", orig_seq)
        # new prediction is sum of all previous elements
        new_pred = orig_seq.sum(dim=1, keepdim=True)
        print("Prediction:      \n", new_pred)