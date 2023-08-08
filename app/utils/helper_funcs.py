# python file with often used helper functions
import os
from os.path import join
import torch as pt


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


if __name__ == '__main__':
    latent_size = 1
    orig_seq = pt.rand(2, (4 * latent_size))
    new_pred = pt.rand(2, (1 * latent_size))
    print(orig_seq)
    print(new_pred)

    new_seq = shift_input_sequence(orig_seq, new_pred)
    print(new_seq)