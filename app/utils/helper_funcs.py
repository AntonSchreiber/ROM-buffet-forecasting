# python file with often used helper functions
import os
from os.path import join


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