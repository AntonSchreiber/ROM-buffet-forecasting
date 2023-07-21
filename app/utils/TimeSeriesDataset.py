import sys
import os
from os.path import join
parent_dir = os.path.abspath(join(os.getcwd(), os.pardir))
app_dir = join(parent_dir, "app")
if app_dir not in sys.path:
      sys.path.append(app_dir)

import numpy as np
import torch as pt
from utils.AutoencoderDataset import AutoencoderDataset
from torch.utils.data import Dataset, TensorDataset, DataLoader
from math import ceil


class TimeSeriesDatasetVAE(Dataset):
    def __init__(self, input_seq, targets) -> None:
        self.input_seq = input_seq
        self.targets = targets

    def __getitem__(self, index: int):
        pass


class DataWindowVAE():
    # class to create data windows from time series datasets
    def __init__(self, train, test, input_width: int, pred_horizon:int, shift: int, batch_size) -> None:
        self.train = train
        self.test = test

        self.batch_size = batch_size
        self.input_width = input_width
        self.pred_horizon = pred_horizon
        self.shift = shift

        # The total window size is the length of the input sequence + the number of predicted values
        self.total_window_size = input_width + shift
        self.window_offset = input_width + pred_horizon

        # Define a slice object for the input sequence and store the corresponding indices
        # The input slice consists of a sequence used to make a prediction and a sequence to compare the prediction to
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # Define a slice object for the prediction sequence and store the corresponding indices
        self.pred_start = self.input_width
        self.pred_slice = slice(self.pred_start, None)
        self.pred_indices = np.arange(self.input_width+ self.pred_horizon)[self.pred_slice]

    
    def split_to_input_pred(self, features):
        inputs = features[:, :, self.input_slice]
        preds = features[:, :, self.pred_slice]

        # the shape is [height, width, time]
        return inputs, preds
    
    def rolling_window(self, offset, dataset_length):
        # TODO could you start the new window earlier?
        last_idx = offset + self.input_width + self.pred_horizon
        return pt.arange(offset, last_idx if last_idx < dataset_length else dataset_length).unfold(0, self.input_width if last_idx < dataset_length else dataset_length - offset, self.shift)
    
    def make_dataset(self, data):
        input_seqs = []
        targets = []
        for i in range(data.shape[2] // self.window_offset):
            for n, input_seq in enumerate(self.rolling_window(offset=i*self.window_offset, dataset_length=data.shape[2])):
                # print(input_seq)
                input_seqs.append(data[:, :, input_seq])
                target_idx = pt.arange(self.input_width + n + i*self.window_offset, self.input_width + n + i*self.window_offset + self.shift)
                #print(target_idx)
                targets.append(data[:, :, target_idx])

        input_seqs = pt.stack(input_seqs).unsqueeze(1)
        targets = pt.stack(targets).unsqueeze(1)
        # print(input_seqs.shape)
        # print(targets.shape)
        dataset = TensorDataset(input_seqs, targets)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return dataset

    @property
    def train_dataset(self):
        return self.make_dataset(self.train)
    
    @property
    def test_dataset(self):
        return self.make_dataset(self.test)
    


if __name__ == "__main__":
    self = DataWindowVAE(
        train = pt.arange(256 * 128 * 500).reshape(256, 128, 500),
        test="",
        input_width=10,
        pred_horizon=5,
        shift=1,
        batch_size=32
    )

    print("Shift of the window:             ", self.shift)
    print("Total window size:               ", self.total_window_size)
    print("Input width:                     ", self.input_width)
    print("input indices:                   ", self.input_indices)
    print("Prediction starts at index:      ", self.pred_start)
    print("Prediction indices:              ", self.pred_indices)
    print("Prediction horizon:              ", self.pred_horizon)

    dataset = self.train_dataset
    print(len(dataset))
    print(dataset[0][1].shape)