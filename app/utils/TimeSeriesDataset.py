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


class DataWindow():
    # class to create data windows from time series datasets
    def __init__(self, train, test, input_width: int, pred_horizon:int, batch_size) -> None:
        self.train = train
        self.test = test

        self.batch_size = batch_size
        self.input_width = input_width
        self.pred_horizon = pred_horizon

        # The total window size is the length of the input sequence + the number of predicted values
        self.total_window_size = input_width + pred_horizon

        # Define a slice object for the input sequence 
        self.input_slice = slice(0, input_width)

        # Define a slice object for the prediction sequence
        self.pred_start = self.input_width
        self.pred_slice = slice(self.pred_start, None)

    def split_to_input_pred(self, features):
        inputs = features[:, :, self.input_slice]
        preds = features[:, :, self.pred_slice]

        # the shape is [height, width, time]
        return inputs, preds
    
    def rolling_window(self, dataset_length):
        # computes the rolling window with indices over full dataset length
        rolling_window = pt.arange(0, dataset_length).unfold(dimension=0, size=self.total_window_size, step=self.pred_horizon)

        # with slice objects, split each window into input and prediction sequence
        input_idx = pt.stack([sequence[self.input_slice] for sequence in rolling_window])
        pred_idx = pt.stack([sequence[self.pred_slice] for sequence in rolling_window])

        return input_idx, pred_idx
    
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
    self = DataWindow(
        train = pt.arange(256 * 128 * 500).reshape(256, 128, 500),
        test="",
        input_width=5,
        pred_horizon=1,
        batch_size=32
    )

    print("Total window size:               ", self.total_window_size)
    print("Input width:                     ", self.input_width)
    print("input indices:                   ", self.input_indices)
    print("Prediction starts at index:      ", self.pred_start)
    print("Prediction indices:              ", self.pred_indices)
    print("Prediction horizon:              ", self.pred_horizon)

    # dataset = self.train_dataset
    # print(len(dataset))
    # print(dataset[0][1].shape)


    dataset_length = 10

    rolling_window = pt.arange(0, dataset_length).unfold(dimension=0, size=self.total_window_size, step=self.pred_horizon)
    print(rolling_window)

    input_idx = pt.stack([sequence[self.input_slice] for sequence in rolling_window])
    pred_idx = pt.stack([sequence[self.pred_slice] for sequence in rolling_window])
    print(input_idx)
    print(pred_idx)