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


class DataWindow():
    # class to create data windows from time series datasets
    def __init__(self, train: pt.Tensor, val: pt.Tensor=pt.Tensor(), test: pt.Tensor=pt.Tensor(), 
                 input_width: int=5, pred_horizon: int=1, batch_size: int=32) -> None:
        self.train = train
        self.val = val
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
    
    def rolling_window(self, dataset_length):
        # computes the rolling window with indices over full dataset length
        rolling_window = pt.arange(0, dataset_length).unfold(dimension=0, size=self.total_window_size, step=self.pred_horizon)

        # with slice objects, split each window into input and prediction sequence
        input_idx = pt.stack([sequence[self.input_slice] for sequence in rolling_window])
        pred_idx = pt.stack([sequence[self.pred_slice] for sequence in rolling_window])

        return input_idx, pred_idx
    
    def make_dataset(self, data):
        input_seq = []
        pred_seq = []
        for input_idx, pred_idx in zip(*self.rolling_window(dataset_length=data.shape[1])):
            input_seq.append(data[:, input_idx])
            pred_seq.append(data[:, pred_idx])

        input_seq = pt.stack(input_seq).unsqueeze(1)
        pred_seq = pt.stack(pred_seq).unsqueeze(1)
        # print(input_seqs.shape)
        # print(targets.shape)
        dataset = TensorDataset(input_seq, pred_seq)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return dataset

    @property
    def train_dataset(self):
        return self.make_dataset(self.train)
    
    @property
    def val_dataset(self):
        return self.make_dataset(self.val)
    


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
    print("Prediction starts at index:      ", self.pred_start)
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