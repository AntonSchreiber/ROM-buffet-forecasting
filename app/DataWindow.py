import numpy as np
import torch as pt
from torch.utils.data import TensorDataset, DataLoader

class DataWindow:
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns = None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        # Dict with label columns and their indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        # Dict with all column names and their indices
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}


        # The total window size is the sum of all input snapshots and the number of predicted snapshots
        self.total_window_size = input_width + shift

        # Define a slice object for the input sequence and store the corresponding indices
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # Define a slice object for the label sequence and store the corresponding indices
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        """
        Splits the total data window into two separate ones for inputs and labels.

        Args:
            features (np.ndarray): The features to split.

        Returns:
            inputs (np.ndarray): The inputs.
            labels (np.ndarray): The labels.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # The shape will be [batch, time, features], only the time dimension is specified for the moment
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def make_dataset(self, data):
        data = pt.tensor(data, dtype=pt.float32)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)    

        return dataloader
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result

