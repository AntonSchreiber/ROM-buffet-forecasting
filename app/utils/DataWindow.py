import torch as pt
from torch.utils.data import Dataset, TensorDataset, DataLoader


class DataWindow():
    """Class to create DataWindows from a dataset in latent space with (n, timesteps) where n is the number of dimensions in the latent space

        The make_dataset function creates a rolling window that contains user-defined feature-label pairs to be fed through prediction models
    """
    def __init__(self, train: pt.Tensor, val: pt.Tensor=pt.Tensor(), test: pt.Tensor=pt.Tensor(), 
                 input_width: int=40, pred_horizon: int=1) -> None:
        self.train = train
        self.val = val
        self.test = test
        assert len(self.train.shape) == 2

        self.input_width = input_width
        self.pred_horizon = pred_horizon

        # The total window size is the length of the input sequence + the number of predicted values
        self.total_window_size = input_width + pred_horizon

        # Define slice objects for input and prediction sequence 
        self.input_slice = slice(0, input_width)
        self.pred_start = self.input_width
        self.pred_slice = slice(self.pred_start, None)
    
    def rolling_window(self, dataset_length: int):
        """ computes a rolling window with indices over full dataset length """
        assert dataset_length >= self.total_window_size, f"Dataset length must be >= total window size"

        # compute rolling window
        rolling_window = pt.arange(dataset_length).unfold(dimension=0, size=self.total_window_size, step=1)

        # with slice objects, split each window into input and prediction sequence
        input_idx = pt.stack([sequence[self.input_slice] for sequence in rolling_window])
        pred_idx = pt.stack([sequence[self.pred_slice] for sequence in rolling_window])

        return input_idx, pred_idx
    
    def make_dataset(self, data):
        input_seq = []
        target_seq = []
        # iterate over feature-label index pairs in rolling window and assign corresponding data pairs
        for input_idx, pred_idx in zip(*self.rolling_window(dataset_length=data.shape[1])):
            input_seq.append(data[:, input_idx])
            target_seq.append(data[:, pred_idx[-1]])        # only include last target since all other are not needed during training

        # stack to pt.Tensors
        input_seq = pt.stack(input_seq)
        target_seq = pt.stack(target_seq)

        # store all feature-label pairs in a Dataset object
        return TimeSeriesTensorDataset(input_seq, target_seq)            

    @property
    def train_dataset(self):
        return self.make_dataset(self.train)
    
    @property
    def val_dataset(self):
        return self.make_dataset(self.val)
    
    @property
    def test_dataset(self):
        return self.make_dataset(self.test)
    

class TimeSeriesTensorDataset(TensorDataset):
    def __init__(self, inputs, targets):
        super().__init__(inputs, targets)

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets