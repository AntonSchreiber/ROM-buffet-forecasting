import unittest
from utils.DataWindow import DataWindow
import torch as pt

# create data tensor
num_timesteps = 7
latent_space_size = 3

values = pt.arange(1, num_timesteps + 1).unsqueeze(0)
data_tensor = values.repeat(latent_space_size, 1)

# print("Data tensor: \n", data_tensor)

# looks like this:
# tensor([[1, 2, 3, 4, 5, 6, 7, 8],
#         [1, 2, 3, 4, 5, 6, 7, 8],
#         [1, 2, 3, 4, 5, 6, 7, 8]])

# set and compute DataWindow Variables
input_width = 4
pred_horizon = 1
num_windows = (num_timesteps - input_width) // pred_horizon



class TestDataWindow(unittest.TestCase):
    def test_rolling_window(self):
        # initialize DataWindow object and call rolling_window method
        data_window = DataWindow(data_tensor, input_width=input_width, pred_horizon=pred_horizon)
        input_idx, pred_idx = data_window.rolling_window(dataset_length=num_timesteps)
        
        # check dimensions of the windows
        self.assertEqual(input_idx.shape, pt.Size([num_windows, input_width]))
        self.assertEqual(pred_idx.shape, pt.Size([num_windows, pred_horizon]))
        print("With {} timesteps, an input width of {} and a prediction horizon of {}, there are {} possible data windows \n".format(num_timesteps, input_width, pred_horizon, num_windows))
        for input_id, pred_id in zip(input_idx, pred_idx):
            print("Input idx: ", input_id, "   Pred idx: ", pred_id)
    
    def test_make_dataset(self):
        # initialize DataWindow object and call rolling_window method
        data_window = DataWindow(data_tensor, input_width=input_width, pred_horizon=pred_horizon)
        train = data_window.train_dataset
        self.assertEqual(len(train), num_windows)

        # rebuild the dataset for the very first feature-label pair
        seq1, pred1 = train[0]

        print("The first feature-label pair is: \n")
        print(seq1) 
        print(pred1, "\n")

if __name__ == '__main__':
    unittest.main()