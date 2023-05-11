from modules.preprocessing import get_coords, get_cp_and_cases, reshape_data, to_dataframe, split_scale_save
from modules import DataWindow

if __name__ == "__main__":
    # Out-of-Loop Pre-Processing to assemble the global dataset
    x, y = get_coords()
    cp, cases = get_cp_and_cases()
    
    # Reshape all of the data into one tensor
    data = reshape_data(x, y, cp, cases)

    # Converting numpy array to pandas dataframe
    df = to_dataframe(data)
    
    # Split, scale and save the dataset into training, validation and testing subsets
    df_train, df_val, df_test = split_scale_save(df, train_size = 0.7, val_size = 0.2, test_size = 0.1)

    #Create DataWindow object
    test_window = DataWindow(input_width = 40, label_width = 40, shift = 1, train_df = df_train, val_df = df_val, test_df = df_test, label_columns = ["cp"])