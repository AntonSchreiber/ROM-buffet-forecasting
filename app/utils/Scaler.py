class StandardScaler(object):
    """Class to scale/re-scale data to the mean and standard deviation (Standardization)"""
    def __init__(self) -> None:
        self.mean = None
        self.std = None
        self.initialized = False

    def fit(self, data):
        self.mean = data.mean().item()
        self.std = data.std().item()
        self.initialized = True
        return self

    def scale(self, data):
        assert self.initialized
        return (data - self.mean) / self.std
    
    def rescale(self, data_standardized):
        assert self.initialized
        return (data_standardized * self.std) + self.mean
    


class MinMaxScaler_0_1(object):
    """Class to scale/re-scale data to the range [0, 1] and back.
    Slightly modified from: https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/ml_intro.ipynb
    """
    def __init__(self):
        self.min = None
        self.max = None
        self.initialized = False

    def fit(self, data):
        self.min = data.min().item()
        self.max = data.max().item()
        self.initialized = True
        return self

    def scale(self, data):
        assert self.initialized
        return (data - self.min) / (self.max - self.min)

    def rescale(self, data_norm):
        assert self.initialized
        return data_norm * (self.max - self.min) + self.min
    



class MinMaxScaler_1_1(object):
    """Class to scale/re-scale data to the range [-1, 1] and back.
    Slightly modified from: https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/ml_intro.ipynb
    """
    def __init__(self):
        self.min = None
        self.max = None
        self.initialized = False

    def fit(self, data):
        self.min = data.min().item()
        self.max = data.max().item()
        self.initialized = True
        return self

    def scale(self, data):
        assert self.initialized
        data_norm = (data - self.min) / (self.max - self.min)
        return 2.0*data_norm - 1.0

    def rescale(self, data_norm):
        assert self.initialized
        data = (data_norm + 1.0) * 0.5
        return data * (self.max - self.min) + self.min



