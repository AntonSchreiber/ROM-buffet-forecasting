class StandardScaler(object):
    """Class to scale/re-scale data to the mean and standard deviation (Standardization)"""
    def __init__(self) -> None:
        self.mean = None
        self.std = None
        self.initialized = False

    def fit(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)
        self.initialized = True
        return self

    def scale(self, data):
        assert self.initialized
        return (data - self.mean) / self.std
    
    def rescale(self, data_standardized):
        assert self.initialized
        return (data_standardized * self.std) + self.mean