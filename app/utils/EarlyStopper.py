import numpy as np

class EarlyStopper:
    def __init__(self, patience=5, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None

    def early_stop(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False
        if (self.mode == 'min' and current_loss < self.best_loss - self.delta) or \
           (self.mode == 'max' and current_loss > self.best_loss + self.delta):
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

