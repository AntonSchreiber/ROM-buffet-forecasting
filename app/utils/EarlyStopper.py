class EarlyStopper:
    def __init__(self, patience: int=5, delta: float=0, mode: str='min'):
        """Create EarlyStopper instance.
        Can be used to stop the training if the validation loss does not increase for a certain number of epochs.

        Args:
            patience (int, optional):  Number of epochs to wait before early stopping if no decrease in loss. Defaults to 5.
            delta (float, optional): Allowed deviation. Defaults to 0.
            mode (str, optional): Minimize or maximize the objective? Defaults to 'min'.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None

    def early_stop(self, current_loss):
        if self.best_loss is None:
            # assign value to best_loss after first epoch
            self.best_loss = current_loss
            return False
        
        if (self.mode == 'min' and current_loss < self.best_loss - self.delta) or \
           (self.mode == 'max' and current_loss > self.best_loss + self.delta):
            # assign a new best loss and reset the counter
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # if the number of epochs without progress exceeds the patience, stop training
                return True
        return False

