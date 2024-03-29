"""
These training loop functions are modified versions of the functions provided in chapter 5 of 'Inside Deep Learning' by Edward Raff;
    refer to:   https://github.com/EdwardRaff/Inside-Deep-Learning/blob/main/idlmam.py (original)
    and:        https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/ml_intro.ipynb (already modified)
"""

import sys
import os
from os.path import join
from glob import glob
from typing import List, Tuple
from shutil import copy
from time import time
from collections import defaultdict
import torch as pt
import pandas as pd

from utils.EarlyStopper import EarlyStopper
from utils.helper_funcs import shift_input_sequence


def run_epoch_VAE(
    model: pt.nn.Module,
    optimizer: pt.optim.Optimizer,
    data_loader: pt.utils.data.DataLoader,
    loss_func: pt.nn.Module,
    device: str,
    results: dict,
    score_funcs: dict,
    prefix: str
    ) -> float:
    """Perform one epoch with optimizing steps on an Convolutional-Variational-Autoencoder model"""

    # keeping track of loss, predictions, and time
    running_loss, labels_true, labels_pred = [], [], []
    start_time = time()

    # loop over all batches
    for surface_data in data_loader:
        surface_data = surface_data.to(device)

        pred = model(surface_data)
        loss = loss_func(surface_data, pred)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss.append(loss.item())

        # the dataset might get shuffled in the next loop
        if len(score_funcs) > 0:
            labels_true.extend(surface_data.detach().cpu().tolist())
            labels_pred.extend(pred.detach().cpu().tolist())

    # keep track of performance
    results[f"{prefix}_loss"].append(sum(running_loss) / len(running_loss))
    for name, func in score_funcs.items():
        results[f"{prefix}_{name}"].append(func(labels_true, labels_pred))

    return time() - start_time


def train_VAE(
    model: pt.nn.Module,
    loss_func: pt.nn.Module,
    train_loader: pt.utils.data.DataLoader,
    val_loader: pt.utils.data.DataLoader = None,
    test_loader: pt.utils.data.DataLoader = None,
    score_funcs: dict = {},
    epochs: int = 100,
    device: str = "cpu",
    checkpoint_file: str = None,
    log_all: bool = False,
    lr_schedule: pt.optim.lr_scheduler._LRScheduler = None,
    optimizer: pt.optim.Optimizer = None,
    early_stopper: EarlyStopper = None
    ) -> pd.DataFrame:
    """Perform the training of a Convoutional-Variational-Autoencoder model"""

    # dictionary for keeping track of training performance
    results = defaultdict(list)
    best_loss = float("inf")
    ref_loss = "train_loss" if val_loader is None else "val_loss"

    # use AdamW as default optimizer if none specified
    delete_optimizer = False
    if optimizer is None:
        optimizer = pt.optim.AdamW(model.parameters())
        delete_optimizer = True

    total_train_time = 0.0
    model.to(device)
    for e in range(epochs):
        # model update
        model = model.train()
        total_train_time += run_epoch_VAE(
            model, optimizer, train_loader, loss_func, device,
            results, score_funcs, prefix="train"
        )
        results["epoch"].append(e)
        results["total_time"].append(total_train_time)
        message = f"Training loss: {results['train_loss'][-1]:2.6e}"


        # validation dataset
        if val_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_VAE(
                    model, optimizer, val_loader, loss_func, device,
                    results, score_funcs, prefix="val"
                )
            message += f"; Validation loss: {results['val_loss'][-1]:2.6e}"

        # update of learning rate
        if lr_schedule is not None:
            if isinstance(lr_schedule, pt.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val_loss"][-1])
            else:
                lr_schedule.step()

        # test dataset
        if test_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_VAE(
                    model, optimizer, test_loader, loss_func, device,
                    results, score_funcs, prefix="test"
                )
            message += f"; Test loss: {results['test_loss'][-1]:2.6e}"

        # save checkpoint
        if checkpoint_file is not None:
            suffix = f"_epoch_{e}" if log_all else ""
            pt.save(
                {
                    "epoch" : e,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results" : results
                }, checkpoint_file + suffix
            )
            latest_loss = results[ref_loss][-1]
            if latest_loss < best_loss:
                best_loss = latest_loss
                copy(checkpoint_file + suffix, checkpoint_file + "_best")

        # early stop condition check after each epoch
        if early_stopper and early_stopper.early_stop(results['val_loss'][-1]):
            print("\nEarly stopping at epoch {}! Validation loss did not improve for {} epochs.".format(e+1, early_stopper.patience))
            break       

        print(
            "\r", f"Epoch {e+1:4d}/{epochs} - " + message, end=""
        )

    # if the optimizer was created in the training loop,
    # delete if to avoid unwanted side effects
    if delete_optimizer:
        del optimizer

    return pd.DataFrame.from_dict(results)


def run_epoch_FC(
    model: pt.nn.Module,
    optimizer: pt.optim.Optimizer,
    data_loader: pt.utils.data.DataLoader,
    loss_func: pt.nn.Module,
    device: str,
    results: dict,
    score_funcs: dict,
    prefix: str
    ) -> float:
    """Perform one epoch with optimizing steps on an autoregressive model"""

    # keeping track of loss, predictions, and time
    running_loss, labels_true, labels_pred = [], [], []
    start_time = time()

    # loop over all batches
    for inputs, targets in data_loader:
        inputs = pt.transpose(inputs, 1, 2).flatten(1, 2).to(device)
        targets = targets.to(device)
        preds = []

        for step in range(0, targets.shape[2]):
            if step != 0:
                inputs = shift_input_sequence(orig_seq=inputs, new_pred=preds[-1])
            preds.append(model(inputs))

        preds = pt.stack(preds, dim=2)

        loss = loss_func(targets, preds)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss.append(loss.item())

        # the dataset might get shuffled in the next loop
        if len(score_funcs) > 0:
            labels_true.extend(targets.detach().cpu().tolist())
            labels_pred.extend(preds.detach().cpu().tolist())

    # keep track of performance
    results[f"{prefix}_loss"].append(sum(running_loss) / len(running_loss))
    for name, func in score_funcs.items():
        results[f"{prefix}_{name}"].append(func(labels_true, labels_pred))

    return time() - start_time


def train_FC(
    model: pt.nn.Module,
    loss_func: pt.nn.Module,
    train_loader: pt.utils.data.DataLoader,
    val_loader: pt.utils.data.DataLoader = None,
    test_loader: pt.utils.data.DataLoader = None,
    score_funcs: dict = {},
    epochs: int = 100,
    device: str = "cpu",
    checkpoint_file: str = None,
    log_all: bool = False,
    lr_schedule: pt.optim.lr_scheduler._LRScheduler = None,
    optimizer: pt.optim.Optimizer = None,
    early_stopper: EarlyStopper = None
    ) -> pd.DataFrame:
    """Perform the autoregressive training of a model"""

    # dictionary for keeping track of training performance
    results = defaultdict(list)
    best_loss = float("inf")
    ref_loss = "train_loss" if val_loader is None else "val_loss"

    # use AdamW as default optimizer if none specified
    delete_optimizer = False
    if optimizer is None:
        optimizer = pt.optim.AdamW(model.parameters())
        delete_optimizer = True

    total_train_time = 0.0
    model.to(device)
    for e in range(epochs):
        # model update
        model = model.train()
        total_train_time += run_epoch_FC(
            model, optimizer, train_loader, loss_func, device,
            results, score_funcs, prefix="train"
        )
        results["epoch"].append(e)
        results["total_time"].append(total_train_time)
        message = f"Training loss: {results['train_loss'][-1]:2.6e}"


        # validation dataset
        if val_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_FC(
                    model, optimizer, val_loader, loss_func, device,
                    results, score_funcs, prefix="val"
                )
            message += f"; Validation loss: {results['val_loss'][-1]:2.6e}"

        # update of learning rate
        if lr_schedule is not None:
            if isinstance(lr_schedule, pt.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val_loss"][-1])
            else:
                lr_schedule.step()

        # test dataset
        if test_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_FC(
                    model, optimizer, test_loader, loss_func, device,
                    results, score_funcs, prefix="test"
                )
            message += f"; Test loss: {results['test_loss'][-1]:2.6e}"

        # save checkpoint
        if checkpoint_file is not None:
            suffix = f"_epoch_{e}" if log_all else ""
            pt.save(
                {
                    "epoch" : e,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results" : results
                }, checkpoint_file + suffix
            )
            latest_loss = results[ref_loss][-1]
            if latest_loss < best_loss:
                best_loss = latest_loss
                copy(checkpoint_file + suffix, checkpoint_file + "_best")

        # early stop condition check after each epoch
        if early_stopper and early_stopper.early_stop(results['val_loss'][-1]):
            print("\nEarly stopping at epoch {}! Validation loss did not improve for {} epochs.".format(e+1, early_stopper.patience))
            break       

        print(
            "\r", f"Epoch {e+1:4d}/{epochs} - " + message, end=""
        )

    # if the optimizer was created in the training loop,
    # delete if to avoid unwanted side effects
    if delete_optimizer:
        del optimizer

    return pd.DataFrame.from_dict(results)


def run_epoch_LSTM(
    model: pt.nn.Module,
    optimizer: pt.optim.Optimizer,
    data_loader: pt.utils.data.DataLoader,
    loss_func: pt.nn.Module,
    device: str,
    results: dict,
    score_funcs: dict,
    prefix: str
    ) -> float:
    """Perform one epoch with optimizing steps on an autoregressively trained model"""

    # keeping track of loss, predictions, and time
    running_loss, labels_true, labels_pred = [], [], []
    start_time = time()

    # loop over all batches
    for inputs, targets in data_loader:
        inputs = inputs.permute(0,2,1).to(device)
        targets = targets.permute(0,2,1).to(device)
        
        # init step 
        preds = model(inputs, targets.shape[1])
        loss = loss_func(targets, preds)  

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss.append(loss.item())

        # the dataset might get shuffled in the next loop
        if len(score_funcs) > 0:
            labels_true.extend(targets.detach().cpu().tolist())
            labels_pred.extend(preds.detach().cpu().tolist())

    # keep track of performance
    results[f"{prefix}_loss"].append(sum(running_loss) / len(running_loss))
    for name, func in score_funcs.items():
        results[f"{prefix}_{name}"].append(func(labels_true, labels_pred))

    return time() - start_time


def train_LSTM(
    model: pt.nn.Module,
    loss_func: pt.nn.Module,
    train_loader: pt.utils.data.DataLoader,
    val_loader: pt.utils.data.DataLoader = None,
    test_loader: pt.utils.data.DataLoader = None,
    score_funcs: dict = {},
    epochs: int = 100,
    device: str = "cpu",
    checkpoint_file: str = None,
    log_all: bool = False,
    lr_schedule: pt.optim.lr_scheduler._LRScheduler = None,
    optimizer: pt.optim.Optimizer = None,
    early_stopper: EarlyStopper = None
    ) -> pd.DataFrame:
    """Perform the autoregressive training of a model"""

    # dictionary for keeping track of training performance
    results = defaultdict(list)
    best_loss = float("inf")
    ref_loss = "train_loss" if val_loader is None else "val_loss"

    # use AdamW as default optimizer if none specified
    delete_optimizer = False
    if optimizer is None:
        optimizer = pt.optim.AdamW(model.parameters())
        delete_optimizer = True

    total_train_time = 0.0
    model.to(device)
    for e in range(epochs):
        # model update
        model = model.train()
        total_train_time += run_epoch_LSTM(
            model, optimizer, train_loader, loss_func, device,
            results, score_funcs, prefix="train"
        )
        results["epoch"].append(e)
        results["total_time"].append(total_train_time)
        message = f"Training loss: {results['train_loss'][-1]:2.6e}"


        # validation dataset
        if val_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_LSTM(
                    model, optimizer, val_loader, loss_func, device,
                    results, score_funcs, prefix="val"
                )
            message += f"; Validation loss: {results['val_loss'][-1]:2.6e}"

        # update of learning rate
        if lr_schedule is not None:
            if isinstance(lr_schedule, pt.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val_loss"][-1])
            else:
                lr_schedule.step()

        # test dataset
        if test_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_LSTM(
                    model, optimizer, test_loader, loss_func, device,
                    results, score_funcs, prefix="test"
                )
            message += f"; Test loss: {results['test_loss'][-1]:2.6e}"

        # save checkpoint
        if checkpoint_file is not None:
            suffix = f"_epoch_{e}" if log_all else ""
            pt.save(
                {
                    "epoch" : e,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results" : results
                }, checkpoint_file + suffix
            )
            latest_loss = results[ref_loss][-1]
            if latest_loss < best_loss:
                best_loss = latest_loss
                copy(checkpoint_file + suffix, checkpoint_file + "_best")

        # early stop condition check after each epoch
        if early_stopper and early_stopper.early_stop(results['val_loss'][-1]):
            print("\nEarly stopping at epoch {}! Validation loss did not improve for {} epochs.".format(e+1, early_stopper.patience))
            break       

        print(
            "\r", f"Epoch {e+1:4d}/{epochs} - " + message, end=""
        )

    # if the optimizer was created in the training loop,
    # delete if to avoid unwanted side effects
    if delete_optimizer:
        del optimizer

    return pd.DataFrame.from_dict(results)


def run_epoch_end_to_end(
    model: pt.nn.Module,
    optimizer: pt.optim.Optimizer,
    data_loader: pt.utils.data.DataLoader,
    loss_func: pt.nn.Module,
    device: str,
    results: dict,
    score_funcs: dict,
    prefix: str
    ) -> float:
    """Perform one epoch with optimizing steps on an autoregressively trained model"""

    # keeping track of loss, predictions, and time
    running_loss, labels_true, labels_pred = [], [], []
    start_time = time()

    # loop over all batches
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        preds = model(inputs, targets.shape[4])
        loss = loss_func(targets, preds)  

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss.append(loss.item())

        # the dataset might get shuffled in the next loop
        if len(score_funcs) > 0:
            labels_true.extend(targets.detach().cpu().tolist())
            labels_pred.extend(preds.detach().cpu().tolist())

    # keep track of performance
    results[f"{prefix}_loss"].append(sum(running_loss) / len(running_loss))
    for name, func in score_funcs.items():
        results[f"{prefix}_{name}"].append(func(labels_true, labels_pred))

    return time() - start_time


def train_end_to_end(
    model: pt.nn.Module,
    loss_func: pt.nn.Module,
    train_loader: pt.utils.data.DataLoader,
    val_loader: pt.utils.data.DataLoader = None,
    test_loader: pt.utils.data.DataLoader = None,
    score_funcs: dict = {},
    epochs: int = 100,
    device: str = "cpu",
    checkpoint_file: str = None,
    log_all: bool = False,
    lr_schedule: pt.optim.lr_scheduler._LRScheduler = None,
    optimizer: pt.optim.Optimizer = None,
    early_stopper: EarlyStopper = None
    ) -> pd.DataFrame:
    """Perform the autoregressive training of a model"""

    # dictionary for keeping track of training performance
    results = defaultdict(list)
    best_loss = float("inf")
    ref_loss = "train_loss" if val_loader is None else "val_loss"

    # use AdamW as default optimizer if none specified
    delete_optimizer = False
    if optimizer is None:
        optimizer = pt.optim.AdamW(model.parameters())
        delete_optimizer = True

    total_train_time = 0.0
    model.to(device)
    for e in range(epochs):
        # model update
        model = model.train()
        total_train_time += run_epoch_end_to_end(
            model, optimizer, train_loader, loss_func, device,
            results, score_funcs, prefix="train"
        )
        results["epoch"].append(e)
        results["total_time"].append(total_train_time)
        message = f"Training loss: {results['train_loss'][-1]:2.6e}"


        # validation dataset
        if val_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_end_to_end(
                    model, optimizer, val_loader, loss_func, device,
                    results, score_funcs, prefix="val"
                )
            message += f"; Validation loss: {results['val_loss'][-1]:2.6e}"

        # update of learning rate
        if lr_schedule is not None:
            if isinstance(lr_schedule, pt.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val_loss"][-1])
            else:
                lr_schedule.step()

        # test dataset
        if test_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch_end_to_end(
                    model, optimizer, test_loader, loss_func, device,
                    results, score_funcs, prefix="test"
                )
            message += f"; Test loss: {results['test_loss'][-1]:2.6e}"

        # save checkpoint
        if checkpoint_file is not None:
            suffix = f"_epoch_{e}" if log_all else ""
            pt.save(
                {
                    "epoch" : e,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results" : results
                }, checkpoint_file + suffix
            )
            latest_loss = results[ref_loss][-1]
            if latest_loss < best_loss:
                best_loss = latest_loss
                copy(checkpoint_file + suffix, checkpoint_file + "_best")

        # early stop condition check after each epoch
        if early_stopper and early_stopper.early_stop(results['val_loss'][-1]):
            print("\nEarly stopping at epoch {}! Validation loss did not improve for {} epochs.".format(e+1, early_stopper.patience))
            break     

        # Monitor memory usage
        memory_allocated = pt.cuda.memory_allocated() / 1024 ** 3
        memory_cached = pt.cuda.memory_reserved() / 1024 ** 3
        message += f"; Allocated: {memory_allocated:.2f} GB; Cached: {memory_cached:.2f} GB"
        print(
            "\r", f"Epoch {e+1:4d}/{epochs} - " + message, end=""
        )

    # if the optimizer was created in the training loop,
    # delete if to avoid unwanted side effects
    if delete_optimizer:
        del optimizer

    return pd.DataFrame.from_dict(results)