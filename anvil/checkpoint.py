"""
checkpoint.py

Module for loading checkpoints - ensuring a copy of model is made so that
we don't get unexpected results
"""
from copy import deepcopy

import torch.optim as optim


def loaded_checkpoint(checkpoint):
    if checkpoint is None:
        return None
    cp_loaded = checkpoint.load()
    return cp_loaded


def train_range(loaded_checkpoint, epochs):
    if loaded_checkpoint is not None:
        cp_epoch = loaded_checkpoint["epoch"]
        train_range = (cp_epoch, cp_epoch + epochs)
    else:
        train_range = (0, epochs)
    return train_range


def loaded_model(loaded_checkpoint, model):
    new_model = deepcopy(model)  # need to copy model so we don't get weird results
    if loaded_checkpoint is not None:
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    return new_model


def loaded_optimizer(loaded_model, loaded_checkpoint):
    # TODO: implement optimizer input
    new_optimizer = optim.Adadelta(loaded_model.parameters())
    if loaded_checkpoint is not None:
        new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return new_optimizer


def current_loss(loaded_checkpoint):
    if loaded_checkpoint is None:
        return None
    return loaded_checkpoint["loss"]
