import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from dataloader import train_test_val_split, get_dataset_statistics
from models import SpikingNetwork, TwoLayerSNN, ThreeLayerSNN
from utils.encoding import Rate_Encoder, Current_Encoder

# get 5-fold validation model performance

def get_binary_metrics(model, test_loader, encoder, threshold, device='cpu'):
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:

            if encoder is not None:
                data = encoder.encode(data)

            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)
    
            y_true.append(target.numpy())
            y_pred.append(output.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = (np.concatenate(y_pred, axis=0)  > threshold).astype(int)
    
    accuracy = (y_true == y_pred).mean()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    return accuracy, sensitivity, specificity, ppv, npv


if __name__ == '__main__':


    import argparse
    import yaml

    # load in data from relavent config files
    parser = argparse.ArgumentParser(description="train a model using specific training configurations")
    parser.add_argument("train_config", help="name of config file to train model")

    args = parser.parse_args()

    # read in the paths to where to read from/to data
    with open('configs/paths.yaml', 'r') as file:
        path_configs = yaml.safe_load(file)
    
    # read in the instructions for how to train model
    with open(f'configs/model_configs/{args.train_config}', 'r') as file:
        train_configs = yaml.safe_load(file)

    seed = train_configs['seed']
    device = train_configs['device']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Create Datasets/DataLoaders --- #    
    binary = train_configs['binary']

    dataset_configs = {
        "data_file": os.path.join(path_configs['data_folder'], train_configs['dataset'], f"{train_configs['dataset']}.hdf5"),
        "metadata":  os.path.join(path_configs['data_folder'], train_configs['dataset'], f"{train_configs['dataset']}.csv"),
        "binary": binary,
        "random_state": 0,
        **train_configs['dataset_configs']
    }

    logging_configs={
        'model_name': train_configs['model_name'],
        'weight_folder': path_configs['weight_folder'],
        'log_folder': path_configs['log_folder'],
        'log_steps': train_configs['log_steps']
    }

    train_batch_size = train_configs['train_batch_size']
    eval_batch_size = train_configs['eval_batch_size']

    train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)
    train_mean, train_std = get_dataset_statistics(train_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)

     # --- Model Configs --- #
    model_configs = train_configs['model_configs']
    if model_configs['neuron_options']['spike_grad'] == 'atan':
        model_configs['neuron_options']['spike_grad'] = surrogate.atan(alpha=2)

    # --- Train the Model --- #
    if train_configs['model_class'] == 'TwoLayerSNN':
        model = TwoLayerSNN(**model_configs)
    else:
        model = SpikingNetwork(**model_configs)

    if train_configs['encoder'] == 'rate':
        encoder = Rate_Encoder(**train_configs['encoder_args'])
    else:
        encoder = Current_Encoder(train_mean, train_std)

    model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])

    threshold = None
    if binary:
        threshold = checkpoint['threshold']


    accuracy, sensitivity, specificity, ppv, npv = get_binary_metrics(model, test_loader, encoder, threshold)

    print(f"--- {train_configs['model_name']} Performance ---")
    print(f'Accuracy: {accuracy}')
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'PPV: {ppv}')
    print(f'NPV: {npv}')