import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import random
import numpy as np
import pandas as pd

from dataloader import ECGWaveformDataset, sample_dataset_statistics
from models import SpikingNetwork, TwoLayerSNN, ThreeLayerSNN
from utils.encoding import Rate_Encoder, Current_Encoder

def train_model(
        model, 
        train_loader, 
        encoder, 
        num_epochs, 
        device='cpu',
        logging_configs={
            'model_name': None,
            'weight_folder': 'train_weights',
        }
    ):
    
    '''
    train a model
    model: model object
    train_loader: dataloader for training data
    val_loader: dataloader for validation data
    encoder: encoding strategy for spiking input
    optimizer: optimizer for training
    loss_fn: loss function for training
    num_epochs: number of training epoch
    val_steps: how often to evaluate model on validation set
    device: device to run model on
    logging_configs: information for saving weights and logging model output

    '''

    train_losses = []
    val_losses = []

    model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'

    for epoch in range(num_epochs):

        for batch_idx, (data, target) in enumerate(train_loader): 
        
            data = encoder.encode(data)
            data, target = data.to(device), target.to(device)

            model.forward_hebbian(data)
            model.apply_weight_change()

            if (batch_idx + 1) % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader)))
                
    
    
    torch.save(model.state_dict(), model_path)

    return

if __name__ == '__main__':
    import argparse
    import yaml

    # load in data from relavent config files
    parser = argparse.ArgumentParser(description="train a model using hebbian-like learning specific training configurations")
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
    }

    train_batch_size = train_configs['train_batch_size']

    train_dataset = ECGWaveformDataset(dataset_configs['data_file'], pd.read_csv(dataset_configs['metadata']), dataset_configs['binary'])
    train_mean, train_std = sample_dataset_statistics(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # --- Model Configs --- #
    model_configs = train_configs['model_configs']
    if model_configs['neuron_options']['spike_grad'] == 'atan':
        model_configs['neuron_options']['spike_grad'] = surrogate.atan(alpha=2)

    logging_configs={
        'model_name': train_configs['model_name'],
        'weight_folder': path_configs['weight_folder'],
    }
    
    # --- Train the Model --- #
    if train_configs['model_class'] == 'TwoLayerSNN':
        model = TwoLayerSNN(**model_configs)
    else:
        model = SpikingNetwork(**model_configs)

    if train_configs['encoder'] == 'rate':
        encoder = Rate_Encoder(**train_configs['encoder_args'])
    else:
        encoder = Current_Encoder(train_mean, train_std)
    
    num_epochs = train_configs['num_epochs']

    train_model(model, train_loader, encoder, num_epochs, device, logging_configs)