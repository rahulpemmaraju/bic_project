import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch.surrogate as surrogate

import os
import random
import numpy as np
import pandas as pd

from dataloader import train_test_val_split, get_dataset_statistics
from models import SpikingNetwork, TwoLayerSNN, ThreeLayerSNN
from utils.encoding import Rate_Encoder, Current_Encoder

from utils.logger import log_model
import utils.metrics as metrics

def train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device):

    model.train()
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader): 
        optimizer.zero_grad()
       
        data = encoder.encode(data)
        data, target = data.to(device), target.to(device)

        output = model(data).squeeze(1)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))
        
        epoch_loss += loss.item() * data.shape[0]

    epoch_loss /= len(train_loader.dataset)
    
    return epoch_loss

def evaluate_model(model, test_loader, encoder, loss_fn, device):

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = encoder.encode(data)
            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)
            loss = loss_fn(output, target)

            test_loss += loss.item() * data.shape[0]

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return test_loss

def train_model(
        model, 
        train_loader, 
        val_loader, 
        encoder, 
        optimizer, 
        loss_fn, 
        num_epochs, 
        val_steps=2, 
        device='cpu',
        logging_configs={
            'model_name': None,
            'weight_folder': 'train_weights',
            'log_folder': 'train_logs',
            'log_steps': 10
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
    log_folder = os.path.join(logging_configs['log_folder'], logging_configs['model_name'])

    os.makedirs(log_folder, exist_ok=True)

    # save best model (based on validation loss)
    min_val_loss = np.inf

    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, encoder, optimizer, loss_fn, epoch, device)
        train_losses.append((epoch, train_loss))

        print('Epoch {} Average Loss: {}'.format(epoch, train_loss))

        if (epoch + 1) % val_steps == 0:
            val_loss = evaluate_model(model, val_loader, encoder, loss_fn, device)
            val_losses.append((epoch, val_loss))

            if val_loss < min_val_loss:
                # save model and optimizer in case we want to restart from last known point
                checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, model_path)

                print('New best model found! Saving weights.\n')
                min_val_loss = val_loss

        if (epoch + 1) % logging_configs['log_steps'] == 0:
            log_model(train_losses, val_losses, log_folder)

    return train_losses, val_losses

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
        "random_state": seed,
        **train_configs['dataset_configs']
    }

    train_batch_size = train_configs['train_batch_size']
    eval_batch_size = train_configs['eval_batch_size']

    train_dataset, val_dataset, test_dataset = train_test_val_split(**dataset_configs)
    train_mean, train_std = get_dataset_statistics(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size)
    print(test_dataset.metadata['patient'].unique())

    # --- Model Configs --- #
    model_configs = train_configs['model_configs']
    if model_configs['neuron_options']['spike_grad'] == 'atan':
        model_configs['neuron_options']['spike_grad'] = surrogate.atan(alpha=2)

    logging_configs={
        'model_name': train_configs['model_name'],
        'weight_folder': path_configs['weight_folder'],
        'log_folder': path_configs['log_folder'],
        'log_steps': train_configs['log_steps']
    }
    
    # --- Train the Model --- #
    if train_configs['model_class'] == 'TwoLayerSNN':
        model = TwoLayerSNN(**model_configs)
    else:
        model = SpikingNetwork(**model_configs)

    if "pretrained_weights" in train_configs:
        model.load_state_dict(torch.load(os.path.join(path_configs['weight_folder'], f"{train_configs['pretrained_weights']}.pth")))

    if "finetune" in train_configs:
        if train_configs["finetune"]:
            for param in model.snn_1.parameters():
                param.requires_grad = False

            for param in model.snn_2.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True


# Make sure fc layer is trainable
for param in model.fc.parameters():
    param.requires_grad = True

    if train_configs['encoder'] == 'rate':
        encoder = Rate_Encoder(**train_configs['encoder_args'])
    else:
        encoder = Current_Encoder(train_mean, train_std)

    learning_rate = train_configs['lr']

    if train_configs['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif train_configs['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif train_configs['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    if binary:
        loss_fn = nn.BCEWithLogitsLoss() # higher pos_weight -> correct positive prediction is more important 
    else:
        loss_fn = nn.CrossEntropyLoss(torch.tensor([1., 1., 1., 1., 1.])) # weighted CE Loss -> more heavily weights arrs

    num_epochs = train_configs['num_epochs']
    val_steps = train_configs['val_steps']

    print('Training Model')

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        encoder,
        optimizer,
        loss_fn,
        num_epochs,
        val_steps,
        device,
        logging_configs,
    )

    model_path = os.path.join(logging_configs['weight_folder'], logging_configs['model_name']) + '.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    threshold = None
    if binary:
        threshold = metrics.get_threshold(model, val_loader, encoder, device)
        checkpoint['threshold'] = threshold
        torch.save(checkpoint, model_path)


    accuracy = metrics.get_accuracy(model, test_loader, encoder, binary, threshold, device)
    print('\nTest Set Accuracy: {}'.format(accuracy))

    if binary:
        metrics.get_roc_curve(model, test_loader, encoder, threshold, os.path.join(logging_configs['log_folder'], logging_configs['model_name']), device)
    else:
        metrics.get_multiclass_metrics(model, test_loader, encoder, os.path.join(logging_configs['log_folder'], logging_configs['model_name']), device)