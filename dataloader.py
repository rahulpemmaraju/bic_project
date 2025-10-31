import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

class ECGWaveformDataset(Dataset):
    def __init__(self, data_file, metadata: pd.DataFrame, binary=False, out_size=None):
        '''
        data_obj: hdf5 file containing the ecg_signals
        metadata: dataframe containing the metadata for the data 
        binary: whether or not the problem is a binary classification problem (normal vs abnormal)
        out_size: if not None, will resize the ecg to a specific length
        '''

        with h5py.File(data_file, 'r') as f:
            self.data = np.array(f['signal_arrays'])

        self.metadata = metadata
        self.out_size = out_size
        self.binary = binary

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = metadata.iloc[idx]

        waveform = self.data[sample.array_index]
        label = sample.label

        if self.binary:
            label = int(label > 0)

        if self.out_size is not None:
            t_old = np.linspace(0, 1, len(waveform))
            t_new = np.linspace(0, 1, self.out_size)
            waveform = interp1d(t_old, waveform)(t_new)

        return torch.tensor(waveform), label
    
def train_test_val_split(data_file, metadata, train_prop, val_prop, test_prop, binary=False, out_size=None, random_state=0):
    # split the data into train/val/test splits at a patient level -> return ECGWaveformDataset objects for each one
    # data_file: path to file with raw data
    # metadata: path to metadata file
    # train_prop: proportion of data to use for training
    # val_prop: proportion of data to use for validation
    # test_prop: proportion of data to use for testing
    # binary: whether or not the problem is a binary classification problem (normal vs abnormal)
    # out_size: used to resize ecg to specific length (default None -> no resizing which is what we probably want)
    # random_state: ensures consistent splitting of data

    metadata_df = pd.read_csv(metadata)

    patients = metadata_df['patient'].unique()
    train_val_patients, test_patients = train_test_split(patients, test_size=test_prop, random_state=random_state)
    
    train_patients, val_patients = train_test_split(train_val_patients, test_size=val_prop, random_state=random_state)

    train_df = metadata_df[metadata_df['patient'].isin(train_patients)]
    val_df = metadata_df[metadata_df['patient'].isin(val_patients)]
    test_df = metadata_df[metadata_df['patient'].isin(test_patients)]

    train_dataset = ECGWaveformDataset(data_file, train_df, binary, out_size)
    val_dataset = ECGWaveformDataset(data_file, val_df, binary, out_size)
    test_dataset = ECGWaveformDataset(data_file, test_df, binary, out_size)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_file = '../arr_only_neurokit_5_2.hdf5'
    metadata = '../arr_only_neurokit_5_2.csv'
    out_size = None

    
    train_test_val_split(data_file, metadata, 0.6, 0.2, 0.2, out_size=None)