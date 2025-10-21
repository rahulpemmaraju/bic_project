import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset, DataLoader

class ECGWaveformDataset(Dataset):
    def __init__(self, data_file, metadata: pd.DataFrame, out_size=None):
        '''
        data_obj: hdf5 file containing the ecg_signals
        metadata: dataframe containing the metadata for the data 
        out_size: if not None, will resize the ecg to a specific length
        '''

        with h5py.File(data_file, 'r') as f:
            self.data = np.array(f['signal_arrays'])

        self.metadata = metadata
        self.out_size = out_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = metadata.iloc[idx]

        waveform = self.data[sample.array_index]
        label = sample.label

        if self.out_size is not None:
            t_old = np.linspace(0, 1, len(waveform))
            t_new = np.linspace(0, 1, self.out_size)
            waveform = interp1d(t_old, waveform)(t_new)

        return torch.tensor(waveform), label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_file = '../arr_only_neurokit_5_2.hdf5'
    metadata = pd.read_csv('../arr_only_neurokit_5_2.csv')
    out_size = None

    dataset = ECGWaveformDataset(data_file, metadata, out_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    wave, label = dataset.__getitem__(100)
    print(wave.shape, label)

    waves, labels = next(iter(dataloader))
    print(waves.shape, labels.shape, labels)

    plt.plot(wave)
    plt.title(f'{label}')
    plt.show()
    