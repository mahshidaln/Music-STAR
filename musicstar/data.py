import os
import sys
import data
import h5py
import tqdm
import torch
import random
import shutil
import librosa
import numpy as np
import torch.utils.data as data
from pathlib import Path
from scipy.io import wavfile
from tempfile import NamedTemporaryFile

from utils import mu_law, setup_logger
from audio import WavFilesDataset, PitchAugmentation

logger = setup_logger(__name__, 'log_data.log')


class H5Dataset(data.Dataset):
    """read data from h5 files"""
    def __init__(self, args, path, augmentation=None):
        self.path = Path(path)
        self.segment_length = args.segment_length
        self.epoch_length = args.epoch_length
        
        self.augmentation = augmentation
        self.short = short
        self.whole_samples = whole_samples

        self.file_paths = list(self.path.glob('**/*.h5'))

        if not self.file_paths:
            logger.error(f'No files found in {self.path}')

        logger.info(f'Dataset created. {len(self.file_paths)} files, '
                    f'augmentation: {self.augmentation is not None}. '
                    f'Path: {self.path}')


    def read_data(self, dataset, path):
        """reads wav data from h5 file"""
        length = dataset.shape[0]
        if length <= self.segment_length:
            logger.debug('Length of %s is %s', path, length)
        start_time = random.randint(0, length - self.segment_length)
        data = dataset[start_time: start_time + self.segment_length]
        assert data.shape[0] == self.segment_length

        return data.T

    def read_h5_file(self, h5file_path):
        """gets h5 file content"""
        f = h5py.File(h5file_path, 'r')
        return  f['wav']

    def _random_file(self):
        """picks a file randomly and return the specific instrument file"""
        #track_no = f'{np.random.randint(len(self.file_paths)//3):03}'
        #track_name = f'{track_no}.{part}.h5'
        return random.choice(self.file_paths)

    def get_random_slice(self):
        """returns the wav data of the selected slice from h5 file"""
        h5file_path = self._random_file()
        dataset = self.read_h5_file(h5file_path)
        return self.read_data(dataset, h5file_path)


    def __getitem__(self):
        """returns the tensor of the segmented data from the specified part of a track"""
        data_and_aug = None
        while data_and_aug is None:
            data = self.get_random_slice()
            if self.augmentation:
                data_and_aug = [data, self.augmentation(data)]
            else:
                data_and_aug = [data, data]
            data_and_aug = [mu_law(x / 2 ** 15) for x in data_and_aug]
        
        return torch.tensor(data_and_aug[0]), torch.tensor(data_and_aug[1])


    def __len__(self):
        return self.epoch_len


class Dataset:
    """return the train and validation dataset"""
    def __init__(self, args, domain_path):
        if args.data_aug:
            augmentation = PitchAugmentation(args)
        else:
            augmentation = None

        self.train_dataset = H5Dataset(args, domain_path / 'train', augmentation=augmentation)
        self.train_loader = data.DataLoader(self.train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

        self.train_iter = iter(self.train_loader)

        self.valid_dataset = H5Dataset(args, domain_path / 'val', augmentation=augmentation)
        self.valid_loader = data.DataLoader(self.valid_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers // 10 + 1,
                                            pin_memory=True)

        self.valid_iter = iter(self.valid_loader)
