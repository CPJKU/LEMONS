import os

import numpy as np
import pandas as pd
from torch.utils import data

from utilities.utils import pickle_load, get_user_id


class MSDDataset(data.Dataset):

    def __init__(self, tracks_dir: str, splits_dir: str, user_name: str, split_set: str, input_length=None):
        """
        Dataset for a single user of the MSD dataset.
        ----------
        tracks_dir : str
            Path to the npy of the tracks.
        splits_dir: str
            Path to the directory with the train, val, test, and labels
        user_name: str
            User name as alias of a user id contained in train_triplets.txt (check utilities.get_user_id)
        split_set: str
            Value in TRAIN, VALID, and TEST
        input_length: int
            Length of the track in datapoints. It depends on the used model.
        """

        # File paths
        self.tracks_dir = tracks_dir
        self.splits_dir = splits_dir

        self.user_name = user_name
        self.user_id = get_user_id(self.user_name)

        self.split_set = split_set
        self.input_length = input_length

        print('Loading data')
        self.track_list, self.labels = self.load_data()

    def load_data(self):
        """
        Loads the user data.
        Returns:
        -------
        track_list: list
             List containing the tracks ids of the MSD.
        labels: dict
            Dictionary containing the labels for the tracks.
        """

        labels = pd.read_csv(os.path.join(self.splits_dir, 'labels.csv'), names=['track_id', 'playcount']) \
            .set_index('track_id')

        if self.split_set == 'TRAIN':
            track_list = pickle_load(os.path.join(self.splits_dir, 'train.pkl'))
        elif self.split_set == 'VAL':
            track_list = pickle_load(os.path.join(self.splits_dir, 'val.pkl'))
        elif self.split_set == 'TEST':
            track_list = pickle_load(os.path.join(self.splits_dir, 'test.pkl'))
        else:
            raise ValueError('Split should be one of [TRAIN, VAL, TEST]')
        return track_list, labels

    def get_npy(self, msdid: str):
        """
        Returns the respective npy given the msdid.
        If split_set==TRAIN, it returns a snippet of the audio else it returns the whole audio.
        Parameters
        ----------
        msdid: str
            Million Song ID of the track

        Returns
        -------
        npy : np.array
            Snippet or Whole Audio selected
        """

        # Load input
        npy_path = os.path.join(self.tracks_dir, '{}.npy'.format(msdid))

        if self.split_set == 'TRAIN':
            npy = np.load(npy_path, mmap_mode='r')
            # It returns a random snippet (with length of input_length) of the file
            random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
            npy = np.array(npy[random_idx:random_idx + self.input_length])
        else:
            npy = np.load(npy_path)

        if npy.dtype == np.int16:
            # Conversion if needed
            npy = npy / np.iinfo(np.int16).max

        return npy

    def __getitem__(self, index):
        msdid = self.track_list[index]

        label_binary = np.array([1 if self.labels.loc[msdid]['playcount'] > 0 else 0], dtype='float32')
        npy = self.get_npy(msdid).astype('float32')

        return npy, label_binary

    def __len__(self):
        return len(self.track_list)


def get_msd_audio_loader(tracks_dir: str, splits_dir: str, user_name: str, batch_size: int, split_set: str,
                         input_length=None, num_workers=0):
    """
    Parameters
    ----------
    tracks_dir : str
        Path to the npy of the tracks.
    splits_dir: str
        Path to the directory with the train, val, test, and labels
    user_name: str
        User name as alias of a user id contained in train_triplets.txt (check utilities.get_user_id)
    batch_size:int
        Batch size
    split_set: str
        Value in TRAIN, VALID, and TEST
    input_length: int
        Length of the track in datapoints. It depends on the used model.
    num_workers: int
        Number of workers
    Returns
    -------
    data_loader:
        Dataloader for the MSDDataset
    """
    userdt = MSDDataset(tracks_dir, splits_dir, user_name, split_set=split_set, input_length=input_length)

    data_loader = data.DataLoader(dataset=userdt,
                                  batch_size=batch_size,
                                  shuffle=(split_set == "TRAIN"),
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader
