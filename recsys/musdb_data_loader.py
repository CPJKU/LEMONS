# coding: utf-8
import os

import librosa
import musdb
from torch.utils import data

from utilities.utils import pickle_dump, pickle_load


def compute_start_length(x, snippet_length, sample_rate):
    center = len(x) // 2
    length = snippet_length * sample_rate
    return center - length // 2, length


class Musdb18Dataset(data.Dataset):
    def __init__(self, musdb_root,
                 precomputed_path='musdb_audios.pt',
                 input_length=30,
                 sample_rate=16000,
                 take_center=False,
                 load_audio=True,
                 snippet_length=None):

        self.input_length = input_length  # currently in seconds
        self.musdb = musdb.DB(root=musdb_root)
        self.sample_rate = sample_rate
        self.take_center = take_center
        self.load_audio = load_audio

        self.snippet_length = None
        if self.take_center:
            assert snippet_length is not None
            self.snippet_length = snippet_length

        if self.load_audio:
            if os.path.exists(precomputed_path):
                self.audios = pickle_load(precomputed_path)
            else:
                self.audios = []
                for m in self.musdb:
                    print("Processing", m)
                    x = m.audio.mean(axis=1)
                    x = librosa.resample(x, m.rate, self.sample_rate)  # take from config
                    self.audios.append(x)
                pickle_dump(self.audios, precomputed_path)

    def get_audio_path(self, index):
        return self.musdb[index].path

    def __getitem__(self, index):
        if not self.load_audio:
            raise Exception("Dataset was initialized without loading audio")
        x = self.audios[index]
        if self.take_center:
            start, length = compute_start_length(x, self.snippet_length, self.sample_rate)
            if len(x) > length:
                x = x[start:start + length]  # taking center piece
        x = x.astype('float32')
        return x, -1

    def __len__(self):
        return len(self.audios)


def get_musdb18_audio_loader(root: str, cache_path: str, batch_size: int, num_workers=0, take_center=False,
                             load_audio=True, snippet_length=None):
    """
    Parameters
    ----------
    root:
    cache_path:
    batch_size
    num_workers
    take_center
    load_audio
    snippet_length

    Returns
    -------

    """

    assert os.path.exists(root), '{} does not exist'.format(root)
    dataset = Musdb18Dataset(root,
                             precomputed_path=cache_path,
                             take_center=take_center,
                             load_audio=load_audio,
                             snippet_length=snippet_length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader
