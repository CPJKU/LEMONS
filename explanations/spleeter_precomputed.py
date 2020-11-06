import os
import pickle

import librosa
import numpy as np
from audioLIME.data_provider import RawAudioProvider
from audioLIME.factorization import DataBasedFactorization

# SpleeterPrecomputedFactorization is not yet built into audioLIME
# TODO: remove this once it is part of audioLIME

try:
    from spleeter.separator import Separator
except ImportError:
    Separator = None


def separate(separator, waveform, target_sr, spleeter_sr):
    waveform = np.expand_dims(waveform, axis=0)
    waveform = librosa.resample(waveform, target_sr, spleeter_sr)
    waveform = np.swapaxes(waveform, 0, 1)
    prediction = separator.separate(waveform)
    return prediction


class SpleeterPrecomputedFactorization(DataBasedFactorization):
    def __init__(self, data_provider, n_temporal_segments, composition_fn, model_name,
                 spleeter_sources_path='/share/cp/projects/ajures/data/precomputed/',
                 target_sr=16000):
        assert isinstance(data_provider, RawAudioProvider)  # TODO: nicer check
        self.model_name = model_name
        self.target_sr = target_sr
        sample_name = os.path.basename(data_provider.get_audio_path().replace(".mp3", ""))
        self.sources_path = os.path.join(spleeter_sources_path,
                                         model_name.replace("spleeter:", ""), sample_name)
        # print(self.sources_path)

        super().__init__(data_provider, n_temporal_segments, composition_fn)

    def initialize_components(self):
        spleeter_sr = 44100

        prediction_path = os.path.join(self.sources_path, "prediction.pt")
        if not os.path.exists(prediction_path):
            separator = Separator(self.model_name, multiprocess=False)
            waveform = self.data_provider.get_mix()
            prediction = separate(separator, waveform, self.target_sr, spleeter_sr)
            if not os.path.exists(self.sources_path):
                os.mkdir(self.sources_path)
            pickle.dump(prediction, open(prediction_path, "wb"))
        else:
            print("Loading", prediction_path)

        prediction = pickle.load(open(prediction_path, "rb"))

        self.original_components = [
            librosa.resample(np.mean(prediction[key], axis=1), spleeter_sr, self.target_sr) for
            key in prediction]
        self._components_names = list(prediction.keys())
