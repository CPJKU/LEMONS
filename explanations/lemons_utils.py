import argparse
import base64
import os
import warnings
from enum import Enum, auto
from io import StringIO, BytesIO
from pathlib import Path

import musdb
import numpy as np
import pandas as pd
import pydub
import streamlit as st
import streamlit.components.v1 as html_component
import torch
from audioLIME.data_provider import RawAudioProvider
from audioLIME.factorization import DataBasedFactorization
from audioLIME.lime_audio import LimeAudioExplainer
from librosa.display import waveplot
from matplotlib import pyplot as plt
from scipy.io import wavfile
from sklearn.preprocessing import normalize

from conf.config import data_path, models_path, msd_npy_path, web_folder, stored_explanations_path, \
    musdb_storage_root, musdb_storage_cache
from explanations.spleeter_precomputed import SpleeterPrecomputedFactorization as SpleeterFactorization
from recsys.msd_user_loader import get_msd_audio_loader
from recsys.musdb_data_loader import compute_start_length
from recsys.musdb_data_loader import get_musdb18_audio_loader
from recsys.predict import Predict
from utilities.utils import pickle_load, get_color, pickle_dump

warnings.filterwarnings("ignore")


class Users(Enum):
    marko = auto()
    matteo = auto()
    johnny = auto()
    elizabeth = auto()
    nina = auto()
    paige = auto()
    sandra = auto()


class ExplanationTypes(Enum):
    Time = auto()
    Source = auto()
    TimeAndSource = auto()


best_models = {
    Users.marko: os.path.join(models_path, "marko/best_model.pth"),
    Users.matteo: os.path.join(models_path, "matteo/best_model.pth"),
    Users.johnny: os.path.join(models_path, "johnny/best_model.pth"),
    Users.elizabeth: os.path.join(models_path, "elizabeth/best_model.pth"),
    Users.nina: os.path.join(models_path, "nina/best_model.pth"),
    Users.paige: os.path.join(models_path, "paige/best_model.pth"),
    Users.sandra: os.path.join(models_path, "sandra/best_model.pth"),
}

user_descriptions = {
    Users.marko: {
        'full': "favourite genre is reggae. He also prefers more niche tracks and loud music.",
        'short': "(reggae, niche tracks, loud music)",
        'image': 'https://i.ibb.co/989QyHF/marko.png'
    },
    Users.matteo: {
        'full': "favourite genres are trance, blues, and progressive.",
        'short': "(trance, blues, progressive)",
        'image': 'https://i.ibb.co/nrhTNtM/matteo.png'
    },
    Users.johnny: {
        'full': "listens to a bit of everything.",
        'short': "(diverse music taste)",
        'image': 'https://i.ibb.co/h85VcPH/johnny.png'
    },
    Users.elizabeth: {
        'full': "her top 3 genres she likes are rock, alternative metal, and heavy metal.",
        'short': "(rock, alternative metal, heavy metal)",
        'image': 'https://i.ibb.co/yWTYnjC/elizabeth.png'
    },
    Users.nina: {
        'full': "favourite genres are rock, emo, and post-hardcore.",
        'short': "(rock, emo, post-hardcore)",
        'image': 'https://i.ibb.co/qJLgRT1/nina.png'
    },
    Users.paige: {
        'full': "mostly listens to popular music.",
        'short': "(popular music)",
        'image': 'https://i.ibb.co/tJBSYNB/paige.png'
    },
    Users.sandra: {
        'full': "favourite genres are hip-hop and rap, especially dirty south rap.",
        'short': "(hiphop, rap, dirty south rap)",
        'image': 'https://i.ibb.co/YZcD3Wm/sandra.png'
    }
}

explanation_types_description = {
    ExplanationTypes.Time: {
        'desc': "*Time-based* explanations show what are the snippets of the audio that influenced the recommendation the most.",
        'url': "https://i.ibb.co/Gk25Mvc/description-time.png"
    },
    ExplanationTypes.Source: {
        'desc': "*Source-based* explanations show what are the source components of the audio that influenced the recommendation the most.",
        'url': "https://i.ibb.co/GsTFT70/description-source.png"
    },
    ExplanationTypes.TimeAndSource: {
        'desc': "*Time+Source-based* explanations combine both the Time-based and Source-based explanations.",
        'url': "https://i.ibb.co/xS6Qf6b/description-timesource.png"
    }
}


@st.cache(show_spinner=False)
def generate_audio_link(file_name, test_dataset, is_explanation=False):
    '''
    The function generates the link for the audio and needs to be adapted. The audio has to be stored on a server.
    '''
    external_file_template = 'https://sanders.cp.jku.at/share.cgi/{}?ssid=07Tk2FX&fid=07Tk2FX&path=/{}&filename={}&openfolder=normal&ep='

    if is_explanation:
        audio_url = external_file_template.format(file_name, test_dataset + '/explanations', file_name)
    else:
        audio_url = external_file_template.format(file_name, test_dataset + '/snippets', file_name)

    return audio_url


@st.cache(show_spinner=False)
def load_meta():
    print('Load Meta - Cache Miss')
    meta = pd.read_csv(os.path.join(data_path, 'meta_data.csv'), index_col=0)
    meta.index.name = 'track'
    return meta


@st.cache(show_spinner=False)
def load_msd_tracks(user, split):
    print('Load MSD Tracks - Cache Miss')

    user_data_path = os.path.join(data_path, 'split/{}-1057386/'.format(user.name))
    labels = pd.read_csv(os.path.join(user_data_path, 'labels.csv'), names=['track', 'playcount']).set_index('track')
    split_data = pickle_load(os.path.join(user_data_path, '{}.pkl'.format(split)))

    user_data = pd.merge(load_meta(), labels, left_index=True, right_index=True).loc[split_data]

    if split == 'train' or split == 'val':
        user_data = user_data.sort_values('playcount', ascending=False)
    elif split == 'test':
        user_data['relevance'] = pickle_load(os.path.join(models_path, user.name, "MSD_{}.pkl".format(user.name)))
        user_data = user_data.sort_values('relevance', ascending=False).round(4)
    else:
        raise ValueError('No split avaiable!')

    # Placing track column at the end
    user_data = user_data.reset_index()
    user_data = user_data[user_data.columns.tolist()[1:] + ['track']]

    return user_data


@st.cache(show_spinner=False)
def load_musdb18_tracks(user):
    print('Load musdb18 Tracks - Cache Miss')
    mus = musdb.DB(root=musdb_storage_root)

    user_data = pd.DataFrame(
        [(m.title, m.artist, 0.0, 1.0, "TR" + str(idx).zfill(3)) for (idx, m) in enumerate(mus)],
        columns=["title", "artist", "playcount", "relevance", "track"])

    results_path = os.path.join(models_path, user.name, "musdb_{}.pt".format(user.name))

    user_data['relevance'] = pickle_load(results_path)
    user_data = user_data.sort_values('relevance', ascending=False)

    return user_data


class TemporalFactorization(DataBasedFactorization):
    def __init__(self, data_provider, n_temporal_segments, composition_fn):
        super().__init__(data_provider, n_temporal_segments, composition_fn)

    def initialize_components(self):
        waveform = self.data_provider.get_mix()
        self.original_components = [waveform]
        self._components_names = ["C"]


def take_top(components, indexes, scores, parameters, trim=True):
    """
    Takes the top-k components and aggregates them on the basis of the source
    Parameters
    ----------
    components: the interpretable components
    indexes: indexes of the components, ordered according to score
    scores: scores obtained by audioLIME
    parameters: used in the take_top, the include:
        k: number of top component or 'all'
        query_instrument: of which instruments the components should be returned (only work for TIme + Source)
        explanation_type: type of explanation to distinguish different possibilites
    trim: removes 0s on both sides if any

    Returns
    -------
    top_components: array of shape [# sources, length_audio] or [# sources, length_audio without zero trails] if trim = True
    """

    k = parameters['k']
    query_instrument = parameters['query_instrument']
    explanation_type = parameters['explanation_type']

    if query_instrument is not None:
        assert query_instrument in range(5), "Query instrument not valid!"
        assert explanation_type == ExplanationTypes.TimeAndSource, 'Querying an instrument does not make sense if Time + Source is not selected'

        # Selecting the indexes and the scores only related to the query instrument
        query_indexes = (indexes % 5 == query_instrument).astype('bool')
        indexes = indexes[query_indexes]
        scores = scores[query_indexes]

    # Normalize scores
    scores = np.array(normalize(scores.reshape(1, -1), norm='l1')[0])

    if k == 'all':
        k_indexes = np.arange(len(indexes))
    else:
        k_indexes = np.argsort(scores)[-k:]

    top_indexes = indexes[k_indexes]
    top_components = components[top_indexes]
    # top_scores = scores[k_indexes]

    # st.write('The weight of the components is: ' + str((np.sum(top_scores) / np.sum(scores)) * 100) + '%')

    # Merge on source
    if explanation_type == ExplanationTypes.Time:
        top_components = np.array(sum(top_components)).reshape([1, -1])
        sources_names = None
    else:
        sources = {}
        for position, index in enumerate(top_indexes):
            source = index % 5
            sources.setdefault(source, []).append(top_components[position])

        sources_names = sorted(list(sources.keys()))
        top_components = np.array([sum(sources[k]) for k in sources_names])  # First dimension sorted by source name

    if trim:
        trim_f = len(top_components[0]) - len(
            np.trim_zeros(sum(top_components), 'f'))  # TODO: is there a more optimized way?
        trim_b = len(np.trim_zeros(sum(top_components), 'b'))
        top_components = top_components[:, trim_f: trim_b]

    return top_components, sources_names


def save_explanation(audio_numpy, track_name, explanation_type, user_name, param_string, test_set):
    exp_file_name = "{}_{}_{}_{}.ex.mp3"
    exp_file_name = exp_file_name.format(track_name, explanation_type, user_name, param_string)

    exp_file = os.path.join(web_folder, test_set, 'explanations', exp_file_name)

    npy_to_mp3(audio_numpy, exp_file)
    return exp_file_name


@st.cache(show_spinner=False)
def load_snippet(file_path, test_dataset):
    raw = RawAudioProvider(file_path)
    if test_dataset == 'musdb18':
        test_start, test_length = compute_start_length(raw.get_mix(), 30, 16000)
        raw.set_analysis_window(test_start, test_length)
    return raw.get_mix()


@st.cache(show_spinner=False)
def generate_encoded_waveplot(mix: np.array, offset=0.0, source_names=None):
    '''

    Parameters
    ----------
    mix: Numpy array of the sources
    offset: offset in number of samples
    source_names: Sorted list of source names

    Returns
    -------

    '''

    # 1° Step - Generate image

    if len(mix.shape) == 1:
        mix = mix.reshape((1, -1))
    n_sources = mix.shape[0]

    mix_start = round(offset / 16000)
    mix_end = round(offset + mix.shape[-1] / 16000)

    num_x_ticks = int(mix_end - mix_start + 1)
    x_ticks = np.linspace(mix_start, mix_end, num=num_x_ticks)
    if num_x_ticks > 25:
        x_ticks = x_ticks[::5]
    elif num_x_ticks > 10:
        x_ticks = x_ticks[::2]
    fig, axs = plt.subplots(nrows=n_sources, sharex=True, sharey=True,
                            figsize=(14, 3.2 if n_sources == 1 else 1.5 * n_sources))

    for i, source in enumerate(mix):
        ax_obj = axs[i] if n_sources > 1 else axs
        waveplot(source, sr=16000, ax=ax_obj, color='cornflowerblue', offset=mix_start, x_axis='s')
        ax_obj.set_xticks(x_ticks)
        ax_obj.set_ylim([-1, 1])
        ax_obj.set_yticks(np.linspace(-1.0, 1.0, num=5))
        if source_names:
            ax_obj.set_ylabel(get_source_name(source_names[i]).capitalize())

    # 2° Step - Encode Image

    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)
    encoded_image = base64.b64encode(imgdata.getvalue().encode("utf-8")).decode("utf-8")

    return encoded_image


def get_source_name(key):
    sources = ['vocals', 'piano', 'drums', 'bass', 'other']
    return sources[key]


def audio_image_player(encoded_image, mp3_path, source_names=None):
    # Scaling factors for the interface
    height_factor = 3.2 if (not source_names or len(source_names) == 1) else 1.5 * len(source_names)
    offset_factor = 18 if source_names else 0

    bar_offset = 59 + offset_factor
    bar_height = 238 if height_factor is None else (height_factor * 238) / 3.2
    component_height = 350 if height_factor is None else (350 * height_factor) / 3.2
    html_component.html(
        """
        <div class="vis" style="width:1200px; margin:0 auto; display: inline-block">
            <img src="data:image/svg+xml;base64,""" + str(encoded_image) + """ " id="img" alt="image not found" class="center">
            <span id="bar" style="float:left; position:absolute; top:22px; left:""" + str(
            bar_offset) + """px; height:""" + str(bar_height) + """px; width:2px; background-color:black; opacity:0.8"></span>

            <div id="player">
                <audio controls id="audio" src=" """ + mp3_path + """ " style="width: 1100px" preload="auto">
                </audio>
            </div>

        <br>
        </div>
        <script type="text/javascript">
            var player = document.getElementById("audio");
            player.addEventListener("timeupdate", function() {
                var time = this.currentTime / this.duration;
                document.getElementById("bar").style.left = (""" + str(bar_offset) + """ + 1041.4 * time) + 'px';
            });
        </script>
        """
        , height=component_height)


@st.cache(show_spinner=False)
def npy_to_mp3(mix: np.array, mp3_path: str, sr=16000):
    wav_io = BytesIO()
    wavfile.write(wav_io, sr, mix)
    wav_io.seek(0)
    sound = pydub.AudioSegment.from_wav(wav_io)
    sound.export(mp3_path, format='mp3')


def show_badge(text, label, value):
    value = round(value, 2)
    st.markdown(
        f"{text} for the selected song: ![{label}](https://img.shields.io/static/v1?label={label}&message={value}"
        f"&color={get_color('standard', value)})")


@st.cache(show_spinner=False)
def generate_explanation(file_path, explanation_type: ExplanationTypes, user_name: str, index, dataset: str):
    print('Generate Explanation - Cache Miss')

    # Try to load the explanation
    explanation_path = os.path.join(stored_explanations_path, '{}/{}_{}_{}.pkl'.format(user_name,
                                                                                       dataset,
                                                                                       index,
                                                                                       explanation_type.name))

    if os.path.isfile(explanation_path):
        print('Explanation Loaded')
        explanation_dict = pickle_load(explanation_path)

        interpretable_components = explanation_dict['components']
        positive_weights_scores = [x for x in explanation_dict['scores'] if x[1] > 0]
        positive_indexes = np.array([x[0] for x in positive_weights_scores]).astype(np.int32)
        positive_scores = np.array([x[1] for x in positive_weights_scores])
        fidelity = explanation_dict['fidelity']
    else:
        print('Explanation not found. Running audioLIME.')

        rap = RawAudioProvider(file_path)
        test_loader = get_test_data_loader(user_name, dataset)

        model_load_path = best_models[Users[user_name]]
        predicter = Predict(test_loader, argparse.Namespace(**{'use_tensorboard': 0,
                                                               'model_load_path': model_load_path,
                                                               'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
                                                               'user_name': user_name
                                                               }))

        def recommender_system(x):
            p = predicter.model.predict(x)
            return np.column_stack((1 - p, p))

        if explanation_type == ExplanationTypes.Time:
            factorization = TemporalFactorization(rap, n_temporal_segments=5,
                                                  composition_fn=None)
        elif explanation_type == ExplanationTypes.Source:
            factorization = SpleeterFactorization(rap, n_temporal_segments=1,
                                                  composition_fn=None,
                                                  model_name='spleeter:5stems')
        else:
            factorization = SpleeterFactorization(rap, n_temporal_segments=5,
                                                  composition_fn=None,
                                                  model_name='spleeter:5stems')

        explainer = LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

        audiolime_explanation = explainer.explain_instance(factorization=factorization,
                                                           predict_fn=recommender_system,
                                                           top_labels=1,
                                                           num_samples=2048,
                                                           batch_size=5
                                                           )

        label = list(audiolime_explanation.score.keys())[0]

        interpretable_components = np.array(factorization.retrieve_components())
        weights_scores = audiolime_explanation.local_exp[label]
        fidelity = audiolime_explanation.score[label]
        explanation_dict = {'components': interpretable_components, 'scores': weights_scores, 'fidelity': fidelity}

        # Save
        Path(os.path.dirname(explanation_path)).mkdir(parents=True, exist_ok=True)
        pickle_dump(explanation_dict, explanation_path)

        positive_weights_scores = [x for x in explanation_dict['scores'] if x[1] > 0]
        positive_indexes = np.array([x[0] for x in positive_weights_scores]).astype(np.int32)
        positive_scores = np.array([x[1] for x in positive_weights_scores])

    return fidelity, interpretable_components, positive_indexes, positive_scores


@st.cache(show_spinner=False)
def get_test_data_loader(user_name: str, test_set: str):
    print('Get Test Data Loader - Cache Miss')
    if test_set == "musdb18":
        return get_musdb18_audio_loader(musdb_storage_root,
                                        cache_path=musdb_storage_cache,
                                        batch_size=1,
                                        take_center=True,
                                        load_audio=False,
                                        snippet_length=30)
    else:
        return get_msd_audio_loader(msd_npy_path,
                                    os.path.join(data_path, 'split', '{}-1057386'.format(user_name)),
                                    user_name,
                                    batch_size=1,
                                    split_set='TEST')


def plot_sidebar(st):
    st.sidebar.header("How does it work?")

    st.sidebar.subheader('Track Recommendation')
    st.sidebar.markdown(
        'The Recommender System (RS) learns the general music preferences of the users from the audio tracks they listened to.\n')
    st.sidebar.markdown(
        'The RS is then used to predict the relevance of new tracks for each user. Tracks with higher relevance are then recommended.'
    )
    st.sidebar.image('./explanations/img/audioRecsys.png', use_column_width=True)

    st.sidebar.subheader('Listenable Explanation Generation')
    st.sidebar.markdown(
        "To explain why the RS recommended a specific track, we decompose the audio in easy-to-understand components such as single instruments or time-consecutive parts of the song.")
    st.sidebar.markdown(
        'Using then the tool *audioLIME*, we find out which of these components influenced the recommendation the most and provide them as listenable explanations.')
    st.sidebar.markdown(
        'There are three way to generate these components: *time-based* explanations, *source-based* explanations, and *time+source-based* explanations.')
    st.sidebar.markdown(
        '**Time-based** explanations segment the audio and show the *snippets* that influenced the recommendation the most.')
    st.sidebar.image('./explanations/img/description_time.png', use_column_width=True)
    st.sidebar.markdown(
        '**Source-based** explanations use source separation and show which *instruments* influenced the recommendation the most.')
    st.sidebar.image('./explanations/img/description_source.png', use_column_width=True)
    st.sidebar.markdown(
        '**Time+Source-based** explanations combine both above approaches and show *the snippets of the instruments* that influenced the recommendation the most.')

    st.sidebar.image('./explanations/img/description_timesource.png', use_column_width=True)
