import base64
import os
import warnings
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import pydub
import streamlit as st
import streamlit.components.v1 as html_component
import torch
from audioLIME.factorization import DataBasedFactorization
from librosa.display import waveplot
from matplotlib import pyplot as plt
from scipy.io import wavfile
from sklearn.preprocessing import normalize

from conf.config import meta_path, model_path, web_folder, track_path, musdb_storage
from recsys.msd_user_loader import get_msd_audio_loader
from recsys.musdb_data_loader import get_musdb18_audio_loader

from audioLIME.lime_audio import LimeAudioExplainer
from explanations.spleeter_precomputed import SpleeterPrecomputedFactorization as SpleeterFactorization

from utilities.utils import pickle_load, get_color

warnings.filterwarnings("ignore")
best_confs = {
    'marko': {'model_load_path': os.path.join(model_path, "2020-9-29_18-39-35.735790/best_model.pth"),
              'model_type': 'base'},
    "matteo": {'model_load_path': os.path.join(model_path, "2020-10-1_4-42-30.828192/best_model.pth"),
               'model_type': 'base'},
    "johnny": {'model_load_path': os.path.join(model_path, "2020-10-1_2-20-5.691752/best_model.pth"),
               'model_type': 'base'},
    "elizabeth": {'model_load_path': os.path.join(model_path, "2020-10-1_8-23-15.276871/best_model.pth"),
                  'model_type': 'base'},
    "nina": {'model_load_path': os.path.join(model_path, "2020-10-1_6-35-0.98642/best_model.pth"),
             'model_type': 'base'},
    "paige": {'model_load_path': os.path.join(model_path, "2020-9-29_20-43-14.410040/best_model.pth"),
              'model_type': 'base'},
    "sandra": {'model_load_path': os.path.join(model_path, "2020-10-1_10-32-21.499646/best_model.pth"),
               'model_type': 'base'},

}

user_description = {
    'marko': "favourite genre is reggae. He also prefers more niche tracks and loud music.",
    "matteo": "favourite genres are trance, blues, and progressive.",
    "johnny": "listens to a bit of everything.",
    "elizabeth": "her top 3 genres she likes are rock, alternative metal, and heavy metal.",
    "nina": "favourite genres are rock, emo, and post-hardcore.",
    "paige": "mostly listens to popular music.",
    "sandra": "favourite genres are hip-hop and rap, especially dirty south rap.",
}

user_short_description = {
    'marko': '(reggae, niche tracks, loud music)',
    'matteo': '(trance, blues, progressive)',
    'johnny': '(diverse music taste)',
    'elizabeth': '(rock, alternative metal, heavy metal)',
    'nina': '(rock, emo, post-hardcore)',
    'paige': '(popular music)',
    'sandra': '(hiphop, rap, dirty south rap)'
}


def user_image(user_name, large=True):
    if not large:
        user_name = user_name + "-32x32"
    return 'https://sanders.cp.jku.at/share.cgi/{}.png?ssid=07Tk2FX&fid=07Tk2FX&path=%2Fusers&filename={}.png&openfolder=normal&ep='.format(
        user_name, user_name)


local_config = {
    "data_seed": 1057386,
    "num_workers": 10,
    "sample_rate": 16000,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "take_center": True,
    "snippet_length": 30,
    "batch_size": 5
}


@st.cache(show_spinner=False)
def load_track_dataframe(dataloader, conf):
    """
    Load the meta data information on the tracks listened by the user
    Parameters
    ----------
    dataloader
    conf

    Returns
    -------

    """
    print('Load Track Dataframe - Cache Miss')

    user_id = dataloader.dataset.user_id
    tracks = dataloader.dataset.fl
    split = dataloader.dataset.split

    # Loading track, artist, album names
    meta = pd.read_csv(os.path.join(meta_path, 'meta_data.csv'), index_col=0)
    meta = meta.loc[tracks]  # filtering and sorting according to the test loader
    meta.index.name = 'track'

    # Include user playcounts
    song_to_track = pd.read_csv(os.path.join(meta_path, 'song_to_track_cleaned.csv'))
    trip = pd.read_csv(os.path.join(meta_path, 'train_triplets_subset.csv'))

    trip = trip[trip.user == user_id]
    user_tracks = trip.merge(song_to_track).drop(columns=['song', 'user']).groupby(['track']).sum().reset_index()

    meta_user_tracks = meta.reset_index().merge(user_tracks, how='left').fillna(value=0)

    if split == 'TEST':
        meta_user_tracks['relevance'] = pickle_load(conf['results_path'])
        meta_user_tracks = meta_user_tracks.sort_values('relevance', ascending=False)
        meta_user_tracks = meta_user_tracks.round(4)
    else:
        meta_user_tracks = meta_user_tracks.sort_values('playcount', ascending=False)
        num_playcounts = int(meta_user_tracks.playcount.sum())

    num_tracks = len(meta_user_tracks[meta_user_tracks.playcount > 0])
    # Placing the track identifier at the end
    meta_user_tracks = meta_user_tracks[meta_user_tracks.columns.tolist()[1:] + ['track']]
    # meta_user_tracks= meta_user_tracks.drop(columns=['track'])
    # We return only the first 100 for train and 10 for test. There is no need to return more!
    if split == 'TEST':
        meta_user_tracks = meta_user_tracks[:10]
        return meta_user_tracks[:10]
    else:
        meta_user_tracks = meta_user_tracks[:100]
        return meta_user_tracks, num_tracks, num_playcounts


@st.cache(show_spinner=False)
def load_musdb_track_dataframe(dataloader, username, take_center, snippet_length):
    """
    Load the meta data information on the tracks listened by the user
    Parameters
    ----------
    dataset
    username
    take_center

    Returns
    -------

    """

    mus = dataloader.dataset.musdb
    column_names = ["title", "artist", "playcount", "relevance", "track"]
    meta_user_tracks = pd.DataFrame(
        [(m.title, m.artist, 0.0, 1.0, "TR" + str(idx).zfill(3)) for (idx, m) in enumerate(mus)],
        columns=column_names)

    results_path = os.path.join('/share/cp/projects/ajures/predictions/',
                                "musdb_{}_{}.pt".format(username, take_center))
    results_path = results_path.replace("_True",
                                        "_True_{}".format(snippet_length))  # only replaces it for take_center=True

    meta_user_tracks['relevance'] = pickle_load(results_path)
    meta_user_tracks = meta_user_tracks.sort_values('relevance', ascending=False)

    # meta_user_tracks = meta_user_tracks.round(4)
    # Return only the top-5
    return meta_user_tracks[:10]


class TemporalFactorization(DataBasedFactorization):
    def __init__(self, data_provider, n_temporal_segments, composition_fn):
        # assert isinstance(data_provider, RawAudioProvider)  # TODO: nicer check
        super().__init__(data_provider, n_temporal_segments, composition_fn)

    def initialize_components(self):
        waveform = self.data_provider.get_mix()
        self.original_components = [waveform]
        self._components_names = ["C"]


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


def take_top(components, indexes, scores, parameters, trim=True):
    """
    Takes the top-k components and aggregates them on the basis of the source
    Parameters
    ----------
    components: the interpretable components
    indexes: indexes of the components, ordered according to score
    scores: scores obtained by audioLIME
    k: number of top component or 'all'
    trim: removes 0s on both sides if any
    enhance: increases the amplitude of the components depending on their respective scores
    query_instrument: of which instruments the components should be returned (only work for TIme + Source)

    Returns
    -------
    top_components: array of shape [# sources, length_audio] or [1, lenght_audio - zeros] if trim = True
    """

    k = parameters['k']
    enhance = parameters['enhance']
    query_instrument = parameters['query_instrument']
    explanation_type = parameters['explanation_type']

    if query_instrument is not None:
        assert query_instrument in range(5), "Query instrument not valid!"
        assert explanation_type == 'Time + Source', 'Querying an instrument does not make sense if Time + Source is not selected'

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
    top_scores = scores[k_indexes]
    # st.write('The weight of the components is: ' + str((np.sum(top_scores) / np.sum(scrs)) * 100) + '%')

    if enhance and len(top_scores) > 1:
        top_components *= top_scores.reshape((-1, 1)) * 3  # TODO: maybe try something that is not linear?

    # Merge on source
    if explanation_type == 'Time':
        top_components = np.array(sum(top_components)).reshape([1, -1])
        sources_names = None
    else:
        sources = {}
        for position, index in enumerate(top_indexes):  # TODO: There is for sure a faster way (top_indexes % 5)
            source = index % 5
            sources.setdefault(source, []).append(top_components[position])

        sources_names = sorted(list(sources.keys()))
        top_components = np.array([sum(sources[k]) for k in sources_names])
        # top_components = np.array([sum(v) for k, v in sources.items()])

    if trim:
        trim_f = len(top_components[0]) - len(np.trim_zeros(sum(top_components), 'f'))  # TODO: FIND A BETTER WAY
        trim_b = len(np.trim_zeros(sum(top_components), 'b'))
        top_components = top_components[:, trim_f: trim_b]
        return top_components, sources_names, trim_f

    return top_components, sources_names


@st.cache(show_spinner=False)
def get_track_name(test_loader, dataset, selected_idx):
    if dataset == 'musdb':
        return str(test_loader.dataset.musdb.tracks[selected_idx])
    elif dataset == 'MSD':
        return str(test_loader.dataset.fl[selected_idx])
    else:
        return None


def get_mp3_path(file_name, dataset, dir):
    mp3_path = os.path.join(web_folder, dataset, dir, file_name + '.mp3')
    return mp3_path


@st.cache(show_spinner=False)
def generate_encoded_waveplot(mix: np.array, start=None, source_names=None):
    # Generate image
    if len(mix.shape) == 1:
        mix = mix.reshape((1, -1))

    real_start = 0.0
    sample_rate = local_config["sample_rate"]
    if start:
        real_start = round(start / sample_rate)
    n_rows = mix.shape[0]
    fig, axs = plt.subplots(nrows=n_rows, sharex=True, sharey=True,
                            figsize=(14, 3.2 if n_rows == 1 else 1.5 * n_rows))
    for source, i in zip(mix, range(n_rows)):
        ax_obj = axs[i] if n_rows > 1 else axs
        waveplot(source, sr=sample_rate, ax=ax_obj, color='cornflowerblue',
                 offset=real_start, x_axis='s')

        x_end = round(real_start + len(source) / sample_rate)
        num = int(x_end - real_start + 1)
        x_ticks = np.linspace(real_start, x_end, num=num)
        if num > 25:
            x_ticks = x_ticks[::5]
        elif num > 10:
            x_ticks = x_ticks[::2]
        ax_obj.set_xticks(x_ticks)
        ax_obj.set_ylim([-1, 1])
        ax_obj.set_yticks(np.linspace(-1.0, 1.0, num=5))
        if source_names:
            ax_obj.set_ylabel(get_source_name(source_names[i]).capitalize())

    # Encode Image
    imgdata = StringIO()
    plt.savefig(imgdata, format='svg', bbox_inches='tight')
    imgdata.seek(0)
    encoded_image = base64.b64encode(imgdata.getvalue().encode("utf-8")).decode("utf-8")
    height = 3.2 if n_rows == 1 else 1.5 * n_rows
    offset = 18 if source_names else 0
    return encoded_image, height, offset


def get_source_name(key: str):
    sources = ['vocals', 'piano', 'drums', 'bass', 'other']
    return sources[key]


def audio_image_player(encoded_image, mp3_path, height=None, bar_length=None, offset=0):
    html_component.html(
        """
        <div class="vis" style="width:1200px; margin:0 auto; display: inline-block">
            <img src="data:image/svg+xml;base64,""" + str(encoded_image) + """ " id="img" alt="image not found" class="center">
            <span id="bar" style="float:left; position:absolute; top:22px; left:""" + str(
            59 + offset) + """px; height:""" + str(
            238 if bar_length is None else (bar_length * 238) / 3.2) + """px; width:2px; background-color:black; opacity:0.8"></span>


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
                document.getElementById("bar").style.left = (""" + str(59 + offset) + """ + 1041.4 * time) + 'px';
            });
        </script>
        """
        , height=350 if height is None else (350 * height) / 3.2)


@st.cache(show_spinner=False)
def npy_to_mp3(mix: np.array, mp3_path: str):
    wav_io = BytesIO()
    wavfile.write(wav_io, 16000, mix)
    wav_io.seek(0)
    sound = pydub.AudioSegment.from_wav(wav_io)
    sound.export(mp3_path, format='mp3')


def show_badge(text, label, value):
    value = round(value, 2)
    st.markdown(
        f"{text} for the selected song: ![{label}](https://img.shields.io/static/v1?label={label}&message={value}"
        f"&color={get_color('standard', value)})")


def get_image_path(explanation_type):
    if explanation_type == 'Time':
        image_path = "./explanations/img/description_time.png"
    elif explanation_type == 'Source':
        image_path = "./explanations/img/description_source.png"
    else:
        image_path = "./explanations/img/description_timesource.png"
    return image_path


@st.cache(show_spinner=False)
def generate_explanation(dp, explanation_type: str, predicter, user, index, dataset):
    print('Explanation - Cache Miss')

    # First trying to load it
    precomputed_explanation = "/share/cp/projects/ajures/stored_explanations/{}/{}_{}.pkl".format(user, dataset, index)
    if os.path.isfile(precomputed_explanation):
        print('Loaded Explanation!')
        if explanation_type == 'Time':
            explanation = pickle_load(precomputed_explanation)['time']
        elif explanation_type == 'Source':
            explanation = pickle_load(precomputed_explanation)['source']
        else:
            explanation = pickle_load(precomputed_explanation)['time_source']

        interpretable_components = explanation['components']
        positive_weights_scores = [x for x in explanation['scores'] if x[1] > 0]
        indexes = np.array([x[0] for x in positive_weights_scores]).astype(np.int32)
        scores = np.array([x[1] for x in positive_weights_scores])
        fidelity = explanation['fidelity']

    else:
        def recommender_system(x):
            p = predicter.model.predict(x)
            return np.column_stack((1 - p, p))

        if explanation_type == 'Time':
            factorization = TemporalFactorization(dp, n_temporal_segments=5,
                                                  composition_fn=None)
        elif explanation_type == 'Source':
            factorization = SpleeterFactorization(dp, n_temporal_segments=1,
                                                  composition_fn=None,
                                                  model_name='spleeter:5stems')
        else:
            factorization = SpleeterFactorization(dp, n_temporal_segments=5,
                                                  composition_fn=None,
                                                  model_name='spleeter:5stems')

        explainer = LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

        explanation = explainer.explain_instance(factorization=factorization,
                                                 predict_fn=recommender_system,
                                                 top_labels=1,
                                                 num_samples=2048,
                                                 batch_size=5
                                                 )

        label = list(explanation.score.keys())[0]
        fidelity = explanation.score[label]
        interpretable_components = np.array(factorization.retrieve_components())
        # Using only positive scores
        positive_weights_scores = [x for x in explanation.local_exp[label] if x[1] > 0]
        indexes = np.array([x[0] for x in positive_weights_scores])
        scores = np.array([x[1] for x in positive_weights_scores])

    return fidelity, interpretable_components, indexes, scores


@st.cache(show_spinner=False)
def get_data_loader(type, local_config, test_set=None):
    print('Get Data Loaders - Cache Miss')
    if type == 'train':
        return get_msd_audio_loader(
            track_path,
            meta_path,
            local_config['username'],
            1,
            split_set='TRAIN',
        )
    else:
        if test_set == "musdb":
            return get_musdb18_audio_loader("/share/cp/temp/web/musdb",
                                            cache_path=musdb_storage,
                                            batch_size=1,
                                            take_center=local_config['take_center'],
                                            load_audio=False,
                                            snippet_length=local_config['snippet_length'])
        else:
            return get_msd_audio_loader(
                track_path,
                meta_path,
                local_config['username'],
                1,
                split_set='TEST',
            )
