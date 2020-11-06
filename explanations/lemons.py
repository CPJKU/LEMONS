import argparse
import os

import numpy as np
import streamlit as st
from audioLIME.data_provider import RawAudioProvider

from conf.config import external_file_template
from explanations.lemons_utils import best_confs, local_config, load_track_dataframe, plot_sidebar, \
    load_musdb_track_dataframe, take_top, get_mp3_path, show_badge, get_track_name, generate_explanation, \
    get_image_path, user_image, user_description, get_data_loader, user_short_description
from explanations.lemons_utils import generate_encoded_waveplot, audio_image_player, npy_to_mp3
from recsys.musdb_data_loader import compute_start_length
from recsys.predict import Predict

plot_sidebar(st)

# source: https://discuss.streamlit.io/t/where-to-set-page-width-when-set-into-non-widescreeen-mode/959/2
st.markdown(
    f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: 1200px;
    }}

        h1 {{ text-align:center }}
        

        div.Widget.row-widget.stRadio > div{{flex-direction:row;}}
        </style>
    """, unsafe_allow_html=True,
)
stereo_source = "https://sanders.cp.jku.at/share.cgi/stereo.png?ssid=07Tk2FX&fid=07Tk2FX&path=%2F&filename=stereo.png&openfolder=normal&ep="
st.markdown(
    "# ![](https://sanders.cp.jku.at/share.cgi/stereo.png?ssid=07Tk2FX&fid=07Tk2FX&path=%2F&filename=stereo.png&openfolder=normal&ep=)   LEMONS: Listenable Explanations for Music recOmmeNder Systems")

st.markdown(
    "## User/Persona Selection\n"
    "Below you can explore the 7 users/personas of our demo. Each user is characterized by a distinctive music preference.\n"
)

entries = [x + ' - ' + user_short_description[x] for x in list(best_confs.keys())]
username = st.selectbox("Which user?", entries, index=3, format_func=lambda x: x.capitalize()).split('-', 1)[0][:-1]

# Modify configuration
local_config = {**local_config,
                "username": username,
                "model_type": best_confs[username]['model_type'],
                "model_load_path": best_confs[username]['model_load_path'],
                "results_path": os.path.dirname(best_confs[username]['model_load_path']) + "/results.pkl",
                "use_tensorboard": 0,
                }

train_loader = get_data_loader('train', local_config)
train_tracks, num_tracks, num_playcounts = load_track_dataframe(train_loader, local_config)
st.subheader('Selected user profile')

consum_string = 'The user listened to *{}* tracks (shown below sorted by playcount) for a total of *{}* listening events.'.format(
    num_tracks,
    num_playcounts)
st.markdown("<img style='float: left;' src='{}'> <div> **{}** <br> {} <br> {} </div>".
            format(user_image(username),
                   username.capitalize(),
                   user_description[username].capitalize(), consum_string),
            unsafe_allow_html=True)

st.dataframe(train_tracks)

st.markdown('## Track Recommendation')
st.markdown("Select the music dataset from which to receive the recommendations.")
test_set_name = st.radio('Which music dataset?', ['musdb18', 'MSD', ])
if test_set_name == 'musdb18':
    test_set = 'musdb'
else:
    test_set = test_set_name

test_loader = get_data_loader('test', local_config, test_set)
predicter = Predict(test_loader, argparse.Namespace(**local_config))

if test_set == "musdb":
    test_tracks = load_musdb_track_dataframe(test_loader, username, local_config['take_center'],
                                             local_config['snippet_length'])
else:
    test_tracks = load_track_dataframe(test_loader, local_config)

st.markdown(
    'The top-10 recommended tracks from the dataset *{}* are the shown below sorted by relevance.'.format(test_set))
st.dataframe(test_tracks)

st.write("Select a recommended track (sorted by relevance)")
selected_track = st.selectbox("Which track do you want to listen to?", np.concatenate(
    ([''], test_tracks['title'].values + ' - ' + test_tracks['artist'].values)), index=0)

if selected_track:

    track_name, artist_name = selected_track.split(' - ', 1)
    selected_idx = test_tracks[(test_tracks.title == track_name) & (test_tracks.artist == artist_name)].index[0]

    track_name = get_track_name(test_loader, test_set, selected_idx)

    mp3_path = get_mp3_path(track_name, test_set, 'snippets')

    dp = RawAudioProvider(mp3_path)

    # TODO: Maybe refactor this?
    if local_config['take_center'] and test_set == 'musdb':
        test_start, test_length = compute_start_length(dp.get_mix(), local_config['snippet_length'],
                                                       16000)  # take from config
        dp.set_analysis_window(test_start, test_length)

    x = dp.get_mix()  # TODO: set analysis window before! (otherwise it will take the full track)
    encode_image, _, _ = generate_encoded_waveplot(x)
    show_badge('Relevance', 'relevance', test_tracks.loc[selected_idx]['relevance'])
    # st.write("You can listen to the selcted audio below:")

    # TODO: move the link generation somewhere else (maybe)

    audio_image_player(encode_image, external_file_template.format(track_name + '.mp3', test_set + '/snippets', track_name + '.mp3'))

    img_exp = {
        "Time": "description_timen.png",
        "Source": "description_sourcen.png",
        "Time + Source": "description_timesourcen.png"
    }

    desc_exp = {
        "Time": "*Time-based* explanations show what are the snippets of the audio that influenced the recommendation the most.",
        "Source": "*Source-based* explanations show what are the source components of the audio that influenced the recommendation the most.",
        "Time + Source": "*Time+Source-based* explanations combine both the Time-based and Source-based explanations."
    }
    st.header("Listenable Explanation Generation")
    st.write(
        'Select the type of listenable explanations you want to hear. The type is defined by the nature of the interpretable components that can be provided as explanations.')
    explanation_type = st.radio('What type of Explanation do you want to hear?', ['Time', 'Source', 'Time + Source'],
                                index=0)
    image_path = get_image_path(explanation_type)

    name = ' '.join(desc_exp[explanation_type].split(' ', 3)[:2])
    url = "https://sanders.cp.jku.at/share.cgi/{}?ssid=07Tk2FX&fid=07Tk2FX&path=%2Fdescriptions&filename={}&openfolder=normal&ep=".format(
        img_exp[explanation_type], img_exp[explanation_type])
    st.markdown("<img style='float: right;' src='{}'> **{}**<div><br>{} <br> </div> ".
                format(url, name, desc_exp[explanation_type]),
                unsafe_allow_html=True)
    # st.image(image_path, width=800)

    if st.button('Compute Explanation'):
        fidelity, interpretable_components, indexes, scores = generate_explanation(dp, explanation_type,
                                                                                   predicter, username, selected_idx,
                                                                                   test_set)

        show_badge('Fidelity of the explanations', 'fidelity', fidelity)

        print(scores)

        explanation_file_name = "{}_{}_{}_{}.ex.mp3"

        st.subheader('Top Highlight')
        st.markdown('What is the component of the audio that influenced the recommendation the most?')
        param = {'k': 1, 'enhance': False, 'query_instrument': None, 'explanation_type': explanation_type}

        top_components, sources_names, trim_f = take_top(interpretable_components, indexes, scores, param)
        file_name = explanation_file_name.format(track_name, explanation_type, username, str(param))
        mp3_path = get_mp3_path(file_name, test_set, 'explanations')
        npy_to_mp3(sum(top_components), mp3_path[:-4])  # todo fix this
        encode_image, height, offset = generate_encoded_waveplot(top_components, start=trim_f,
                                                                 source_names=sources_names)
        dir = os.path.join(test_set, 'explanations')
        print(external_file_template.format(file_name, test_set + '/explanations', file_name))
        audio_image_player(encode_image, external_file_template.format(file_name, test_set + '/explanations', file_name),
                           offset=offset)

        st.subheader('Top-3 Components')
        st.markdown('What are the top-3 components of the audio that influenced the recommendation the most?')
        param = {'k': 3, 'enhance': False, 'query_instrument': None, 'explanation_type': explanation_type}

        top_components, sources_names, trim_f = take_top(interpretable_components, indexes, scores, param)
        file_name = explanation_file_name.format(track_name, explanation_type, username, str(param))
        mp3_path = get_mp3_path(file_name, test_set, 'explanations')
        npy_to_mp3(sum(top_components), mp3_path[:-4])  # todo fix this
        encode_image, height, offset = generate_encoded_waveplot(top_components, start=trim_f,
                                                                 source_names=sources_names)
        dir = os.path.join(test_set, 'explanations')
        print(external_file_template.format(file_name, test_set + '/explanations', file_name))
        audio_image_player(encode_image, external_file_template.format(file_name, test_set + '/explanations', file_name),
                           height=height, bar_length=height, offset=offset)
