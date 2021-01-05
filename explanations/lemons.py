import os

import numpy as np
import streamlit as st

from conf.config import web_folder
from explanations.lemons_utils import best_models, plot_sidebar, \
    load_musdb18_tracks, take_top, show_badge, generate_explanation, \
    user_descriptions, Users, load_msd_tracks, load_snippet, ExplanationTypes, \
    explanation_types_description, save_explanation, generate_audio_link
from explanations.lemons_utils import generate_encoded_waveplot, audio_image_player

plot_sidebar(st)

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

st.markdown(
    "# ![](https://i.ibb.co/zsTgKk9/stereo.png)   LEMONS: Listenable Explanations for Music recOmmeNder Systems")

# ------- Persona Selection ------- #

st.markdown(
    "## User/Persona Selection\n"
    "Below you can explore the 7 users/personas of our demo. Each user is characterized by a distinctive music preference.\n"
)

entries = [x.name + ' - ' + user_descriptions[x]['short'] for x in best_models.keys()]
user_name = st.selectbox("Which user?", entries, index=3, format_func=lambda x: x.capitalize()).split('-', 1)[0][:-1]
user = Users[user_name]

# ------- User profile ------- #

st.subheader('Selected user profile')

train_tracks = load_msd_tracks(user, 'train')

music_taste_description = 'The user listened to *{}* tracks (shown below sorted by playcount) for a total of *{}* listening events.'.format(
    len(train_tracks),
    train_tracks['playcount'].sum())
st.markdown("<img style='float: left;' src='{}'> <div> **{}** <br> {} <br> {} </div>".
            format(user_descriptions[user]['image'],
                   user_name.capitalize(),
                   user_descriptions[user]['full'].capitalize(), music_taste_description),
            unsafe_allow_html=True)

st.dataframe(train_tracks[train_tracks.playcount > 0])

# ------- Track Recommendation ------- #

st.markdown('## Track Recommendation')
st.markdown("Choose the music dataset from which to receive the recommendations.")

test_dataset = st.radio('Which music dataset?', ['musdb18', 'MSD'])
test_tracks = load_msd_tracks(user, 'test') if test_dataset == 'MSD' else load_musdb18_tracks(user)

st.markdown(
    'The top-10 recommended tracks from the dataset *{}* are the shown below sorted by relevance.'.format(test_dataset))
st.dataframe(test_tracks[:10])

# ------- Track Selection ------- #

st.write("Select a recommended track (sorted by relevance)")
selected_track = st.selectbox("Which track do you want to listen to?", np.concatenate(
    ([''], test_tracks['title'].values[:10] + ' - ' + test_tracks['artist'].values[:10])), index=1)

if selected_track:

    track_name, artist_name = selected_track.split(' - ', 1)
    selected_idx = test_tracks[(test_tracks.title == track_name) & (test_tracks.artist == artist_name)].index[0]

    file_name = '{}.mp3'.format(
        test_tracks.loc[selected_idx]['track']) if test_dataset == 'MSD' else '{} - {}.mp3'.format(artist_name,
                                                                                                   track_name)
    file_path = os.path.join(web_folder, test_dataset, 'snippets', file_name)

    audio = load_snippet(file_path, test_dataset)

    encode_image = generate_encoded_waveplot(audio)

    show_badge('Relevance', 'relevance', test_tracks.loc[selected_idx]['relevance'])

    audio_url = generate_audio_link(file_name, test_dataset)

    audio_image_player(encode_image, audio_url)

    # ------- Listenable Explanation ------- #

    st.header("Listenable Explanation Generation")
    st.write(
        'Select the type of listenable explanations you want to hear. The type is defined by the nature of the interpretable components that can be provided as explanations.')

    explanation_type_text = st.radio('What type of Explanation do you want to hear?',
                                     ['Time', 'Source', 'Time + Source'],
                                     index=2)
    explanation_type = ExplanationTypes[
        explanation_type_text if explanation_type_text != 'Time + Source' else 'TimeAndSource']

    st.markdown("<img style='float: right;' src='{}'> **{}**<div><br>{} <br> </div> "
                .format(explanation_types_description[explanation_type]['url'],
                        explanation_type_text,
                        explanation_types_description[explanation_type]['desc']),
                unsafe_allow_html=True)

    if not st.button('Compute Explanation'):
        fidelity, interpretable_components, indexes, scores = generate_explanation(file_path, explanation_type,
                                                                                   user_name, selected_idx,
                                                                                   test_dataset)

        show_badge('Fidelity of the explanations', 'fidelity', fidelity)

        # ------- Top Highlight ------- #
        st.subheader('Top Highlight')
        st.markdown('What is the component of the audio that influenced the recommendation the most?')
        param = {'k': 1,
                 'explanation_type': explanation_type,
                 'query_instrument': None}
        top_components, sources_names = take_top(interpretable_components, indexes, scores, param)
        exp_file_name = save_explanation(sum(top_components), track_name, explanation_type, user_name, str(param),
                                         test_dataset)
        encode_image = generate_encoded_waveplot(top_components, source_names=sources_names)

        audio_url = generate_audio_link(exp_file_name, test_dataset, is_explanation=True)
        audio_image_player(encode_image, audio_url, sources_names)

        # ------- Top-3 Components ------- #
        st.subheader('Top-3 Components')
        st.markdown('What are the top-3 components of the audio that influenced the recommendation the most?')
        param = {'k': 3,
                 'explanation_type': explanation_type,
                 'query_instrument': None}

        top_components, sources_names = take_top(interpretable_components, indexes, scores, param)

        exp_file_name = save_explanation(sum(top_components), track_name, explanation_type, user_name, str(param),
                                         test_dataset)

        encode_image = generate_encoded_waveplot(top_components, source_names=sources_names)

        audio_url = generate_audio_link(exp_file_name, test_dataset, is_explanation=True)
        audio_image_player(encode_image, audio_url, sources_names)
