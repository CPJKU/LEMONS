import argparse
import os

import numpy as np
from tqdm import tqdm

from conf.config import meta_path, track_path, musdb_storage
from explanations.lemons_utils import best_confs, local_config, load_track_dataframe, load_musdb_track_dataframe, \
    TemporalFactorization, \
    get_mp3_path, get_track_name
from recsys.msd_user_loader import get_msd_audio_loader
from recsys.musdb_data_loader import get_musdb18_audio_loader, compute_start_length
from recsys.predict import Predict
from utilities.utils import get_user_id, pickle_dump

from audioLIME.data_provider import RawAudioProvider
from explanations.spleeter_precomputed import SpleeterPrecomputedFactorization as SpleeterFactorization
from audioLIME.lime_audio import LimeAudioExplainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--testset", type=str, required=True, choices=['MSD', 'musdb'])
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--results_root", type=str, default=".")
    parser.add_argument("--take_center", action='store_true', default=False)
    parser.add_argument("--snippet_length", type=int, default=None)

    args = parser.parse_args()

    username = args.username
    testset = args.testset
    top_n = args.top_n
    results_root = args.results_root
    take_center = args.take_center
    snippet_length = args.snippet_length

    if testset == 'MSD':
        print("take_center is ignored msd")
        if snippet_length is not None:
            print("snippet_length is ignored for msd")

    print('Working on user ' + username)
    user_id = get_user_id(username)

    local_config = {**local_config,
                    "model_type": best_confs[username]['model_type'],
                    "model_load_path": best_confs[username]['model_load_path'],
                    "results_path": os.path.dirname(best_confs[username]['model_load_path']) + "/results.pkl",
                    "use_tensorboard": 0,
                    }

    user_store_folder = os.path.join(results_root, 'stored_explanations', username)
    if not os.path.isdir(user_store_folder):
        os.mkdir(user_store_folder)

    if testset == 'MSD':
        test_loader = get_msd_audio_loader(track_path,
                                           meta_path,
                                           username,
                                           1,
                                           split='TEST',
                                           seed=local_config["data_seed"])
    else:
        test_loader = get_musdb18_audio_loader("/share/cp/datasets/musdb18/",
                                               cache_path=musdb_storage,
                                               batch_size=1,
                                               take_center=take_center,
                                               load_audio=False,
                                               snippet_length=snippet_length)

    predicter = Predict(test_loader, argparse.Namespace(**local_config))

    if testset == 'MSD':
        track_dataframe = load_track_dataframe(test_loader, local_config)
    else:
        track_dataframe = load_musdb_track_dataframe(test_loader, username, take_center, snippet_length)

    # Only the top-n
    top_n_df = track_dataframe[:top_n]

    explainer = LimeAudioExplainer(verbose=False, absolute_feature_sort=False)

    for selected_idx in tqdm(top_n_df.track.index, desc='Tracks'):
        file_path = os.path.join(user_store_folder, '{}_{}.pkl'.format(testset, selected_idx))
        if os.path.isfile(file_path):
            print('Already computed for {}'.format(selected_idx))
            continue

        track_explanations = dict()

        # todo: make generic enough for musdb
        track_name = get_track_name(test_loader, testset, selected_idx)
        mp3_path = get_mp3_path(track_name, testset, 'snippets')

        # input_logit = predicter.model.predict(x, logits=True)

        for style in ['time', 'source', 'time_source']:

            dp = RawAudioProvider(mp3_path)  ## todo: is this necessary?
            x = dp.get_mix()
            print("x", len(x))

            if take_center and testset == 'musdb':
                test_start, test_length = compute_start_length(x, snippet_length, 16000)  # take from config
                print("test_start, test_length", test_start, test_length)
                dp.set_analysis_window(test_start, test_length)

            if style == 'time':
                factorization = TemporalFactorization(dp, n_temporal_segments=5,
                                                      composition_fn=None)
            elif style == 'source':
                factorization = SpleeterFactorization(dp, n_temporal_segments=1,
                                                      composition_fn=None,
                                                      model_name='spleeter:5stems')
            else:
                factorization = SpleeterFactorization(dp, n_temporal_segments=5,
                                                      composition_fn=None,
                                                      model_name='spleeter:5stems')

            explanation = explainer.explain_instance(factorization=factorization,
                                                     predict_fn=lambda x: np.column_stack((
                                                         1 - predicter.model.predict(x), predicter.model.predict(x))),
                                                     top_labels=1,
                                                     num_samples=2048,
                                                     batch_size=10
                                                     )

            label = list(explanation.score.keys())[0]
            input_scores = np.array(sorted(explanation.local_exp[label], key=lambda x: x[0]))
            input_components = np.array(factorization.retrieve_components())
            fidelity = explanation.score[label]
            track_explanations[style] = {'components': input_components, 'scores': input_scores, 'fidelity': fidelity}
        pickle_dump(track_explanations, file_path)

        # Save track explantions

        # user_results.setdefault(selected_idx, []).append(track_results)

    # pickle_dump(user_results, os.path.join(results_root,
    #       'stored_explanations/{}_{}_top_{}_explanations.pkl'.format(testset,
    #                                                                 username, top_n)))
