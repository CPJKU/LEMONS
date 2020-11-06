import argparse
import os

import numpy as np
import torch

from conf.config import meta_path, track_path, audio_path
from explanations.lemons_utils import local_config, best_confs

# from audioLIME.factorization import SpleeterFactorization
from explanations.spleeter_precomputed import SpleeterPrecomputedFactorization as SpleeterFactorization
from utilities.utils import pickle_dump, msdid_to_path
from audioLIME import lime_audio
from audioLIME.data_provider import RawAudioProvider

from recsys.predict import Predict
from recsys.msd_user_loader import get_audio_loader

# composition_fn

from argparse import ArgumentParser

storage_path = '/share/cp/projects/ajures/stability/'
spleeter_sources_path = '/share/cp/projects/ajures/data/precomputed/'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=storage_path)
    parser.add_argument("--model_for_user", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_temporal_segments", type=int, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--num_explanations", type=int, default=50)
    parser.add_argument("--debug", action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    username = args.model_for_user

    # Modify configuration
    local_config = {**local_config,
                    "model_type": best_confs[username]['model_type'],
                    "model_load_path": best_confs[username]['model_load_path'],
                    "results_path": os.path.dirname(best_confs[username]['model_load_path']) + "/results.pkl",
                    "use_tensorboard": 0,
                    }

    num_explanations = args.num_explanations

    test_loader = get_audio_loader(track_path,
                                   meta_path,
                                   username,
                                   1,
                                   split='TEST',
                                   seed=local_config["data_seed"])

    length_samples = len(test_loader.dataset)
    print(length_samples)
    sample_indeces = np.array(range(length_samples))[::length_samples // num_explanations][:num_explanations]
    # TODO: check if test samples are random enough to use this way of selecting
    assert len(sample_indeces) == num_explanations

    path_experiments = args.out_dir
    k_components = args.k
    batch_size = args.batch_size
    num_samples = args.num_samples
    n_segments = args.n_temporal_segments

    if args.debug:
        sample_indeces = sample_indeces[:5]

    n_repeats = args.repeat

    sample_rate = 16000
    # model = Predict.get_model(argparse.Namespace(**local_config))
    predicter = Predict(test_loader, argparse.Namespace(**local_config))


    def model(x):
        x = torch.tensor(x).cuda()
        return predicter.model.predict(x)


    results = {}
    for i, sample_idx in enumerate(sample_indeces):
        print("Processing sample {}/{}".format(i + 1, len(sample_indeces)))
        results[sample_idx] = []
        x, y = test_loader.dataset[sample_idx]
        msdid = test_loader.dataset.fl[sample_idx]
        outputs = model(x)
        print("outputs", outputs)

        data_provider = RawAudioProvider(os.path.join(audio_path, msdid_to_path(msdid)))

        factorization = SpleeterFactorization(data_provider,
                                              n_temporal_segments=n_segments,
                                              composition_fn=None,
                                              model_name='spleeter:5stems',
                                              spleeter_sources_path=spleeter_sources_path
                                              )
        explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=True)

        # factorization.set_analysis_window(snippet_starts[sn], config.input_length)
        # print("mix length", len(factorization.data_provider.get_mix()))

        test_example_results = []
        for _ in range(n_repeats):
            explanation = explainer.explain_instance(factorization=factorization,
                                                     predict_fn=lambda x: np.column_stack((
                                                         1 - predicter.model.predict(x), predicter.model.predict(x))),
                                                     top_labels=1,
                                                     num_samples=num_samples,
                                                     batch_size=batch_size)

            label = list(explanation.local_exp.keys())[0]
            _, component_indeces = explanation.get_sorted_components(label,
                                                                     positive_components=True,
                                                                     negative_components=True,
                                                                     num_components=k_components,
                                                                     return_indeces=True)
            test_example_results.append(component_indeces)
        unique_components = set(np.concatenate(test_example_results))
        results[sample_idx].append(unique_components)

    prefix = "debug" if args.debug else "stability"
    print(results)
    results_path = os.path.join(path_experiments, "{}_{}_temp{}_rep{}_k{}_Ns{}.pt".
                                format(prefix, username,
                                       n_segments, n_repeats, k_components, num_samples))
    pickle_dump(results, results_path)
