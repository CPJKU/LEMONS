import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utilities.utils import pickle_load

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_path", type=str, default="/share/cp/projects/ajures/")
    parser.add_argument("--prefix", type=str, default="stability")
    # parser.add_argument("--model_user", type=str, required=True)
    parser.add_argument("--n_temporal_segments", type=int, required=True)
    args = parser.parse_args()

    results_path = args.results_path
    # model_user = args.model_user
    prefix = args.prefix
    n_temporal_segments = args.n_temporal_segments

    figure_path = os.path.join(results_path, "figures", "{}.png".format(prefix))
    results_path = os.path.join(results_path, "stability")

    results_list = [os.path.join(results_path, res_file) for res_file in os.listdir(results_path)
                    if res_file.startswith(prefix)]
    results = dict([(res_file, pickle_load(res_file)) for res_file in results_list])

    print(results)
    columns = ["stability", 'model_user', 'Ns']
    data = pd.DataFrame(None, columns=columns)
    for k, v in results.items():
        stability_scores = []
        file_name = os.path.basename(k).replace(".pt", "")
        file_name = file_name.replace("temp", "").replace("Ns", "")
        _, model_user, temp, _, _, Ns = file_name.split("_")

        # results_path = os.path.join(path_experiments, "{}_{}_temp{}_rep{}_k{}_Ns{}.pt".
        #                             format(prefix, username,
        #                                    n_segments, n_repeats, k_components, num_samples))

        if int(temp) != n_temporal_segments:
            print("skipping", model_user, Ns, temp)
            continue

        print("adding", model_user, Ns, temp)

        for _, v1 in v.items():
            print(v1)
            if len(v1) > 1:
                v1 = set(np.concatenate(v1))
                print(v1)
            assert len(v1) == 1, "currently only 1 snippet per sample"
            # print(v1[0])
            stability_scores.append((len(v1[0]), model_user, int(Ns)))
        data = data.append(pd.DataFrame(stability_scores, columns=columns))

    print(data)
    data["stability"] = data["stability"].astype(float)
    print(data.groupby(["model_user", "Ns"]).agg(["mean", "median", 'count']))

    sns.violinplot(x="Ns", y="stability", hue="model_user", data=data)
    plt.title("Stability")
    plt.savefig(figure_path)
    plt.show()
