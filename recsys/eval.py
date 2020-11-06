# coding: utf-8
import argparse
import os

import torch
from sacred import Experiment
from sacred.observers import MongoObserver

from conf.config import meta_path, track_path
from recsys.msd_user_loader import get_msd_audio_loader
from recsys.predict import Predict
from utilities.utils import get_user_id

# Local experiment configuration #
local_conf = {
    'mongodb_url': 'mdb.cp.jku.at:27017',  # TODO: To remove
    'mongodb_db_name': 'alessandro_ajures_test',  # TODO: TO remove
    'experiment_name': 'LEMONS',
}

ex = Experiment(local_conf['experiment_name'])
ex.observers.append(MongoObserver(url=local_conf['mongodb_url'], db_name=local_conf['mongodb_db_name']))


@ex.config
def experiment_config():
    # --Logging Parameters-- #

    use_tensorboard = 1  # if also tensorboard (together with sacred) should be used

    # --Evaluation Parameters-- #

    input_length = 16000  # 1 second
    model_load_path = ''  # path to the trained model
    results_path = os.path.dirname(model_load_path) + "/results.pkl"
    batch_size = 20  # batch size
    num_workers = 10  # number of workers
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # which device to use

    # --Data Parameters-- #

    data_path = track_path  # path to the npys
    meta_path = meta_path  # path to the meta data
    user_name = 'elizabeth'  # users (check utils, get_user_id)
    user_id = get_user_id(user_name)


@ex.automain
def main(_config):
    print(_config)
    conf = argparse.Namespace(**_config)

    test_loader = get_msd_audio_loader(
        conf.data_path,
        conf.meta_path,
        conf.user_name,
        batch_size=1,
        split_set='TEST',
        input_length=conf.input_length,
        num_workers=conf.num_workers
    )

    p = Predict(test_loader, conf)
    p.test(ex)
