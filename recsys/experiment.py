import argparse
import os

import torch
from sacred import Experiment
from sacred.observers import MongoObserver

from conf.config import track_path, meta_path
from recsys.msd_user_loader import get_msd_audio_loader
from recsys.solver import Solver
from utilities.utils import generate_uid, get_user_id, reproducible

# Local experiment configuration #
local_conf = {
    'mongodb_url': 'mdb.cp.jku.at:27017',  # TODO: To remove
    'mongodb_db_name': 'ajures',  # TODO: To remove
    'experiment_name': 'LEMONS',
}
ex = Experiment(local_conf['experiment_name'])
ex.observers.append(MongoObserver(url=local_conf['mongodb_url'], db_name=local_conf['mongodb_db_name']))


# Configuration for the Experiments
@ex.config
def experiment_config():
    # --Logging Parameters-- #

    uid = generate_uid()
    model_save_path = '../experiments/{}/'.format(uid)
    use_tensorboard = 0  # if also tensorboard (together with sacred) should be used
    log_step = 100  # how many batches have to pass before logging the batch loss (NB. this is not for avg_loss)

    # --Training Parameters-- #

    training_seed = 1930289  # seed used for recsys (independent of the data seed)
    input_length = 16000  # 1 second
    model_load_path = ''  # if load pre-trained model
    batch_size = 20  # batch size
    n_epochs = 1000  # epochs for recsys
    lr = 1e-3  # learning rate
    wd = 1e-4  # weight decay
    num_workers = 0  # number of workers
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # which device to use

    # --Data Parameters-- #

    data_path = track_path  # path to the npys
    meta_path = meta_path  # path to the meta data
    user_name = 'marko'  # users (check utils, get_user_id)
    user_id = get_user_id(user_name)

    macro_experiment_name = ''  # used only when running multiple experiments for the architecture


@ex.automain
def main(_config):
    print(_config)
    conf = argparse.Namespace(**_config)

    # path for models
    if not os.path.exists(conf.model_save_path):
        os.makedirs(conf.model_save_path)

    # Get data
    train_loader = get_msd_audio_loader(
        conf.data_path,
        conf.meta_path,
        conf.user_name,
        batch_size=conf.batch_size,
        split_set='TRAIN',
        input_length=conf.input_length,
        num_workers=conf.num_workers
    )

    val_loader = get_msd_audio_loader(
        conf.data_path,
        conf.meta_path,
        conf.user_name,
        batch_size=1,
        split_set='VAL',
        input_length=conf.input_length,
        num_workers=conf.num_workers
    )

    # Ensuring reproducibility
    reproducible(conf.training_seed)

    solver = Solver(train_loader, val_loader, conf)
    return solver.train(ex)
