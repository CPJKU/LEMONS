import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

from recsys.experiment import ex

# Local experiment configuration #
local_conf = {
    'mongodb_url': 'mdb.cp.jku.at:27017',
    'mongodb_db_name': 'ajures_aggregated',
    'experiment_name': 'ajures_baseres',
}
exag = Experiment(local_conf['experiment_name'])
exag.observers.append(MongoObserver(url=local_conf['mongodb_url'], db_name=local_conf['mongodb_db_name']))


@exag.config
def experiment_config():
    # --Logging Parameters-- #
    users = ['marko', 'paige', 'johnny', 'matteo', 'nina', 'elizabeth', 'sandra']
    training_seeds = [1930289, 3049502, 88849483]


@exag.automain
def main(_config, _run):
    print(_config)
    users = _config['users']
    training_seeds = _config['training_seeds']

    seed_results = []
    for tr_seed in training_seeds:
        user_results = []
        for user in users:
            r = ex.run(config_updates={'user_name': user, 'seed': 1057386, 'training_seed': tr_seed,
                                       'model_type': 'baseres', 'macro_experiment_name': local_conf['experiment_name']})

            user_results.append(r.result)
        seed_results.append(np.mean(user_results))
    f = float(np.mean(seed_results))
    exag.log_scalar('mean2', f)
