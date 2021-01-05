# coding: utf-8
import datetime
import os

import torch
import torch.nn as nn
from sklearn import metrics

from recsys.model import FCN
from utilities.experiment_utils import TensorBoardLogger, SacredLogger, Tracker, evaluate
from utilities.utils import pickle_dump


class Predict(object):
    def __init__(self, test_loader, config):

        # --Logging Parameters-- #
        self.use_tensorboard = config.use_tensorboard

        # --Evaluation Parameters-- #
        self.model_load_path = config.model_load_path
        self.results_path = config.results_path if 'results_path' in config else \
            os.path.dirname(self.model_load_path) + "/MSD_{}.pkl".format(config.user_name)
        self.device = torch.device(config.device)

        # --Data Parameters-- #
        self.test_loader = test_loader

        # Build model
        self.model = None
        self.build_model()

    def build_model(self):

        self.model = FCN(n_class=1)
        self.model = self.model.to(self.device)

        # load model
        self.load(self.model_load_path)

    def load(self, filename):
        dic = torch.load(filename, map_location=self.device)
        if 'spec.0.mel_scale.fb' in dic.keys():
            self.model.spec[0].mel_scale.fb = dic['spec.0.mel_scale.fb']
        self.model.load_state_dict(dic)

    def test(self, ex):

        reconst_loss = nn.BCEWithLogitsLoss()

        loggers = {SacredLogger(ex)}
        if self.use_tensorboard:
            loggers.add(TensorBoardLogger())

        tracker = Tracker(*loggers, pre_tag='test', log_every=-1,
                          metrics=[metrics.roc_auc_score, metrics.average_precision_score],
                          metrics_names=['roc_auc', 'avg_prec'])

        test_loss, preds = evaluate(self.model, self.test_loader, reconst_loss, tracker)
        print("[%s] Finished Testing! test_loss is %f" % (
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), test_loss))

        # Save results
        pickle_dump(preds, self.results_path)
