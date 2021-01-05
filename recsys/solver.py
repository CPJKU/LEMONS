import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

from recsys.model import FCN
from utilities.experiment_utils import SacredLogger, Tracker, evaluate, update, TensorBoardLogger


class Solver(object):
    def __init__(self, train_loader, val_loader, config):

        # --Logging Parameters-- #
        self.log_step = config.log_step
        self.use_tensorboard = config.use_tensorboard
        self.model_save_path = config.model_save_path

        # --Training Parameters-- #
        self.model_load_path = config.model_load_path
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.wd = config.wd
        self.device = torch.device(config.device)

        # --Data Parameters-- #
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Build model
        self.model = None
        self.optimizer = None
        self.build_model()

    def build_model(self):

        self.model = FCN(n_class=1)
        self.model = self.model.to(self.device)

        # Loading pre-trained model if specified
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.wd)

    def load(self, filename):

        dic = torch.load(filename, map_location=self.device)
        if 'spec.mel_scale.fb' in dic.keys():
            self.model.spec.mel_scale.fb = dic['spec.mel_scale.fb']

        self.model.load_state_dict(dic, strict=False)

    def save(self, filename):
        path = os.path.join(self.model_save_path, filename)
        model = self.model.state_dict()
        torch.save(model, path)

    def train(self, ex):
        reconst_loss = nn.BCEWithLogitsLoss()

        loggers = {SacredLogger(ex)}
        if self.use_tensorboard:
            loggers.add(TensorBoardLogger())

        tr_trk = Tracker(*loggers, pre_tag='train', log_every=self.log_step)
        vd_trk = Tracker(*loggers, pre_tag='val', log_every=self.log_step,
                         metrics=[metrics.roc_auc_score, metrics.average_precision_score],
                         metrics_names=['roc_auc', 'avg_prec'])

        # Baseline evaluation
        evaluate(self.model, self.train_loader, reconst_loss, tr_trk)

        best_metric = np.inf
        for epoch in range(self.n_epochs):
            if epoch % 5 == 0:
                print("[%s] Epoch [%d/%d]" % (
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, self.n_epochs))

            update(self.model, self.train_loader, reconst_loss, self.optimizer, tr_trk)
            val_loss, _ = evaluate(self.model, self.val_loader, reconst_loss, vd_trk)

            if val_loss < best_metric:
                best_metric = val_loss
                print("[%s] Epoch [%d/%d] New Best Model Found! best val_loss is %f" % (
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, self.n_epochs, best_metric))
                self.save('best_model.pth')
        return best_metric
