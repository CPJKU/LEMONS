# coding: utf-8
from math import sqrt

import torch
import torch.nn as nn
import torchaudio


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight, sqrt(2))
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class FCN(nn.Module):
    """
    Fully Convolutional Neural Network.
    """

    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=256,
                 n_class=1):
        super(FCN, self).__init__()

        self.spec = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                 n_fft=n_fft,
                                                 f_min=f_min,
                                                 f_max=f_max,
                                                 n_mels=n_mels),
            torchaudio.transforms.AmplitudeToDB()
        )
        self.spec_bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(0.5)

        self.layer1 = Conv_2d(1, 64, pooling=2)
        self.layer2 = Conv_2d(64, 128, pooling=2)
        self.layer3 = Conv_2d(128, 128, pooling=2)
        self.layer4 = Conv_2d(128, 128, pooling=2)
        self.layer5 = Conv_2d(128, 64, pooling=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64 * 2, n_class)

        self.apply(init_weights)

    def forward(self, x):
        # [batch_size, 16000 (1 sec)]
        x = self.spec(x)
        # [batch_size, 256, 63]

        x = x.unsqueeze(1)
        x = self.spec_bn(x)
        # [batch_size, 1, 256, 63]

        x = self.layer1(x)
        # [batch_size, 64, 128, 31]
        x = self.layer2(x)
        # [batch_size, 128, 64, 15]
        x = self.layer3(x)
        # [batch_size, 128, 32, 7]
        x = self.layer4(x)
        # [batch_size, 128, 16, 3]
        x = self.layer5(x)
        # [batch_size, 64, 8, 1]

        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)

        x = torch.cat([x1, x2], dim=1)

        # [batch_size, 128, 1, 1]
        x = x.squeeze(2).squeeze(2)

        # [batch_size, 128]
        x = self.dropout(x)
        x = self.fc(x)
        # [batch_size, 1]

        return x

    @torch.no_grad()
    def predict(self, x, logits=False):
        self.eval()
        device = next(self.parameters()).device
        x = torch.tensor(x).to(device)
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        x = self.forward(x)
        if not logits:
            x = torch.nn.Sigmoid()(x)
        return x.detach().cpu().squeeze()
