import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(x, file_path):
    return pickle.dump(x, open(file_path, 'wb'))


def generate_uid(prefix=None, postfix=None):
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)
    if not prefix is None:
        uid = prefix + "_" + uid
    if not postfix is None:
        uid = uid + "_" + postfix
    return uid


def get_user_id(name: str):
    # Check streamlit for a description of the users
    if name == 'marko':
        return "d7d2d888ae04d16e994d6964214a1de81392ee04"
    elif name == 'paige':
        return "0c2932cb475b83b61039bdfbb72c14580b8fad2b"
    elif name == 'johnny':
        return "6d625c6557df84b60d90426c0116138b617b9449"
    elif name == 'matteo':
        return "a05e548059abb1f77cad6cb9c3c0c48e0616f551"
    elif name == 'nina':
        return "57262c4ed3cb3ed2db7cab8c627091757c6437d8"
    elif name == 'elizabeth':
        return '18765abd13462c176d9ccc89e71bfc23265dfed7'
    elif name == 'sandra':
        return '3b93435988354b1889de1e71810d1dd65c4ba17c'
    else:
        raise ValueError('User not found!')


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_color(var_name, value):
    the_ranges = {
        'standard': [
            (0.0, 0.2, 'red'),
            (0.2, 0.4, 'orange'),
            (0.4, 0.6, 'yellow'),
            (0.6, 0.8, 'green'),
            (0.8, 1.0, 'brightgreen')
        ]
    }
    for v_min, v_max, color in the_ranges[var_name]:
        if v_min <= value <= v_max:
            return color

    raise ValueError('nothing found for {} and value {}'.format(var_name, value))


def msdid_to_path(msdid):
    return os.path.join(msdid[2], msdid[3], msdid[4], msdid + ".mp3")
