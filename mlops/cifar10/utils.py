import argparse
import random
import re
from typing import Dict

import numpy as np
import torch


class DictPairParser(argparse.Action):
    METAVAR = "KEY1=VALUE1,KEY2=VALUE2,KEY3=VALUE3"

    def __call__(self, parser, namespace, values, option_string=None):
        dict_: Dict[str, str] = {}
        for kv_string in re.split(r",|\s", values):
            k, v = re.split(r"\s?=\s?", kv_string)
            dict_[k] = v
        setattr(namespace, self.dest, dict_)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)