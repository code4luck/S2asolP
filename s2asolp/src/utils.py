import numpy as np
import pandas as pd
import argparse


def get_diff_args(args, mode="optim-"):
    mode_args = argparse.Namespace()
    mode_len = len(mode)
    for k, v in vars(args).items():
        if k.startswith(mode):
            setattr(mode_args, k[mode_len:], v)
    return mode_args
