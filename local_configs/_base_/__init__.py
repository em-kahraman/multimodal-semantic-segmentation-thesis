import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C

C.seed = 12345

# remoteip = os.popen('pwd').read()
C.root_dir = "datasets"  # os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")


def setup_paths(C, exp_name):
    """Set the log and checkpoint paths of an experiment and create its log directory."""
    C.log_dir = osp.abspath(osp.join("checkpoints", C.dataset_name + "_" + exp_name))
    C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
    C.log_dir_link = C.log_dir
    C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
    os.makedirs(C.log_dir, exist_ok=True)
    exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    C.log_file = C.log_dir + "/log_" + exp_time + ".log"
    C.link_log_file = C.log_file + "/log_last.log"
    C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
    C.link_val_log_file = C.log_dir + "/val_last.log"
