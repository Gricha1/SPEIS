"""
From the root of Sample Factory repo this can be run as:
python training_sample_factory.py --algo=APPO --env=polamp_env --experiment=example
After training for a desired period of time, evaluate the policy by running:
python enjoy_sample_factory.py --algo=APPO --env=polamp_env --experiment=example
"""

from pickle import TRUE
import sys
import multiprocessing
from argparse import Namespace
import gym
import json
import wandb
import yaml
import numpy as np
from torch import nn
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape, nonlinearity
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm
# from EnvLib.ObstGeomEnvSampleFactory import *
from lib.envs import POLAMPEnvironment
from lib.utils_operations import generateDataSet

use_wandb = True

def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).
    """
    parser = arg_parser(argv, evaluation=evaluation)

    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    cfg.evaluation = evaluation
    cfg.encoder_type = 'mlp'
    cfg.encoder_subtype = 'mlp_mujoco'
    cfg.encoder_extra_fc_layers = 0
    cfg.train_for_env_steps = 50000000
    cfg.use_rnn = True
    cfg.rnn_num_layers = 1
    cfg.batch_size = 2048
    cfg.rollout = 128
    cfg.ppo_epochs = 1
    cfg.ppo_clip_value = 5.0
    cfg.reward_scale = 0.1
    cfg.reward_clip = 5.0
    cfg.with_vtrace = False
    cfg.nonlinearity = 'relu'
    cfg.exploration_loss_coeff = 0.005
    cfg.kl_loss_coeff = 0.3
    cfg.value_loss_coeff = 0.5
    cfg.use_wandb = use_wandb
    cfg.with_wandb = use_wandb

    return cfg

def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    dataSet = generateDataSet(our_env_config, name_folder="safety", total_maps=1, dynamic=False)
    maps, trainTask, valTasks = dataSet["obstacles"]

    environment_config = {
        'vehicle_config': car_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        'evaluation': cfg.evaluation,
    }

    cfg.other_keys = environment_config
    return POLAMPEnvironment(full_env_name, cfg['other_keys'])


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='polamp_env',
        make_env_func=make_custom_env_func,
    )

with open("configs/train_configs.json", 'r') as f:
    train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

def main():
    register_custom_components()
    cfg = custom_parse_args()

    if use_wandb:
        wandb.init(config=cfg, project='sample-factory-POLAMP', entity='brian_angulo', save_code=False, sync_tensorboard=True)
    
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())