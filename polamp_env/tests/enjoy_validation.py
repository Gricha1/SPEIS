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
from lib.envs import POLAMPEnvironment
from lib.utils_operations import generateDataSet

use_wandb = False

def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).
    """
    # name_sf_policy = "dyn_correct_from_scratch"
    # name_sf_policy = "true_actions"
    # name_sf_policy = "polamp_safety_layer"
    name_sf_policy = "polamp_wo_steering"
    name_sf_policy = "polamp_wsteering_acc"
    name_sf_policy = "polamp_clearance_collision_5"
    # name_sf_policy = "polamp_refactoring"
    argv = [f'--algo={"APPO"}', f'--env={"polamp_env"}', f'--experiment={name_sf_policy}']
    parser = arg_parser(argv, evaluation=evaluation)

    # add custom args here
    parser.add_argument('--my_custom_arg', type=int, default=42, help='Any custom arguments users might define')
    # SampleFactory parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    cfg.evaluation = evaluation
    # print(f"config {cfg}")
    # cfg.encoder_type = 'mlp'
    # cfg.encoder_subtype = 'mlp_mujoco'
    # cfg.rollout = 1
    # cfg.cpc_forward_steps = 1
    # cfg.recurrence = 1
    # cfg.use_rnn = 0
    # cfg.encoder_extra_fc_layers = 0
    cfg.encoder_type = 'mlp'
    cfg.encoder_subtype = 'mlp_mujoco'
    cfg.encoder_extra_fc_layers = 0
    cfg.train_for_env_steps = 400000000
    # cfg.custom_env_episode_len = 250
    # cfg.num_workers = 5
    # # cfg.policy_workers_per_policy=2
    # cfg.num_envs_per_worker = 1
    # cfg.batch_size = 256
    cfg.use_rnn = True
    cfg.rnn_num_layers = 1
    cfg.batch_size = 2048
    # cfg.ppo_clip_ratio = 0.3
    cfg.rollout = 128
    cfg.ppo_epochs = 1
    cfg.ppo_clip_value = 5.0
    cfg.reward_scale = 0.1
    cfg.reward_clip = 5.0
    cfg.with_vtrace = False
    # cfg.ppo_epochs = 1
    # cfg.actor_critic_share_weights = True
    cfg.nonlinearity = 'relu'
    # cfg.ppo_epochs = 10
    # cfg.max_policy_lag = 1000000
    cfg.exploration_loss_coeff = 0.005
    # cfg.max_grad_norm = 0.0
    cfg.kl_loss_coeff = 0.3
    cfg.value_loss_coeff = 0.5
    # cfg.reward_clip = 100
    cfg.use_wandb = use_wandb
    cfg.with_wandb = use_wandb
    # cfg.experiment_summaries_interval = 5

    return cfg

def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    dataSet = generateDataSet(our_env_config)
    # maps, trainTask, valTasks = dataSet["empty"]
    maps, trainTask, valTasks = dataSet["obstacles"]
    # maps, trainTask, valTasks = dataSet["dyn_obstacles"]
    # maps = {}
    # trainTask = {}
    # valTasks = {}
    
    # if not our_env_config["empty"]:
    #     maps = maps_obst
    #     trainTask = trainTask_obst
    #     valTasks = valTasks_obst
    # if not our_env_config["obstacles"]:
    #     maps = maps_dyn_obst
    #     trainTask = trainTask_dyn_obst
    #     valTasks = valTasks_dyn_obst

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
    # if not use_wandb:
    #     print(f"cfg {cfg}")
    return POLAMPEnvironment(full_env_name, cfg['other_keys'])


def add_extra_params_func(env, parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    # p.add_argument('--custom_env_episode_len', default=250, type=int, help='Number of steps in the episode')
    # p.env_framestack = 1
    # p.encoder_type = 'mlp'
    # p.encoder_subtype = 'mlp_mujoco'
    
    # dataSet = generateDataSet(our_env_config)
    # maps, trainTask, valTasks = dataSet["empty"]
    # maps_obst, trainTask_obst, valTasks_obst = dataSet["obstacles"]
    # maps_dyn_obst, trainTask_dyn_obst, valTasks_dyn_obst = dataSet["dyn_obstacles"]
    
    # if not our_env_config["empty"]:
    #     maps = maps_obst
    #     trainTask = trainTask_obst
    #     valTasks = valTasks_obst
    # if not our_env_config["obstacles"]:
    #     maps = maps_dyn_obst
    #     trainTask = trainTask_dyn_obst
    #     valTasks = valTasks_dyn_obst

    # environment_config = {
    #     'vehicle_config': car_config,
    #     'tasks': trainTask,
    #     'valTasks': valTasks,
    #     'maps': maps,
    #     'our_env_config' : our_env_config,
    #     'reward_config' : reward_config
    # }

    # p.other_keys = environment_config


# def override_default_params_func(env, parser):
#     """
#     Override default argument values for this family of environments.
#     All experiments for environments from my_custom_env_ family will have these parameters unless
#     different values are passed from command line.
#     """
#     parser.set_defaults(
#         encoder_custom='custom_env_encoder',
#         hidden_size=128,
#     )

def polamp_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    # score = np.mean(policy_avg_stats["Score"])
    # log.debug(f'Score: {round(float(score), 3)}')
    # summary_writer.add_scalar('Score', score, env_steps)

    # num_achievements = np.mean(policy_avg_stats["Num_achievements"])
    # log.debug(f'Num_achievements: {round(float(num_achievements), 3)}')
    # summary_writer.add_scalar('Num_achievements', num_achievements, env_steps)
    print(f"policy_id : {policy_id}")
    # print(f"policy_avg_stats : {policy_avg_stats}")
    # print(f"env_steps : {env_steps}")
    # print(f"summary_writer : {summary_writer}")

def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='polamp_env',
        make_env_func=make_custom_env_func,
        add_extra_params_func=add_extra_params_func,
        # override_default_params_func=override_default_params_func,
    )
    EXTRA_PER_POLICY_SUMMARIES.append(polamp_extra_summaries)

with open("configs/train_configs.json", 'r') as f:
    train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)
    # print(our_env_config)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)


def main():
    register_custom_components()
    cfg = custom_parse_args()
    # print(f"cfg: {cfg}")
    if use_wandb:
        wandb.init(config=cfg, project='sample-factory-POLAMP', entity='brian_angulo', save_code=False, sync_tensorboard=True)
    
    status = run_algorithm(cfg)
    return status


# if __name__ == '__main__':
#     sys.exit(main())