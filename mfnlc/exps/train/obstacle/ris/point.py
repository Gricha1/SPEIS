import argparse

import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.obstacle.ris.base import train


def learn(args):
    train(env_name="GCPoint",
          total_timesteps=args.total_timesteps,
          learning_starts=10_000,
          action_noise=None,
          new_policy_kwargs={"net_arch": [256, 256]},
          policy_to_delete_kwargs={"net_arch": [100, 100]},
          train_freq=(1, "episode"), #train_freq=(200, "step"),
          gradient_steps=1, #gradient_steps=100,
          h_lr=1e-4, # RIS
          q_lr=1e-3, # RIS
          pi_lr=1e-4, # RIS
          epsilon=1e-16, # RIS
          alpha=args.alpha, # RIS
          Lambda=0.05, # RIS
          n_ensemble=20, # RIS
          clip_v_function=-150, # RIS,
          buffer_size_her=args.buffer_size_her, # HER
          fraction_goals_are_rollout_goals=0.2, # HER
          fraction_resampled_goals_are_env_goals=0.0, # HER
          fraction_resampled_goals_are_replay_buffer_goals=0.5, # HER
          no_safety=args.no_safety,
          cost_limit=args.cost_limit,
          lambda_initialization=args.lambda_initialization,
          train_sac=args.train_sac,
          critic_max_grad_norm=None, # RIS
          actor_max_grad_norm=None, # RIS
          use_one_safe_critic=args.use_one_safe_critic,
          add_subgoal_reinforce_sg_num=args.add_subgoal_reinforce_sg_num,
          sgg_optimizing=args.sgg_optimizing,
          safe_critic_behave=args.safe_critic_behave,
          n_envs=1,
          batch_size=args.batch_size,
          log_interval=4,
          validate_freq=args.validate_freq,
          validate=args.validate,
          validate_robot_video=args.validate_robot_video,
          validate_subgoal_video=args.validate_subgoal_video,
          validate_video_idx=args.validate_video_idx,
          load_model=args.load_model,
          load_model_folder=args.load_model_folder,
          obs_noise_std=args.obs_noise_std,
          action_noise_std=args.action_noise_std)


def evaluate_controller():
    inspect_training_simu(env_name="GCPoint",
                          algo="e2e",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_one_safe_critic", action='store_true', default=False)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--load_model", action='store_true', default=False)
    parser.add_argument("--load_model_folder", default="", type=str)
    parser.add_argument("--total_timesteps", default=2_000_000, type=int)
    parser.add_argument("--validate_freq", default=10_000, type=int)
    parser.add_argument("--validate_robot_video", action='store_true', default=False)
    parser.add_argument("--validate_subgoal_video", action='store_true', default=False)
    parser.add_argument("--validate_video_idx", default=0, type=int)
    # subgoal policy 
    parser.add_argument("--add_subgoal_reinforce_sg_num", default=0, type=int)
    parser.add_argument("--sgg_optimizing", action='store_true', default=False)
    # SPEIS
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--lambda_initialization", default=0.5, type=float)
    parser.add_argument("--buffer_size_her", default=500_000, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    # sac
    parser.add_argument("--train_sac", action='store_true', default=False)
    parser.add_argument("--no_safety", action='store_true', default=False)
    parser.add_argument("--safe_critic_behave", default="min", type=str)
    parser.add_argument("--cost_limit", default=3.0, type=float)
    # noise
    parser.add_argument("--obs_noise_std", default=0.0, type=float, help="Standard deviation of Gaussian noise added to observations")
    parser.add_argument("--action_noise_std", default=0.0, type=float, help="Standard deviation of Gaussian noise added to actions")
    args = parser.parse_args()    
    
    assert args.safe_critic_behave in ["min", "max", "mean"]
    learn(args)
    #evaluate_controller()
