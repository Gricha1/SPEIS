import os
import time
from pathlib import Path
import shutil
import pathlib
import json
import tempfile

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import wandb
import gym
from gym.envs.registration import register
import comet_ml
try:
    import imageio
except ImportError:
    imageio = None

from utils.logger import Logger
from polamp_RIS import RIS
from polamp_env.lib.utils_operations import generateDataSet


if __name__ == "__main__":	
    parser = argparse.ArgumentParser()

    # validation
    parser.add_argument('--not_visual_validation', default=False, action='store_true')
    parser.add_argument('--plot_trajectory', default=False, action='store_true', help='Plot trajectory without agent visualization')
    parser.add_argument('--with_noise', default=False, action='store_true', help='Add noise to observations and actions')
    parser.add_argument('--obs_noise_std', default=0.01, type=float, help='Standard deviation for observation noise')
    parser.add_argument('--action_noise_std', default=0.01, type=float, help='Standard deviation for action noise')

    # environment
    parser.add_argument("--env",                  default="polamp_env")
    parser.add_argument("--test_env",             default="polamp_env")
    parser.add_argument("--dataset",              default="cross_dataset_test_level_2")
    parser.add_argument("--dataset_curriculum",   default=False) # medium dataset -> hard dataset
    parser.add_argument("--dataset_curriculum_treshold", default=0.95, type=float) # medium dataset -> hard dataset
    parser.add_argument("--uniform_feasible_train_dataset", default=False)
    parser.add_argument("--random_train_dataset",           default=False)
    parser.add_argument("--train_sac",            default=False, type=bool)
    # ris
    parser.add_argument("--epsilon",            default=1e-16, type=float)
    parser.add_argument("--n_critic",           default=2, type=int) # 1
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=int(3e4), type=int) # 3e4
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--batch_size",         default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=5e5, type=int) # 5e5
    parser.add_argument("--n_eval",             default=5, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int) # 42
    parser.add_argument("--exp_name",           default="RIS_ant")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float) # 0.1
    parser.add_argument("--n_ensemble",         default=20, type=int) # 10
    parser.add_argument("--use_dubins_filter",  default=False, type=bool) # 10
    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-4, type=float)
    parser.add_argument("--clip_v_function",    default=-150, type=float) # -368
    parser.add_argument("--add_obs_noise",           default=False, type=bool)
    parser.add_argument("--curriculum_alpha_val",        default=0, type=float)
    parser.add_argument("--curriculum_alpha_treshold",   default=500000, type=int) # 500000
    parser.add_argument("--curriculum_alpha",        default=False, type=bool)
    parser.add_argument("--curriculum_high_policy",  default=False, type=bool)
    parser.add_argument("--max_grad_norm",              default=6.0, type=float)
    parser.add_argument("--scaling",              default=1.0, type=float)
    parser.add_argument("--lambda_initialization",  default=0.1, type=float)
    
    # her
    parser.add_argument("--fraction_goals_are_rollout_goals",  default=0.2, type=float) # 20
    parser.add_argument("--fraction_resampled_goals_are_env_goals",  default=0.0, type=float) # 20
    parser.add_argument("--fraction_resampled_goals_are_replay_buffer_goals",  default=0.5, type=float) # 20
    # encoder
    parser.add_argument("--use_decoder",             default=True, type=bool)
    parser.add_argument("--use_encoder",             default=True, type=bool)
    parser.add_argument("--state_dim",               default=40, type=int) # 20
    # safety
    parser.add_argument("--safety_add_to_high_policy", default=False, type=bool)
    parser.add_argument("--safety",                    default=False, type=bool)
    parser.add_argument("--cost_limit",                default=5.0, type=float)
    parser.add_argument("--update_lambda",             default=1000, type=int)
    # logging
    parser.add_argument("--using_wandb",        default=False, type=bool)
    parser.add_argument("--using_comet",        default=True, type=bool)
    parser.add_argument("--wandb_project",      default="validate_ris_polamp", type=str)
    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()

    with open("goal_polamp_env/goal_environment_configs.json", 'r') as f:
        goal_our_env_config = json.load(f)

    with open("polamp_env/configs/train_configs.json", 'r') as f:
        train_config = json.load(f)

    with open("polamp_env/configs/environment_configs.json", 'r') as f:
        our_env_config = json.load(f)

    with open("polamp_env/configs/reward_weight_configs.json", 'r') as f:
        reward_config = json.load(f)

    with open("polamp_env/configs/car_configs.json", 'r') as f:
        car_config = json.load(f)

    if args.dataset == "medium_dataset":
        total_maps = 12
    elif args.dataset == "test_medium_dataset":
        total_maps = 3
    elif args.dataset == "hard_dataset_simplified_test":
        total_maps = 2
    else:
        total_maps = 1
    dataSet = generateDataSet(our_env_config, name_folder=args.dataset, total_maps=total_maps, dynamic=False)
    maps, trainTask, valTasks = dataSet["obstacles"]
    print(f"trainTask: {len(trainTask['map0'])}")
    print(f"valTasks: {len(valTasks['map0'])}")
    # if args.validate_train_dataset:
    #     valTasks = trainTask
    goal_our_env_config["dataset"] = args.dataset
    goal_our_env_config["uniform_feasible_train_dataset"] = args.uniform_feasible_train_dataset
    goal_our_env_config["random_train_dataset"] = args.random_train_dataset
    goal_our_env_config["with_noise"] = args.with_noise
    goal_our_env_config["obs_noise_std"] = args.obs_noise_std
    goal_our_env_config["action_noise_std"] = args.action_noise_std
    if not goal_our_env_config["static_env"]:
        maps["map0"] = []

    args.evaluation = False
    environment_config = {
        'vehicle_config': car_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        'evaluation': args.evaluation,
        'goal_our_env_config' : goal_our_env_config,
    }
    args.other_keys = environment_config

    train_env_name = "polamp_env-v0"
    test_env_name = train_env_name

    # Set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # register polamp env
    register(
        id=train_env_name,
        entry_point='goal_polamp_env.env:GCPOLAMPEnvironment',
        kwargs={'full_env_name': "polamp_env", "config": args.other_keys}
    )

    env         = gym.make(train_env_name)
    test_env    = gym.make(test_env_name)
    action_dim = env.action_space.shape[0]
    env_obs_dim = env.observation_space["observation"].shape[0]
    if args.use_encoder:
        state_dim = args.state_dim 
    else:
        state_dim = env_obs_dim

    load_folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
    load_results = os.path.isdir(load_folder)
    print("weights folder:", load_folder)

    # Create logger
    # TODO: save_git_head_hash = True by default, change it if neccesary
    logger = None
    if args.using_wandb:
        run = wandb.init(project=args.wandb_project)
        
    if args.using_comet:
        comet_ml.login()
        comet_ml_experiment = comet_ml.start(project_name="speis")
        comet_ml_experiment.log_parameters(args)
    
    # Initialize policy
    env_state_bounds = {"x": 100, "y": 100, 
                        "theta": (-np.pi, np.pi),
                        "v": (env.environment.agent.dynamic_model.min_vel, 
                              env.environment.agent.dynamic_model.max_vel), 
                        "steer": (-env.environment.agent.dynamic_model.max_steer, 
                                 env.environment.agent.dynamic_model.max_steer)}
    R = env.environment.agent.dynamic_model.wheel_base / np.tan(env.environment.agent.dynamic_model.max_steer)
    curvature = 1 / R
    policy = RIS(state_dim=state_dim, action_dim=action_dim, 
                 alpha=args.alpha,
                 use_decoder=args.use_decoder,
                 use_encoder=args.use_encoder,
                 safety=args.safety,
                 n_critic=args.n_critic,
                 train_sac=args.train_sac,
                 use_dubins_filter=args.use_dubins_filter,
                 safety_add_to_high_policy=args.safety_add_to_high_policy,
                 cost_limit=args.cost_limit, update_lambda=args.update_lambda,
                 Lambda=args.Lambda, epsilon=args.epsilon,
                 h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, 
                 n_ensemble=args.n_ensemble,
                 clip_v_function=args.clip_v_function, max_grad_norm=args.max_grad_norm, lambda_initialization=args.lambda_initialization,
                 device=args.device, logger=logger if args.log_loss else None, 
                 env_obs_dim=env_obs_dim, add_ppo_reward=env.add_ppo_reward,
                 add_obs_noise=args.add_obs_noise,
                 curriculum_high_policy=args.curriculum_high_policy,
                 vehicle_curvature=curvature,
                 env_state_bounds=env_state_bounds,
                 lidar_max_dist=env.environment.MAX_DIST_LIDAR,
                 train_env=env,
    )

    if load_results:
        policy.load(load_folder, old_version=False)
        print("weights is loaded")
    else:
        print("WEIGHTS ISN'T LOADED")
        assert 1 == 0

    success_rate = 0
    fail_rate = 0
    num_val_tasks = 35
    val_keys = ["map0"]
    val_key = "map0"
    failed_tasks_idx = []

    # print(test_env.maps.keys())
    # print(len(test_env.valTasks[val_key]))
    # validate_tasks = [17, 24, 32, 62, 98, 102, 106, 138, 194, 212, 219]
    # validate_tasks = list(range(num_val_tasks))
    # validate_tasks = [1, 10, 25]

    from ris_train_polamp_env import evalPolicy

    eval_distance, success_rate, eval_reward, \
    val_state, val_goal, \
    mean_actions, eval_episode_length, validation_info \
                    = evalPolicy(policy, test_env, 
                                plot_full_env=not args.not_visual_validation,
                                plot_subgoals=True,
                                plot_value_function=False,
                                render_env=False,
                                plot_only_agent_values=False, 
                                plot_decoder_agent_states=False,
                                plot_subgoal_dispertion=True,
                                plot_lidar_predictor=False,
                                #video_validate_tasks = [("map0", i) for i in range(15)],
                                #video_validate_tasks = [("map0", i) for i in range(15, 30)],
                                #video_validate_tasks = [("map0", i) for i in range(30, 45)],
                                #video_validate_tasks = [("map0", i) for i in range(45, 60)],
                                #video_validate_tasks = [("map0", 0), ("map0", 30), ("map0", 67), ("map0", 84)],
                                video_validate_tasks = [("map0", 0)],
                                value_function_angles=["theta_agent", 0, -np.pi/2],
                                dataset_plot=False,
                                dataset_validation=args.dataset,
                                full_validation=True,
                                skip_not_video_tasks=False,
                                plot_trajectory=args.plot_trajectory,
                                with_noise=args.with_noise,
                                obs_noise_std=args.obs_noise_std,
                                action_noise_std=args.action_noise_std)
    wandb_log_dict = {}
    wandb_log_dict[f'validation/val_rate({args.n_eval} episodes)'] = success_rate
    wandb_log_dict[f'validation/eval_reward({args.n_eval} episodes)'] = eval_reward
    wandb_log_dict[f'validation/eval_distance({args.n_eval} episodes)'] = eval_distance
    wandb_log_dict[f'validation/eval_episode_length({args.n_eval} episodes)'] = eval_episode_length
    
    # Log scalar metrics from validation_info (skip non-scalar values like lists, dicts, etc.)
    scalar_keys = ["eval_cost", "eval_collisions", "eval_min_clearance", "eval_mean_clearance"]
    for val_key in scalar_keys:
        if val_key in validation_info:
            wandb_log_dict[f"validation/{val_key}({args.n_eval} episodes)"] = validation_info[val_key]
    
    # Log other scalar values from validation_info (but skip complex structures)
    for val_key in validation_info:
        if val_key not in scalar_keys and val_key not in ["videos", "trajectory", "task_statuses", "unsuccessful_tasks", "action_info"]:
            val = validation_info[val_key]
            # Only log scalar values (numbers)
            if isinstance(val, (int, float, np.integer, np.floating)):
                wandb_log_dict[f"validation/{val_key}({args.n_eval} episodes)"] = float(val)
    if not args.not_visual_validation:
        for map_name, task_indx, video in validation_info["videos"]:
            wandb_log_dict["validation_video"+"_"+map_name+"_"+f"{task_indx}"] = wandb.Video(video, fps=10, format="gif")
    
    # Add trajectory images to wandb
    if args.plot_trajectory and "trajectory" in validation_info:
        for map_name, task_indx, trajectory_image in validation_info["trajectory"]:
            # Convert numpy array to PIL Image for wandb
            import PIL.Image
            if len(trajectory_image.shape) == 3:
                pil_image = PIL.Image.fromarray(trajectory_image)
                wandb_log_dict[f"validation_trajectory_{map_name}_{task_indx}"] = wandb.Image(pil_image)
    
    if args.using_wandb:
        run.log(wandb_log_dict)
        
    
    print("validation_info keys:", validation_info.keys())
    
    # Log metrics to comet_ml (even if not_visual_validation is set, metrics should be logged)
    if args.using_comet:
        # Log scalar metrics
        comet_metrics = {
            "validation/success_rate": success_rate,
            "validation/eval_reward": eval_reward,
            "validation/eval_distance": eval_distance,
            "validation/eval_episode_length": eval_episode_length,
        }
        if "eval_cost" in validation_info:
            comet_metrics["validation/eval_cost"] = validation_info["eval_cost"]
        if "eval_collisions" in validation_info:
            comet_metrics["validation/eval_collisions"] = validation_info["eval_collisions"]
        if "eval_min_clearance" in validation_info:
            comet_metrics["validation/eval_min_clearance"] = validation_info["eval_min_clearance"]
        if "eval_mean_clearance" in validation_info:
            comet_metrics["validation/eval_mean_clearance"] = validation_info["eval_mean_clearance"]
        if "eval_subgoal_collision_rate" in validation_info:
            comet_metrics["validation/eval_subgoal_collision_rate"] = validation_info["eval_subgoal_collision_rate"]
        comet_ml_experiment.log_metrics(comet_metrics)
    
    if args.using_comet and not args.not_visual_validation:
        for map_name, task_indx, video in validation_info["videos"]:
            # Video is in format (T, C, H, W) from ris_train_polamp_env.py line 555
            # Convert to (T, H, W, C) format for saving
            if len(video.shape) == 4:
                if video.shape[1] == 3 or video.shape[1] == 1:  # (T, C, H, W) with C=3 or C=1
                    video_to_save = np.transpose(video, (0, 2, 3, 1))  # (T, H, W, C)
                elif video.shape[3] == 3 or video.shape[3] == 1:  # Already (T, H, W, C)
                    video_to_save = video
                else:
                    # Assume (T, H, W) grayscale
                    video_to_save = np.expand_dims(video, axis=-1) if len(video.shape) == 3 else video
            else:
                video_to_save = video
            
            # Ensure video is uint8 in range [0, 255]
            if video_to_save.dtype != np.uint8:
                if video_to_save.max() <= 1.0:
                    video_to_save = (video_to_save * 255).astype(np.uint8)
                else:
                    video_to_save = np.clip(video_to_save, 0, 255).astype(np.uint8)
            
            # Remove single channel dimension if grayscale
            if len(video_to_save.shape) == 4 and video_to_save.shape[3] == 1:
                video_to_save = video_to_save.squeeze(axis=3)
            
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_video_path = tmp_file.name
            
            try:
                if imageio is not None:
                    # imageio.mimwrite expects (T, H, W, C) or (T, H, W) for grayscale
                    imageio.mimwrite(tmp_video_path, video_to_save, fps=10, codec='libx264')
                else:
                    # Fallback: save as GIF using PIL if imageio is not available
                    import PIL.Image
                    if len(video_to_save.shape) == 3:  # Grayscale (T, H, W)
                        frames = [PIL.Image.fromarray(frame, mode='L') for frame in video_to_save]
                    else:  # Color (T, H, W, C)
                        frames = [PIL.Image.fromarray(frame) for frame in video_to_save]
                    frames[0].save(tmp_video_path.replace('.mp4', '.gif'), 
                                 save_all=True, append_images=frames[1:], 
                                 duration=100, loop=0)
                    tmp_video_path = tmp_video_path.replace('.mp4', '.gif')
                
                comet_ml_experiment.log_video(
                    tmp_video_path,
                    name=f"validation_video_{map_name}_{task_indx}",
                    overwrite=True
                )
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_video_path):
                    os.unlink(tmp_video_path)
    
    # Log trajectory images to comet_ml
    if args.using_comet and args.plot_trajectory and "trajectory" in validation_info:
        for map_name, task_indx, trajectory_image in validation_info["trajectory"]:
            # Save trajectory image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_image_path = tmp_file.name
            
            try:
                import PIL.Image
                pil_image = PIL.Image.fromarray(trajectory_image)
                pil_image.save(tmp_image_path)
                
                comet_ml_experiment.log_image(
                    tmp_image_path,
                    name=f"validation_trajectory_{map_name}_{task_indx}",
                    overwrite=True
                )
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_image_path):
                    os.unlink(tmp_image_path)
    
    # Print all validation metrics
    print("=" * 60)
    print("VALIDATION METRICS:")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Eval Reward: {eval_reward:.4f}")
    print(f"Eval Distance: {eval_distance:.4f}")
    print(f"Eval Episode Length: {eval_episode_length:.4f}")
    if "eval_cost" in validation_info:
        print(f"Eval Cost: {validation_info['eval_cost']:.4f}")
    if "eval_collisions" in validation_info:
        print(f"Eval Collisions: {validation_info['eval_collisions']:.4f}")
    if "eval_min_clearance" in validation_info:
        print(f"Eval Min Clearance: {validation_info['eval_min_clearance']:.4f}")
    if "eval_mean_clearance" in validation_info:
        print(f"Eval Mean Clearance: {validation_info['eval_mean_clearance']:.4f}")
    if "eval_subgoal_collision_rate" in validation_info:
        print(f"Eval Subgoal Collision Rate: {validation_info['eval_subgoal_collision_rate']:.4f}")
    print("=" * 60)

    