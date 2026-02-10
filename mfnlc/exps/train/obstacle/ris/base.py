import os
from typing import Union, Type, Tuple, Optional, Dict, Any

import numpy as np
import gym
import torch as th
from stable_baselines3 import TD3, HerReplayBuffer, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback, GymEnv
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import wandb
import comet_ml
from wandb.integration.sb3 import WandbCallback

from mfnlc.config import get_path, default_device
from mfnlc.envs import get_env, get_sub_proc_env
from mfnlc.exps.train.utils import copy_current_model_to_log_dir
from mfnlc.learn.safety_ris import SafetyRis
from mfnlc.learn.subgoal import LaplacePolicy, EnsembleCritic, GaussianPolicy, CustomActorCriticPolicy
# from mfnlc.envs.difficulty import choose_level

def train(env_name,
          total_timesteps: int,
          policy_to_delete: Union[str, Type[TD3Policy]] = "MlpPolicy",
          learning_rate: Union[float, Schedule] = 1e-3,
          buffer_size: int = 1_000_000,  # 1e6
          learning_starts: int = 100,
          batch_size: int = 100,
          tau: float = 0.005,
          gamma: float = 0.99,
          train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
          gradient_steps: int = 40,
          action_noise: Optional[ActionNoise] = None,
          goal_selection_strategy = "future", # Available strategies (cf paper): future, final, episode
          replay_buffer_class: Optional[ReplayBuffer] = None,
          replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
          optimize_memory_usage: bool = False,
          ent_coef = "auto",
          target_update_interval = 1,
          target_entropy = "auto",
          use_sde = False,
          sde_sample_freq = -1,
          use_sde_at_warmup = False,
          new_policy_kwargs: Optional[Dict[str, Any]] = None, # RIS
          use_encoder=False,
          encoder_dim=20,
          h_lr = 1e-3, # RIS
          q_lr = 1e-3, # RIS
          pi_lr = 1e-4, # RIS
          epsilon: float = 1e-16, # RIS
          alpha = 0.1, # RIS
          Lambda = 0.1, # RIS
          n_ensemble = 10, # RIS
          clip_v_function = -150, # RIS,
          buffer_size_her: int = 500_000,
          fraction_goals_are_rollout_goals: float = 0.2,
          fraction_resampled_goals_are_env_goals: float = 0.0,
          fraction_resampled_goals_are_replay_buffer_goals: float = 0.5,
          no_safety: bool = False,
          cost_limit = 3.0,
          lambda_initialization: float = 0.5,
          train_sac: bool = False,
          critic_max_grad_norm: float = None, # RIS
          actor_max_grad_norm: float = None, # RIS
          use_one_safe_critic = False,
          safe_critic_behave = "min",
          add_subgoal_reinforce_sg_num = 0,
          subgoal_max_grad_norm: float = None, # RIS
          sgg_optimizing: bool = False,
          create_eval_env: bool = False,
          policy_to_delete_kwargs: Optional[Dict[str, Any]] = None,
          verbose: int = 1,
          seed: Optional[int] = None,
          callback: MaybeCallback = None,
          log_interval: int = 4,
          eval_env: Optional[GymEnv] = None,
          eval_freq: int = -1,
          n_eval_episodes: int = 5,
          tb_log_name: str = "SAFETY_RIS",
          eval_log_path: Optional[str] = None,
          reset_num_timesteps: bool = True,
          n_envs: int = 1,
          validate_freq: int = 5000,
          use_wandb=False,
          use_comet=True,
          validate=False,
          validate_robot_video=False,
          validate_subgoal_video=True,
          validate_video_idx=0,
          load_model=False,
          load_model_folder="",
          obs_noise_std=0.0,
          action_noise_std=0.0
          ):
    algo = "ris"
    if validate:
        total_timesteps = 1
        validate_freq = 1
        test_freq_multipier = 1
    else:
        test_freq_multipier = 4
    if use_wandb:
        if validate:
            wandb_run_name = f"validate_{algo}_load_model={load_model_folder}"
        else:
            wandb_run_name = f"train_{algo}"
        run = wandb.init(
            project="train_safety_ris_safety_gym",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            name=wandb_run_name,
        )
    if use_comet:
        if validate:
            comet_run_name = f"validate_{algo}_load_model={load_model_folder}"
        else:
            comet_run_name = f"train_{algo}"
        comet_ml.login()
        comet_ml_experiment = comet_ml.start(project_name="speis")
        
    if n_envs == 1:
        env = get_env(env_name)
    elif n_envs > 1:
        env = get_sub_proc_env(env_name, n_envs)
    else:
        raise ValueError(f"n_envs should be greater than 0, but it is {n_envs}")
    
    # Add noise wrapper if noise is enabled
    if obs_noise_std > 0 or action_noise_std > 0:
        from mfnlc.envs.base import NoiseWrapper
        env = NoiseWrapper(env, obs_noise_std=obs_noise_std, action_noise_std=action_noise_std)

    robot_name = env_name.split("-")[0]

    tensorboard_log = get_path(robot_name, algo, "log")
    
    class CometCallback(BaseCallback):
        def __init__(self, 
                    eval_env: gym.Env, 
                    comet_experiment: comet_ml.Experiment,
                    render_freq: int, 
                    n_eval_episodes: int = 1, 
                    deterministic: bool = True,
                    verbose: int = 0,
                    model_save_path: Optional[str] = None,
                    model_save_freq: int = 0,
                    validate_robot_video: bool = False,
                    validate_subgoal_video: bool = True,
                    validate_video_idx: int = 0,
                    add_subgoal_reinforce_sg_num: int = 0):
            super().__init__(verbose)
            self._eval_env = eval_env
            self.comet_experiment = comet_experiment
            self._render_freq = render_freq
            self._n_eval_episodes = n_eval_episodes
            self._deterministic = deterministic
            self._is_success_buffer = []
            self._episode_costs = []
            self.collisions = []
            self.custom_success_rate = []
            self.old_success_rate = None
            self.model_save_path = model_save_path
            self.model_save_freq = model_save_freq
            self.validate_robot_video = validate_robot_video
            self.validate_subgoal_video = validate_subgoal_video
            self.validate_video_idx = validate_video_idx
            self.add_subgoal_reinforce_sg_num = add_subgoal_reinforce_sg_num

        def _on_step(self) -> bool:
            def run_episodes_and_log_comet(validation_dataset=True, num_episodes=self._n_eval_episodes):
                if validation_dataset:
                    log_folder_name = "eval"
                    prefix = "val"
                else:
                    log_folder_name = "test"
                    prefix = "test"
                
                robot_screens = []
                positions_screens = []
                self._is_success_buffer = []
                self._episode_costs = []
                self.collisions = []
                self.custom_success_rate = []
                dubug_info = {"acc_reward": 0, "t": 0, "acc_cost": 0}
                debug_v_s_sg = []
                debug_v_sg_g = []

                def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                    assert len(_locals["env"].envs) == 1
                    if _locals['done'] and _locals['info']["collision"]:
                        self.collisions.append(1.0)
                    elif _locals['done']:
                        self.collisions.append(0.0)
                        self.custom_success_rate.append(1.0)
                    
                    dubug_info["a0"] = _locals["actions"][0][0]
                    dubug_info["a1"] = _locals["actions"][0][1]
                    dubug_info["acc_reward"] += _locals["reward"]
                    dubug_info["acc_cost"] += _locals["info"]["clearance_is_enough"]
                    dubug_info["v_s_sg"] = []
                    dubug_info["v_sg_g"] = []
                    dubug_info["t"] += 1
                    
                    if not self.model.sac:
                        with th.no_grad():
                            state = _locals["observations"]["observation"]
                            goal = _locals["observations"]["desired_goal"]
                            to_torch_state = th.FloatTensor(state).to(default_device).unsqueeze(0)
                            to_torch_goal = th.FloatTensor(goal).to(default_device).unsqueeze(0)
                            
                            if self.model.use_encoder:
                                encoded_state = self.model.encoder(to_torch_state)
                                encoded_goal = self.model.encoder(to_torch_goal)
                            else:
                                encoded_state = to_torch_state
                                encoded_goal = to_torch_goal

                            subgoals = []
                            _locals["env"].envs[0].setup_subgoals()
                            plot_subgoals = max(1, self.add_subgoal_reinforce_sg_num)
                            
                            for i in range(plot_subgoals):
                                subgoal_distribution = self.model.subgoal_net(encoded_state, encoded_goal)
                                subgoal = subgoal_distribution.loc
                                encoded_goal = subgoal
                                
                                if self.model.use_encoder:
                                    cuda_decoded_subgoal = self.model.policy.encoder.decoder(subgoal)
                                    decoded_subgoal = cuda_decoded_subgoal.cpu()
                                else:
                                    cuda_decoded_subgoal = subgoal
                                    decoded_subgoal = subgoal.cpu()
                                    
                                if self._eval_env.plot_subgoal:
                                    _locals["env"].envs[0].set_subgoal_pos(i, decoded_subgoal)

                    if _locals["episode_counts"][_locals["i"]] == self.validate_video_idx:
                        if self.validate_robot_video:
                            if self._eval_env.plot_only_start_goal_pose:
                                if dubug_info["t"] == 1:
                                    screen = self._eval_env.custom_render(positions_render=False)
                                    robot_screens.append(screen)
                            else:
                                screen = self._eval_env.custom_render(positions_render=False)
                                robot_screens.append(screen)
                        
                        if self.validate_subgoal_video:
                            if self._eval_env.plot_only_start_goal_pose:
                                if dubug_info["t"] == 1:
                                    screen = self._eval_env.custom_render(positions_render=True, dubug_info=dubug_info)
                                    positions_screens.append(screen)
                            else:
                                screen = self._eval_env.custom_render(positions_render=True, dubug_info=dubug_info)
                                positions_screens.append(screen)

                    if _locals["done"]:
                        maybe_is_success = _locals["info"].get("goal_is_arrived")
                        if maybe_is_success is not None:
                            self._is_success_buffer.append(maybe_is_success)
                        episode_cost = _locals["info"].get("episode_cost")
                        if episode_cost is not None:
                            self._episode_costs.append(episode_cost)

                print("---------------- start validation ---------------")
                print("validation tasks:", num_episodes)
                self.model.policy.setup_actor_critic()
                
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self._eval_env,
                    callback=grab_screens,
                    return_episode_rewards=True,
                    n_eval_episodes=num_episodes,
                    deterministic=self._deterministic,
                )

                # Log videos to Comet ML
                if self.validate_robot_video and robot_screens:
                    self.comet_experiment.log_video(
                        np.array(robot_screens).transpose(0, 3, 1, 2),
                        name=f"{prefix}_robot_video_step_{self.n_calls}",
                        fps=10
                    )

                if self.validate_subgoal_video and positions_screens:
                    self.comet_experiment.log_video(
                        np.array(positions_screens).transpose(0, 3, 1, 2),
                        name=f"{prefix}_subgoal_video_step_{self.n_calls}",
                        fps=10
                    )

                # Calculate metrics
                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                min_reward, max_reward = np.min(episode_rewards), np.max(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                collision_rate = np.mean(self.collisions) if self.collisions else 0
                success_rate = np.mean(self._is_success_buffer) if self._is_success_buffer else 0
                mean_cost = np.mean(self._episode_costs) if self._episode_costs else 0

                # Log metrics to Comet ML
                metrics = {
                    f"{prefix}/reward": float(mean_reward),
                    f"{prefix}/ep_length": mean_ep_length,
                    f"{prefix}/reward_std": float(std_reward),
                    f"{prefix}/reward_min": min_reward,
                    f"{prefix}/reward_max": max_reward,
                    f"{prefix}/collision_rate": collision_rate,
                    f"{prefix}/success_rate": success_rate,
                    f"{prefix}/mean_cost": mean_cost,
                }

                for metric_name, metric_value in metrics.items():
                    self.comet_experiment.log_metric(metric_name, metric_value, step=self.n_calls)

                # Also log to console
                self.logger.record(f"{log_folder_name}/{log_folder_name}_reward", float(mean_reward))
                self.logger.record(f"{log_folder_name}/{log_folder_name}_ep_length", mean_ep_length)
                self.logger.record(f"{log_folder_name}/{log_folder_name}_collision_rate", collision_rate)
                self.logger.record(f"{log_folder_name}/{log_folder_name}_success_rate", success_rate)
                self.logger.record(f"{log_folder_name}/{log_folder_name}_mean_cost", mean_cost)

                print("---------------- end validation ---------------")
                return success_rate

            if self.n_calls % self._render_freq == 0:
                val_success_rate = run_episodes_and_log_comet(validation_dataset=True)

                # Save model
                if self.model_save_path:
                    folder = self.model_save_path + "/last_"
                    os.makedirs(folder, exist_ok=True)
                    self.model.save(folder)

                    # Save best model
                    if self.old_success_rate is None or val_success_rate >= self.old_success_rate:
                        self.old_success_rate = val_success_rate
                        folder = self.model_save_path + "/best_"
                        os.makedirs(folder, exist_ok=True)
                        self.model.save(folder)

                return True

            return True

    # add custom video callback
    class VideoRecorderCallback(WandbCallback):
        def __init__(self, 
                    eval_env: gym.Env, 
                    render_freq: int, 
                    n_eval_episodes: int = 1, 
                    deterministic: bool = True,
                    verbose: int = 0,
                    model_save_path: Optional[str] = None,
                    model_save_freq: int = 0,
                    gradient_save_freq: int = 0):
            super().__init__(gradient_save_freq=gradient_save_freq, # error if > 0 
                             model_save_path=model_save_path,
                             verbose=verbose)
            self._eval_env = eval_env
            self._render_freq = render_freq
            self._n_eval_episodes = n_eval_episodes
            self._deterministic = deterministic
            self._is_success_buffer = []
            self._episode_costs = []
            self.old_success_rate = None

        def _on_step(self) -> bool:
            def run_episodes_and_log_wandb(validation_dataset=True, num_episodes=self._n_eval_episodes):
                if validation_dataset:
                    wandb_folder_name = "eval"
                else:
                    wandb_folder_name = "test"
                if validate_robot_video:
                    robot_screens = []
                if validate_subgoal_video:
                    positions_screens = []
                self._is_success_buffer = []
                self._episode_costs = []
                self.collisions = []
                self.custom_success_rate = []
                dubug_info = {"acc_reward" : 0, "t": 0, "acc_cost" : 0}
                debug_v_s_sg = []
                debug_v_sg_g = []

                def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                    # predict subgoal and set to env
                    assert len(_locals["env"].envs) == 1
                    if _locals['done'] and _locals['info']["collision"]:
                        self.collisions.append(1.0)
                    elif _locals['done']:
                        self.collisions.append(0.0)
                        self.custom_success_rate.append(1.0)
                    # print(f"action: {_locals['actions']}")
                    dubug_info["a0"] = _locals["actions"][0][0]
                    dubug_info["a1"] = _locals["actions"][0][1]
                    dubug_info["acc_reward"] += _locals["reward"]
                    dubug_info["acc_cost"] += _locals["info"]["clearance_is_enough"]
                    dubug_info["v_s_sg"] = []
                    dubug_info["v_sg_g"] = []
                    dubug_info["t"] += 1
                    if not self.model.sac:
                        with th.no_grad():
                            state = _locals["observations"]["observation"]
                            goal = _locals["observations"]["desired_goal"]
                            to_torch_state = th.FloatTensor(state).to(default_device).unsqueeze(0)
                            to_torch_goal = th.FloatTensor(goal).to(default_device).unsqueeze(0)
                            if self.model.use_encoder:
                                encoded_state = self.model.encoder(to_torch_state)
                                encoded_goal = self.model.encoder(to_torch_goal)
                            else:
                                encoded_state = to_torch_state
                                encoded_goal = to_torch_goal

                            subgoals = []
                            _locals["env"].envs[0].setup_subgoals()
                            plot_subgoals = max(1, add_subgoal_reinforce_sg_num)
                            for i in range(plot_subgoals):
                                subgoal_distribution = self.model.subgoal_net(encoded_state, encoded_goal)
                                subgoal = subgoal_distribution.loc
                                encoded_goal = subgoal
                                if self.model.use_encoder:
                                    cuda_decoded_subgoal = policy.encoder.decoder(subgoal)
                                    decoded_subgoal = cuda_decoded_subgoal.cpu()
                                else:
                                    cuda_decoded_subgoal = subgoal
                                    decoded_subgoal = subgoal.cpu()
                            #if _locals["episode_counts"][_locals["i"]] == 0:
                            #    debug_v_s_sg.append(self.model.value(encoded_state, subgoal).cpu().item())
                            #    debug_v_sg_g.append(self.model.value(subgoal, subgoal).cpu().item())
                            #    dubug_info["v_s_sg"] = debug_v_s_sg
                            #    dubug_info["v_sg_g"] = debug_v_sg_g
                                if self._eval_env.plot_subgoal:
                                    _locals["env"].envs[0].set_subgoal_pos(i, decoded_subgoal)
                        # dubug subgoal
                        #if _locals["episode_counts"][_locals["i"]] == 0 and dubug_info["t"] == 1:
                        #    print("state:", state)
                        #    print("subgoal:", subgoal[0])
                        #    print("goal:", goal)
                    # get video
                    if _locals["episode_counts"][_locals["i"]] == validate_video_idx:
                        if validate_robot_video:
                            if self._eval_env.plot_only_start_goal_pose:
                                if dubug_info["t"] == 1:
                                    screen = self._eval_env.custom_render(positions_render=False)
                                    robot_screens.append(screen.transpose(2, 0, 1))
                            else:
                                screen = self._eval_env.custom_render(positions_render=False)
                                robot_screens.append(screen.transpose(2, 0, 1))
                        if validate_subgoal_video:
                            if self._eval_env.plot_only_start_goal_pose:
                                if dubug_info["t"] == 1:
                                    screen = self._eval_env.custom_render(positions_render=True, dubug_info=dubug_info)
                                    positions_screens.append(screen.transpose(2, 0, 1))
                            else:
                                screen = self._eval_env.custom_render(positions_render=True, dubug_info=dubug_info)
                                positions_screens.append(screen.transpose(2, 0, 1))
                    # get success rate
                    if _locals["done"]:
                        maybe_is_success = _locals["info"].get("goal_is_arrived")
                        if maybe_is_success is not None:
                            self._is_success_buffer.append(maybe_is_success)
                        episode_cost = _locals["info"].get("episode_cost")
                        if episode_cost is not None:
                            self._episode_costs.append(episode_cost)

                print("---------------- start validation ---------------")
                print("validation tasks:", num_episodes)
                self.model.policy.setup_actor_critic()
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self._eval_env,
                    callback=grab_screens,
                    return_episode_rewards=True,
                    n_eval_episodes=num_episodes,
                    deterministic=self._deterministic,
                )
                if validate_robot_video:
                    print("---------------- save robot video ---------------")
                    self.logger.record(
                        f"{wandb_folder_name}/{wandb_folder_name}_video",
                        Video(th.ByteTensor([robot_screens]), fps=40),
                        exclude=("stdout", "log", "json", "csv"),
                    )
                if validate_subgoal_video:
                    print("---------------- save subgoal video ---------------")
                    print("video len:", len(positions_screens))
                    print("image shape:", positions_screens[0].shape)
                    self.logger.record(
                        f"{wandb_folder_name}/{wandb_folder_name}_pos_video",
                        #Video(th.ByteTensor([positions_screens]), fps=40),
                        th.ByteTensor(np.array([positions_screens])),
                        exclude=("stdout", "log", "json", "csv"),
                    )
                    # test bug with save video in training
                    if validate:
                        wandb_log_dict = {}
                        if validation_dataset:
                            prefix = "val_dataset"
                        else:
                            prefix = "test_dataset"
                        wandb_log_dict[f"{prefix}_video"] = \
                            wandb.Video(np.array(positions_screens), fps=10, format="gif", caption=f"steps: {self.n_calls}")
                        if use_comet:
                            comet_ml_experiment.log_parameters(wandb_log_dict)
                        if use_wandb:    
                            run.log(wandb_log_dict)

                if validate_robot_video:
                    del robot_screens
                if validate_subgoal_video:
                    del positions_screens
                del debug_v_s_sg
                del debug_v_sg_g
                del dubug_info

                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                min_reward, max_reward = np.min(episode_rewards), np.max(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                collision_rate = np.mean(self.collisions)
                # Add to current Logger
                self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_reward", float(mean_reward))
                self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_ep_length", mean_ep_length)
                self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_reward_min", min_reward)
                self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_reward_max", max_reward)
                self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_collision_rate", collision_rate)
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_success_rate", success_rate)
                else:
                    success_rate = np.mean(self._is_success_buffer)
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_success_rate", 0)
                
                if len(self._episode_costs) > 0:
                    mean_cost = np.mean(self._episode_costs)
                    max_cost = np.max(self._episode_costs)
                    min_cost = np.min(self._episode_costs)
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_mean_cost", mean_cost)
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_max_cost", max_cost)
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_min_cost", min_cost)
                else:
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_mean_cost", 0)
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_max_cost", 0)
                    self.logger.record(f"{wandb_folder_name}/{wandb_folder_name}_min_cost", 0)

                if validate:
                    wandb_log_dict = {}
                    wandb_log_dict["val_reward"] = float(mean_reward)
                    wandb_log_dict["val_ep_length"] = mean_ep_length                    
                    wandb_log_dict["val_collision_rate"] = collision_rate
                    if len(self._is_success_buffer) > 0:
                        success_rate = np.mean(self._is_success_buffer)
                        wandb_log_dict["val_success_rate"] = success_rate
                    else:
                        success_rate = np.mean(self._is_success_buffer)
                        wandb_log_dict["val_success_rate"] = 0
                    if len(self._episode_costs) > 0:
                        mean_cost = np.mean(self._episode_costs)
                        wandb_log_dict["val_mean_cost"] = mean_cost
                    else:
                        wandb_log_dict["val_mean_cost"] = 0
                    if use_wandb:
                        run.log(wandb_log_dict)
                    if use_comet:
                        comet_ml_experiment.log_parameters(wandb_log_dict)

                    print("solved tasks:", self._is_success_buffer)
                    print("custom solved tasks:", self.custom_success_rate)
                    assert 1 == 0, "end validation"

                print("---------------- end validation ---------------")

                return success_rate
            
            if (self.n_calls % self._render_freq == 0):
                val_success_rate = run_episodes_and_log_wandb(validation_dataset=True)

                # Save (current) results
                hyperparams_tune = False
                folder = self.model_save_path + "/" + "last_"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                if not hyperparams_tune:
                    self.model.save(folder)
                # Save (best) results
                if self.old_success_rate is None or val_success_rate >= self.old_success_rate:
                    self.old_success_rate = val_success_rate
                    folder = self.model_save_path + "/" + "best_"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    if not hyperparams_tune:
                        self.model.save(folder)
            
                return True
            
            elif self.n_calls % (test_freq_multipier * self._render_freq + 1) == 0:
                if not (robot_name == "GCNav"):
                    self._eval_env.set_test_env()
                    test_success_rate = run_episodes_and_log_wandb(validation_dataset=False, num_episodes=100)
                    self._eval_env.set_eval_env()
            else:
                return super()._on_step()
        
    if n_envs == 1:
        callback_eval_env = get_env(env_name)
    else:
        assert 1 == 0
    
    # Add noise wrapper to eval env if noise is enabled
    if obs_noise_std > 0 or action_noise_std > 0:
        from mfnlc.envs.base import NoiseWrapper
        callback_eval_env = NoiseWrapper(callback_eval_env, obs_noise_std=obs_noise_std, action_noise_std=action_noise_std)

    # test eval env
    obs = callback_eval_env.reset()
    print("obs type:", type(obs))
    #print("obs:", callback_eval_env.observation_space.keys())
    print("image shape:", callback_eval_env.custom_render(positions_render=True).shape)

    if callback_eval_env.is_custom_dataset:
        assert validate

    env_obs_dim = env.observation_space["observation"].shape[0]
    env_goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    assert env_obs_dim == env_goal_dim
    run_id = 0
    video_recorder = None
    if use_wandb:
        assert not use_comet
        run_id = run.id
        video_recorder = VideoRecorderCallback(callback_eval_env, 
                                            n_eval_episodes=len(callback_eval_env.custom_dataset["start"]) if callback_eval_env.is_custom_dataset else 10, 
                                            render_freq=validate_freq,
                                            gradient_save_freq=0, # error if > 0 
                                            model_save_path=f"models/{run_id}",
                                            verbose=2)
        wandb.config["model_save_path"] = video_recorder.model_save_path
        
    if use_comet:
        run_id = comet_ml_experiment.get_key()
        video_recorder = CometCallback(
            callback_eval_env,
            comet_ml_experiment,
            render_freq=validate_freq,
            n_eval_episodes=len(callback_eval_env.custom_dataset["start"]) if callback_eval_env.is_custom_dataset else 10,
            model_save_path=f"models/{run_id}",
            validate_robot_video=validate_robot_video,
            validate_subgoal_video=validate_subgoal_video,
            validate_video_idx=validate_video_idx,
            add_subgoal_reinforce_sg_num=add_subgoal_reinforce_sg_num
        )
        comet_ml_experiment.log_parameter("model_save_path", video_recorder.model_save_path)
    
    """
    if use_comet:
        run_id = comet_ml_experiment.get_key()
        video_recorder = VideoRecorderCallback(callback_eval_env, 
                                            n_eval_episodes=len(callback_eval_env.custom_dataset["start"]) if callback_eval_env.is_custom_dataset else 10, 
                                            render_freq=validate_freq,
                                            gradient_save_freq=0, # error if > 0 
                                            model_save_path=f"models/{run_id}",
                                            verbose=2)
        wandb.config["model_save_path"] = video_recorder.model_save_path
    """
        
    print("Ending")
    env_state_dim = env_obs_dim
    if use_encoder:
        state_dim = encoder_dim
    else:
        state_dim = env_obs_dim
        
    actor = GaussianPolicy(state_dim, action_dim, 
                           hidden_dims=new_policy_kwargs["net_arch"]).to(default_device)
    critic = EnsembleCritic(state_dim, action_dim, 
                            hidden_dims=new_policy_kwargs["net_arch"],
                            n_Q=2).to(default_device)
    critic_cost = EnsembleCritic(state_dim, action_dim, 
                            hidden_dims=new_policy_kwargs["net_arch"],
                            n_Q=1 if use_one_safe_critic else 2).to(default_device)
    subgoal_net = LaplacePolicy(state_dim=state_dim, 
                                goal_dim=state_dim, 
                                hidden_dims=new_policy_kwargs["net_arch"]).to(default_device)
    policy = CustomActorCriticPolicy(default_device, add_subgoal_reinforce_sg_num)
    policy.actor = actor
    policy.critic = critic
    policy.critic_cost = critic_cost
    if validate:
        policy.subgoal_net = subgoal_net

    print("******************************")
    print("rew critics:", critic.n_Q)
    print("cost critics:", critic_cost.n_Q)
    print("add_subgoal_reinforce_sg_num:", add_subgoal_reinforce_sg_num)

    model = SafetyRis(
        use_encoder,
        env_state_dim,
        policy,
        subgoal_net,
        state_dim,
        action_dim,
        "MultiInputPolicy",  
        env, 
        h_lr, 
        q_lr,
        pi_lr,
        epsilon,
        no_safety,
        cost_limit,
        lambda_initialization,
        safe_critic_behave,
        train_sac,
        critic_max_grad_norm,
        actor_max_grad_norm,
        subgoal_max_grad_norm,
        sgg_optimizing,
        learning_rate, buffer_size, learning_starts, batch_size, tau, gamma,
        train_freq, gradient_steps, action_noise, 
        HerReplayBuffer, #replay_buffer_class
        dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
        ), # replay_buffer_kwargs
        optimize_memory_usage, ent_coef, target_update_interval, target_entropy, 
        use_sde, sde_sample_freq, use_sde_at_warmup, 
        alpha, Lambda, n_ensemble, clip_v_function,
        buffer_size_her,
        fraction_goals_are_rollout_goals,
        fraction_resampled_goals_are_env_goals,
        fraction_resampled_goals_are_replay_buffer_goals,
        tensorboard_log, create_eval_env, policy_to_delete_kwargs, verbose, seed, default_device)
    
    # load model
    if use_wandb:
        wandb.config["load_model"] = load_model
    if use_comet:
        comet_ml_experiment.log_parameter("load_model", load_model)
    if load_model:
        #folder = "models/m0m2u2vh/"
        folder = f"models/{load_model_folder}/"
        load_results = os.path.isdir(folder)
        assert load_results
        model.load(folder)
        print("weights is loaded")
        wandb.config["loaded_model_path"] = folder
    else:
        print("WEIGHTS ISN'T LOADED")
    
    model.learn(total_timesteps=total_timesteps, callback=video_recorder, log_interval=log_interval)

    model_path = get_path(robot_name, algo, "model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    copy_current_model_to_log_dir(robot_name, algo)
