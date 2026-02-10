import io
import os.path
import pathlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable

import gym
import numpy as np
import torch as th
import torch.nn as nn
import time
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update, check_for_correct_spaces
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from torch.nn import functional as F
from stable_baselines3.common.utils import safe_mean

from mfnlc.config import default_device
from mfnlc.learn.utils import list_dict_to_dict_list
from mfnlc.learn.subgoal import LaplacePolicy, GaussianPolicy, EnsembleCritic, CustomActorCriticPolicy, Encoder
from mfnlc.learn.HER import HERReplayBuffer, PathBuilder
from collections import deque


class SafetyRis(SAC):
    def __init__(
        self,
        use_encoder: bool,
        env_state_dim: int,
        policy: CustomActorCriticPolicy,
        subgoal_net: LaplacePolicy,
        state_dim: int,
        action_dim: int,
        policy_to_delete: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        h_lr: float = 1e-4, 
        q_lr: float = 1e-3,
        pi_lr: float = 1e-4, 
        epsilon: float = 1e-16,
        no_safety: bool = False,
        cost_limit = 3.0, 
        lambda_initialization = 5.0,
        safe_critic_behave = "min", 
        train_sac: bool = False,
        critic_max_grad_norm: float = None,
        actor_max_grad_norm: float = None,
        subgoal_max_grad_norm: float = None,
        sgg_optimizing: bool = False,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        alpha: float = 0.1,
        Lambda: float = 0.1, 
        n_ensemble: int = 10, 
        clip_v_function: float = -150,
        buffer_size_her: int = 500_000,
        fraction_goals_are_rollout_goals: float = 0.2,
        fraction_resampled_goals_are_env_goals: float = 0.0,
        fraction_resampled_goals_are_replay_buffer_goals: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(SafetyRis, self).__init__(
            policy_to_delete,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        # policy
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.new_policy = policy
        self.critic_max_grad_norm = critic_max_grad_norm
        self.actor_max_grad_norm = actor_max_grad_norm
        self.adaptive_collision_reward = False

        # encoder
        self.use_encoder = use_encoder
        self.use_decoder = self.use_encoder
        assert self.use_encoder or (self.use_encoder == self.use_decoder), "use decoder only with encoder"
        self.new_policy.use_encoder = self.use_encoder
        self.new_policy.use_decoder = self.use_decoder
        self.enc_lr = 1e-4
        if self.use_encoder:
            self.encoder = Encoder(input_dim=env_state_dim, state_dim=self.state_dim, use_decoder=self.use_decoder).to(device)
            self.new_policy.encoder = self.encoder
            self.encoder_optimizer = th.optim.Adam(self.encoder.parameters(), lr=self.enc_lr)
            if self.use_decoder:
                self.autoencoder_criterion = nn.MSELoss()
                self.autoencoder_optimizer = th.optim.Adam(self.encoder.decoder.parameters(), lr=self.enc_lr)

        # subgoal
        self.subgoal_net = subgoal_net
        self.subgoal_optimizer = th.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)
        self.alpha = alpha
        self.Lambda = Lambda
        self.n_ensemble = n_ensemble
        self.clip_v_function = clip_v_function
        self.epsilon = epsilon
        self.subgoal_max_grad_norm = subgoal_max_grad_norm
        self.sgg_optimizing = sgg_optimizing

        # additional buffer
        self.ep_collision_buffer = deque(maxlen=100)
        self.ep_min_distance_buffer = deque(maxlen=100)
        self.ep_cost_buffer = deque(maxlen=100)

        # path builder for HER
        vectorized = False
        self.path_builder = PathBuilder()
        self.custom_replay_buffer = HERReplayBuffer(
            max_size=buffer_size_her,
            env=env,
            fraction_goals_are_rollout_goals = fraction_goals_are_rollout_goals,
            fraction_resampled_goals_are_env_goals = fraction_resampled_goals_are_env_goals,
            fraction_resampled_goals_are_replay_buffer_goals = fraction_resampled_goals_are_replay_buffer_goals,
            ob_keys_to_save     =["collision", "clearance_is_enough"],
            desired_goal_keys   =["desired_goal"],
            observation_key     = 'observation',
            desired_goal_key    = 'desired_goal',
            achieved_goal_key   = 'achieved_goal',
            vectorized          = vectorized 
        )

        # safety
        self.safety = not no_safety
        self.safe_critic_behave = safe_critic_behave
        self.cost_limit = cost_limit
        self.lambda_initialization = lambda_initialization

        # sac
        self.sac = train_sac
        self.sac_alpha = 0.2

        if _init_setup_model:
            self._setup_model()
        
        # Ensure _stats_window_size is set (required by stable_baselines3)
        # This is normally set in _setup_model, but we ensure it's set here
        if not hasattr(self, '_stats_window_size') or self._stats_window_size is None:
            self._stats_window_size = 100

    def _setup_model(self) -> None:
        super(SafetyRis, self)._setup_model()
        self._setup_alias()
        
        # Ensure _stats_window_size is set after setup (required by stable_baselines3)
        if not hasattr(self, '_stats_window_size') or not isinstance(self._stats_window_size, int):
            self._stats_window_size = 100
    

    def _setup_alias(self) -> None:
        # setup new actor critic
        del self.actor
        del self.critic
        del self.critic_target
        del self.policy.actor
        del self.policy.critic
        del self.policy.critic_target
        del self.policy
        self.policy = self.new_policy
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=self.pi_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=self.q_lr)
        if self.safety:
            cost_limit = self.cost_limit
            max_episode_steps = 300
			# we should use the timestep_cost_limit
            self.timestep_cost_limit = cost_limit * (1 - self.gamma ** max_episode_steps) / (1 - self.gamma) / max_episode_steps
            print(f"timestep_cost_limit: {self.timestep_cost_limit}")
            self.critic_cost = self.policy.critic_cost
            self.critic_cost_target = deepcopy(self.critic_cost)
            self.critic_cost_optimizer = th.optim.Adam(self.critic_cost.parameters(), lr=self.q_lr)
            self.update_lambda = 1000
            lambda_initialization = self.lambda_initialization
            self.lambda_coefficient = th.tensor(lambda_initialization, requires_grad=True)
            self.lambda_optimizer = th.optim.Adam([self.lambda_coefficient], lr=5e-4)

    
    def sample_and_preprocess_batch(self, replay_buffer, env, batch_size=256, device=th.device("cuda")):
        # Extract 
        batch = replay_buffer.random_batch(batch_size)
        state_batch         = batch["observations"]
        action_batch        = batch["actions"]
        next_state_batch    = batch["next_observations"]
        goal_batch          = batch["resampled_goals"]
        reward_batch        = batch["rewards"]
        done_batch          = batch["terminals"]
        clearance_is_enough_batch = batch["clearance_is_enough"]
        collision_batch     = batch["collision"]       
        
        # vel_pos = int(env.envs[0].env.obstacle_in_obs) * 2
        # i_v = vel_pos + 1
        # e_v = i_v + 2
        shift_v = int(next_state_batch.shape[1] / env.envs[0].frame_stack * (env.envs[0].frame_stack - 1))
        # Compute sparse rewards: -1 for all actions until the goal is reached
        reward_batch = np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, shift_v:shift_v+2], 2).sum(-1, keepdims=True)) # distance: next_state to goal
        if "Nav" in env.envs[0].env.robot_name: 
            done_batch   = 1.0 * (reward_batch <= env.envs[0].env.arrive_radius)# terminal condition
        else:
            # For SafetyGym environments, try to get goal_size from different locations
            env_obj = env.envs[0].env
            goal_threshold = None
            # Try different paths to get goal_size or arrive_radius
            if hasattr(env_obj, 'arrive_radius'):
                goal_threshold = env_obj.arrive_radius
            elif hasattr(env_obj, 'env') and hasattr(env_obj.env, 'goal_size'):
                goal_threshold = env_obj.env.goal_size
            elif hasattr(env_obj, 'goal_size'):
                goal_threshold = env_obj.goal_size
            else:
                # Fallback: use default goal size for SafetyGym
                goal_threshold = 0.3
            
            done_batch = 1.0 * (reward_batch <= goal_threshold)

        # done_batch   = 1.0 * (reward_batch <= env.envs[0].env.env.goal_size) + \
        #     1.0 * (np.sqrt(np.power(np.array(next_state_batch)[:, -e_v:-i_v], 2).sum(-1, keepdims=True)) > 0.1)
        done_batch = 1.0 * collision_batch + (1.0 - 1.0 * collision_batch) * (done_batch)
        # done_batch = 1.0 * collision_batch + (1.0 - 1.0 * collision_batch) * (done_batch // (1.0 + 1.0))
        if not self.adaptive_collision_reward or (self.adaptive_collision_reward and self.num_timesteps < 300_000):
            reward_batch = (- np.ones_like(done_batch) * (-env.envs[0].env.time_step_reward)) * (1.0 - collision_batch) \
                            + (env.envs[0].env.collision_penalty) * collision_batch
        else:
            reward_batch = (- np.ones_like(done_batch) * (-env.envs[0].env.time_step_reward)) * (1.0 - collision_batch) \
                            + (env.envs[0].env.collision_penalty - 200) * collision_batch

        cost_batch = clearance_is_enough_batch

        # Convert to Pytorch
        state_batch         = th.FloatTensor(state_batch).to(device)
        action_batch        = th.FloatTensor(action_batch).to(device)
        reward_batch        = th.FloatTensor(reward_batch).to(device)
        cost_batch        = th.FloatTensor(cost_batch).to(device)
        next_state_batch    = th.FloatTensor(next_state_batch).to(device)
        done_batch          = th.FloatTensor(done_batch).to(device)
        goal_batch          = th.FloatTensor(goal_batch).to(device)

        return state_batch, action_batch, reward_batch, cost_batch, next_state_batch, done_batch, goal_batch

    
    # RIS requires training each time when env.step()
    def _on_step(self):
        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            gradient_steps = self.gradient_steps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ):
        # collect_rollouts should collect one full episode to self.path_builder
        # TODO vec env resets env if it is terminated, so i have to accout last obs
        assert train_freq.unit == TrainFrequencyUnit.EPISODE and \
               train_freq.frequency == 1 
        self.path_builder = PathBuilder()
        rollout = super().collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
        self.custom_replay_buffer.add_path(self.path_builder.get_all_stacked())  
        return rollout
    
    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            maybe_is_collision = info.get("collision")
            min_goal_distance = info.get("min_goal_distance")
            episode_cost = info.get("episode_cost")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
            if maybe_is_collision is not None and dones[idx]:
                self.ep_collision_buffer.append(maybe_is_collision)
            if min_goal_distance is not None and dones[idx]:
                self.ep_min_distance_buffer.append(min_goal_distance)
            if episode_cost is not None and dones[idx]:
                self.ep_cost_buffer.append(episode_cost)
    
    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        if len(self.ep_collision_buffer) > 0:
            self.logger.record("rollout/train_collision_rate", safe_mean(self.ep_collision_buffer))
        if len(self.ep_min_distance_buffer) > 0:
            self.logger.record("rollout/avg_min_distance", safe_mean(self.ep_min_distance_buffer))
        if len(self.ep_cost_buffer) > 0:
            self.logger.record("rollout/cumulative_cost", safe_mean(self.ep_cost_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        assert self.n_envs == 1
        self.path_builder.add_all(
            observations=self._last_original_obs,
            actions=buffer_action,
            rewards=reward_,
            next_observations=next_obs,
            terminals=[1.0*dones[0]]
        )
        
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    def train_lagrangian(self, state, action, goal, debug_info={}):
        Q_cost = self.critic_cost(state, action, goal)
        if self.safe_critic_behave == "min":
            Q_cost = th.min(Q_cost, -1, keepdim=True)[0]
        elif self.safe_critic_behave == "max":
            Q_cost = th.max(Q_cost, -1, keepdim=True)[0]
        elif self.safe_critic_behave == "mean":
            Q_cost = th.mean(Q_cost, -1, keepdim=True)[0]
        else:
            assert 1 == 0
        Q_cost = th.clamp(Q_cost, min=0.0)
        violation = Q_cost - self.timestep_cost_limit
        lambda_loss =  self.lambda_coefficient * violation.detach()
        lambda_loss = -lambda_loss.mean()
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        debug_info["lambda_loss"].append(lambda_loss.mean().item())

    def train_highlevel_policy(self, state, goal, subgoal, debug_info={}, sgg_optimizing=False):
		# Compute subgoal distribution 
        batch_size = state.shape[0] # 2048
        subgoal_distribution = self.subgoal_net(state, goal)
        if sgg_optimizing:
            goal = subgoal_distribution.loc
            subgoal_distribution = self.subgoal_net(state, goal)
        with th.no_grad():
            # Compute target value
            new_subgoal = subgoal_distribution.loc # 2048 x 20
            policy_v_1 = self.value(state, new_subgoal) # 2048 x 1
            policy_v_2 = self.value(new_subgoal, goal) # 2048 x 1
            policy_v = th.cat([policy_v_1, policy_v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]

            # Compute subgoal distance loss
            v_1 = self.value(state, subgoal)
            v_2 = self.value(subgoal, goal)
            v = th.cat([v_1, v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]
            adv = - (v - policy_v)
            weight = F.softmax(adv/self.Lambda, dim=0)

        log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)
        subgoal_loss = - (log_prob * weight).mean()
        if not sgg_optimizing:
            prefix = ""
            debug_info[prefix+"subgoal_net_losses"].append(subgoal_loss.item())
            debug_info[prefix+"advs"].append(adv.mean().item())
            debug_info[prefix+"target_subgoal_V"].append(v.mean().item())
            debug_info[prefix+"subgoal_V"].append(policy_v.mean().item())
            debug_info[prefix+"v(s, s_g)"].append(policy_v_1.mean().item())
            debug_info[prefix+"v(s_g, g)"].append(policy_v_2.mean().item())

        # Update network
        self.subgoal_optimizer.zero_grad()
        subgoal_loss.backward()
        if not(self.subgoal_max_grad_norm is None):
                if self.subgoal_max_grad_norm > 0:
                    th.nn.utils.clip_grad_norm_(self.subgoal_net.parameters(), max_norm=self.subgoal_max_grad_norm)
        self.subgoal_optimizer.step()

    def sample_action_and_KL(self, state, goal):
        batch_size = state.size(0)
        # Sample action, subgoals and KL-divergence
        action_dist = self.actor(state, goal)
        action = action_dist.rsample()

        with th.no_grad():
            subgoal = self.sample_subgoal(state, goal)

        prior_action_dist = self.actor_target(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), subgoal)
        prior_prob = prior_action_dist.log_prob(action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.action_dim)).sum(-1, keepdim=True).exp()
        prior_log_prob = th.log(prior_prob.mean(1) + self.epsilon)
        D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob

        action = th.tanh(action)
        return action, D_KL
    
    def value(self, state, goal):
        _, _, action = self.actor.sample(state, goal)
        V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
        return V
    
    def sample_subgoal(self, state, goal):
        subgoal_distribution = self.subgoal_net(state, goal)
        subgoal = subgoal_distribution.rsample((self.n_ensemble,))
        subgoal = th.transpose(subgoal, 0, 1) # 2048x10x20
        return subgoal

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        actor_losses, critic_losses = [], []
        critic_cost_losses = []
        if self.use_decoder:
            autoencoder_losses = []
        debug_info = {}
        debug_info["subgoal_net_losses"] = []
        debug_info["advs"] = []
        debug_info["Q"] = []
        debug_info["target_subgoal_V"] = []
        debug_info["subgoal_V"] = []
        debug_info["target_Q"] = []
        debug_info["v(s, s_g)"] = []
        debug_info["v(s_g, g)"] = []
        if self.safety:
            debug_info["Q_cost"] = []
            debug_info["target_Q_cost"] = []
            debug_info["lambda_loss"] = []
            debug_info["lambda_multiplier"] = []

        for gradient_step in range(gradient_steps):
            state, action, reward, cost, next_state, done, goal = self.sample_and_preprocess_batch(
                self.custom_replay_buffer, 
                env=self.env,
                batch_size=batch_size,
                device=self.device
            )
            # Sample subgoal candidates uniformly in the replay buffer
            subgoal = th.FloatTensor(self.custom_replay_buffer.random_state_batch(batch_size)).to(self.device)
            
            """ Encode images (if vision-based environment), use data augmentation """
            if self.use_encoder:
                # Stop gradient for subgoal goal and next state
                if self.use_decoder:
                    env_state_decoder = state.clone().detach().to(self.device)
                state = self.encoder(state)
                with th.no_grad():
                    goal = self.encoder(goal)
                    next_state = self.encoder(next_state)
                    subgoal = self.encoder(subgoal)


            """ Critic """
            # Compute target Q
            with th.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state, goal)
                target_Q = self.critic_target(next_state, next_action, goal)
                if self.sac:
                    target_Q -= self.sac_alpha * next_log_prob
                target_Q = th.min(target_Q, -1, keepdim=True)[0]
                target_Q = reward + (1.0-done) * self.gamma*target_Q
                if self.safety:
                    target_Q_cost = self.critic_cost_target(next_state, next_action, goal)
                    if self.safe_critic_behave == "min":
                        target_Q_cost = th.min(target_Q_cost, -1, keepdim=True)[0]
                    elif self.safe_critic_behave == "max":
                        target_Q_cost = th.max(target_Q_cost, -1, keepdim=True)[0]
                    elif self.safe_critic_behave == "mean":
                        target_Q_cost = th.mean(target_Q_cost, -1, keepdim=True)[0]
                    else:
                        assert 1 == 0
                    target_Q_cost = th.clamp(target_Q_cost, min=0.0)
                    target_Q_cost = cost + (1.0-done) * self.gamma*target_Q_cost

            # Compute critic loss
            Q = self.critic(state, action, goal)
            critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()
            critic_losses.append(critic_loss.item())
            debug_info["Q"].append(Q.mean().item())
            debug_info["target_Q"].append(target_Q.mean().item())

            # Optimize the critic
            if self.use_encoder: self.encoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if not(self.critic_max_grad_norm is None):
                if self.critic_max_grad_norm > 0:
                    th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad_norm)
            if self.use_encoder: self.encoder_optimizer.step()
            self.critic_optimizer.step()

            if self.safety:
                # Compute safety critic loss
                Q_cost = self.critic_cost(state, action, goal)
                critic_cost_loss = 0.5 * (Q_cost - target_Q_cost).pow(2).sum(-1).mean()
                critic_cost_losses.append(critic_cost_loss.item())
                debug_info["Q_cost"].append(Q_cost.mean().item())
                debug_info["target_Q_cost"].append(target_Q_cost.mean().item())

                # Optimize the safety critic
                self.critic_cost_optimizer.zero_grad()
                critic_cost_loss.backward()
                if not(self.critic_max_grad_norm is None):
                    if self.critic_max_grad_norm > 0:
                        th.nn.utils.clip_grad_norm_(self.critic_cost.parameters(), max_norm=self.critic_max_grad_norm)
                self.critic_cost_optimizer.step()

            # Optimize autoencoder
            if self.use_decoder:
                y = self.encoder.autoencoder_forward(env_state_decoder)
                autoencoder_loss = self.autoencoder_criterion(env_state_decoder, y)
                autoencoder_losses.append(autoencoder_loss.item())
                self.autoencoder_optimizer.zero_grad()
                autoencoder_loss.backward()
                self.autoencoder_optimizer.step()

            # Stop backpropagation to encoder
            if self.use_encoder:
                state = state.detach()
                goal = goal.detach()
                subgoal = subgoal.detach()
                
            # Optimize the subgoal policy
            if not self.sac:
                if not self.adaptive_collision_reward or (self.adaptive_collision_reward and self.num_timesteps < 300_000):
                    self.train_highlevel_policy(state, goal, subgoal, debug_info) # test
                    if self.sgg_optimizing:
                        self.train_highlevel_policy(state, goal, subgoal, debug_info, sgg_optimizing=True) # test
                else:
                    debug_info["subgoal_net_losses"].append(0)
                    debug_info["advs"].append(0)
                    debug_info["target_subgoal_V"].append(0)
                    debug_info["subgoal_V"].append(0)
                    debug_info["v(s, s_g)"].append(0)
                    debug_info["v(s_g, g)"].append(0)
            
            if self.safety and (self.num_timesteps - 1) % self.update_lambda == 0:
                self.train_lagrangian(state, action, goal, debug_info)

            """ Actor """
            if self.sac:
                action, log_prob, _ = self.actor.sample(state, goal)
            else:
                action, D_KL = self.sample_action_and_KL(state, goal)                
            # Compute actor loss
            Q = self.critic(state, action, goal)
            Q = th.min(Q, -1, keepdim=True)[0]
            if self.safety:
                Q_cost = self.critic_cost(state, action, goal)
                if self.safe_critic_behave == "min":
                    Q_cost = th.min(Q_cost, -1, keepdim=True)[0]
                elif self.safe_critic_behave == "max":
                    Q_cost = th.max(Q_cost, -1, keepdim=True)[0]
                elif self.safe_critic_behave == "mean":
                    Q_cost = th.mean(Q_cost, -1, keepdim=True)[0]
                else:
                    assert 1 == 0
                with th.no_grad():
                    lambda_multiplier = th.nn.functional.softplus(self.lambda_coefficient).detach()
                debug_info["lambda_multiplier"].append(lambda_multiplier.item())
            
            if self.sac:
                if self.safety:
                    actor_loss = (self.sac_alpha * log_prob - Q + lambda_multiplier * Q_cost).mean()
                else:
                    actor_loss = (self.sac_alpha * log_prob - Q).mean()
            else:
                if self.safety:
                    actor_loss = (self.alpha*D_KL - Q + lambda_multiplier * Q_cost).mean()
                else:
                    actor_loss = (self.alpha*D_KL - Q).mean()
            
            actor_losses.append(actor_loss.item())
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if not(self.actor_max_grad_norm is None):
                if self.actor_max_grad_norm > 0:
                    th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad_norm)
            self.actor_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau) # test
                if self.safety:
                    polyak_update(self.critic_cost.parameters(), self.critic_cost_target.parameters(), self.tau)
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))        
        self.logger.record("train/Q", np.mean(debug_info["Q"])) 
        self.logger.record("train/target_Q", np.mean(debug_info["target_Q"]))
        if not self.sac:      
            self.logger.record("train/subgoal_net_loss", np.mean(debug_info["subgoal_net_losses"]))
            self.logger.record("train/adv", np.mean(debug_info["advs"]))
            self.logger.record("train/D_KL", D_KL.mean().item())
            self.logger.record("train/target_subgoal_V", np.mean(debug_info["target_subgoal_V"]))
            self.logger.record("train/subgoal_V", np.mean(debug_info["subgoal_V"]))
            self.logger.record("train/v(s, s_g)", np.mean(debug_info["v(s, s_g)"]))
            self.logger.record("train/v(s_g, g)", np.mean(debug_info["v(s_g, g)"]))
        if self.safety:
            self.logger.record("train/critic_cost_loss", np.mean(critic_cost_losses))    
            self.logger.record("train/Q_cost", np.mean(debug_info["Q_cost"])) 
            self.logger.record("train/target_Q_cost", np.mean(debug_info["target_Q_cost"]))
            self.logger.record("train/lambda_loss", np.mean(debug_info["lambda_loss"]) if len(debug_info["lambda_loss"]) > 0 else 0)
            self.logger.record("train/lambda_multiplier", np.mean(debug_info["lambda_multiplier"]))
        if self.use_decoder:
            self.logger.record("train/autoencoder_loss", np.mean(autoencoder_losses))        

    def save(self, folder, save_optims=False):
        th.save(self.actor.state_dict(),		 folder + "actor.pth")
        th.save(self.critic.state_dict(),		folder + "critic.pth")
        if self.safety:
            th.save(self.critic_cost.state_dict(),		folder + "critic_cost.pth")
        th.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
        if self.use_encoder:
            th.save(self.encoder.state_dict(), folder + "encoder.pth")
        if save_optims:
            th.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
            th.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
            th.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
            if self.use_encoder:
                th.save(self.encoder_optimizer.state_dict(), folder + "encoder_opti")
    
    def load(self, folder, old_version=False, best=True):
        if old_version:
            run_name = ""	
        else:
            run_name = "best_" if best else "last_"
        print(f"load run_name: {run_name}")
        self.actor.load_state_dict(th.load(folder+run_name+"actor.pth", map_location=self.device))
        self.critic.load_state_dict(th.load(folder+run_name+"critic.pth", map_location=self.device))
        if self.safety:
            self.critic_cost.load_state_dict(th.load(folder+run_name+"critic_cost.pth", map_location=self.device))
        self.subgoal_net.load_state_dict(th.load(folder+run_name+"subgoal_net.pth", map_location=self.device))
        if self.use_encoder:
            self.encoder.load_state_dict(th.load(folder+run_name+"encoder.pth", map_location=self.device))

    def _excluded_save_params(self) -> List[str]:
        return super(SafetyRis, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
