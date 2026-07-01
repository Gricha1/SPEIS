import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from polamp_Models import GaussianPolicy, EnsembleCritic, LaplacePolicy, Encoder, LidarPredictor
from utils.data_aug import random_translate
from utils.data_aug import NormalNoise
#from PythonRobotics.PathPlanning.DubinsPath import dubins_path_planner

def normalize_state(new_subgoal, env_state_bounds, validate=False):
	if not validate:
		new_subgoal[:, 0] = new_subgoal[:, 0] / env_state_bounds["x"]
		new_subgoal[:, 1] = new_subgoal[:, 1] / env_state_bounds["y"]
		new_subgoal[:, 2] = new_subgoal[:, 2] / env_state_bounds["theta"]
		new_subgoal[:, 3] = new_subgoal[:, 3] / env_state_bounds["v"]
		new_subgoal[:, 4] = new_subgoal[:, 4] / env_state_bounds["steer"]
	else:
		new_subgoal[0] = new_subgoal[0] / env_state_bounds["x"]
		new_subgoal[1] = new_subgoal[1] / env_state_bounds["y"]
		new_subgoal[2] = new_subgoal[2] / env_state_bounds["theta"]
		new_subgoal[3] = new_subgoal[3] / env_state_bounds["v"]
		new_subgoal[4] = new_subgoal[4] / env_state_bounds["steer"]
	return new_subgoal

class RIS(object):
	def __init__(self, state_dim, action_dim, alpha=0.1, Lambda=0.1,
				 use_decoder=False, use_encoder=False,
				 n_critic=1, 
				 train_sac=False, sac_alpha=0.2,
				 safety=False, safety_add_to_high_policy=False, cost_limit=0.5, update_lambda=1000, 
				 use_dubins_filter = False,
				 n_ensemble=10, gamma=0.99, tau=0.005, target_update_interval=1, 
				 h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, 
				 clip_v_function=-100, max_grad_norm = 4.0, lambda_initialization = 0.1,
				 max_episode_steps = 250,
				 logger=None, device=torch.device("cuda"),
				 env_obs_dim=None, add_ppo_reward=False, add_obs_noise=False, 
				 curriculum_high_policy=False,
				 vehicle_curvature=0.1,
				 lidar_max_dist=None,
				 env_state_bounds={},
				 train_env=None):		

		print(f"lambda_initialization: {lambda_initialization}")
		assert not (use_decoder and not use_encoder), 'cant use decoder without encoder'
		assert add_ppo_reward == False, "didnt implement PPO reward for high level policy"
		assert not safety_add_to_high_policy or (safety_add_to_high_policy and safety)
		# normalize states
		self.env_state_bounds = env_state_bounds
		self.safety = safety
		self.safety_add_to_high_policy = safety_add_to_high_policy
		self.curriculum_high_policy = curriculum_high_policy
		self.stop_train_high_policy = False
		self.max_grad_norm = max_grad_norm
		self.actor_max_grad_norm = 2.0
		self.additional_debug = False
		# SAC
		self.train_ris_with_sac = False
		self.train_sac = train_sac
		self.sac_alpha = sac_alpha
		self.sac_use_v_entropy = False

		# Actor
		self.actor = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
		self.actor_target = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		# Critic
		self.critic 		= EnsembleCritic(state_dim, action_dim, n_Q=n_critic).to(device)
		self.critic_target 	= EnsembleCritic(state_dim, action_dim, n_Q=n_critic).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

		# Safety Critic
		if self.safety:
			self.test_case_soft_critic = False
			self.test_max_critic = True
			self.critic_cost 		= EnsembleCritic(state_dim, action_dim, n_Q=n_critic).to(device)
			self.critic_cost_target = EnsembleCritic(state_dim, action_dim, n_Q=n_critic).to(device)
			self.critic_cost_target.load_state_dict(self.critic_cost.state_dict())
			self.critic_cost_optimizer = torch.optim.Adam(self.critic_cost.parameters(), lr=q_lr)

			self.cost_limit         = cost_limit
			# we should use the timestep_cost_limit
			self.timestep_cost_limit = cost_limit * (1 - gamma ** max_episode_steps) / (1 - gamma) / max_episode_steps
			print(f"timestep_cost_limit: {self.timestep_cost_limit}")
			# print(f"max_episode_steps: {max_episode_steps}")
			self.update_lambda      = update_lambda
			self.lambda_coefficient = torch.tensor(lambda_initialization, requires_grad=True)
			self.lambda_optimizer = torch.optim.Adam([self.lambda_coefficient], lr=5e-4)
			
		# Lidar data predictor
		self.env_state_bounds = env_state_bounds
		self.use_lidar_predictor = False
		if train_sac:
			self.use_lidar_predictor = False
		assert not self.use_lidar_predictor or (self.use_lidar_predictor and not use_encoder)
		if self.use_lidar_predictor:
			self.without_state_goal = True
			self.subgoal_dim = 5
			self.frame_stack = 4
			self.agent_state_dim = state_dim # 176
			self.lidar_data_dim = 39
			self.lidar_predictor = LidarPredictor(subgoal_dim=self.subgoal_dim, 
												  agent_state_dim=self.agent_state_dim, 
												  lidar_data_dim=self.lidar_data_dim, 
												  lidar_max_dist=lidar_max_dist,
												  without_state_goal=self.without_state_goal).to(device)
			self.lidar_predictor_criterion = nn.MSELoss()
			self.lidar_predictor_optimizer = torch.optim.Adam(self.lidar_predictor.predictor.parameters(), lr=enc_lr)
		# Subgoal policy (only for RIS, not for SAC)
		self.high_level_without_frame = False
		self.use_dubins_filter = use_dubins_filter
		self.curvature = vehicle_curvature
		if train_sac:
			self.high_level_without_frame = False
			self.use_dubins_filter = False
			# Don't create subgoal_net for SAC-Lagrangian
			self.subgoal_net = None
			self.subgoal_optimizer = None
		else:
			# Create subgoal_net only for RIS
			if self.high_level_without_frame:
				self.subgoal_net = LaplacePolicy(state_dim=state_dim, 
											goal_dim=self.subgoal_dim if self.use_lidar_predictor else state_dim).to(device)
			else:
				self.subgoal_net = LaplacePolicy(state_dim=state_dim, 
											goal_dim=self.subgoal_dim*self.frame_stack if self.use_lidar_predictor else state_dim).to(device)
			self.subgoal_optimizer = torch.optim.Adam(self.subgoal_net.parameters(), lr=h_lr)

		# Encoder
		self.add_obs_noise = add_obs_noise
		if self.add_obs_noise:
			obs_noise_x = NormalNoise(sigma=1)
			obs_noise_y = NormalNoise(sigma=1)
			obs_noise_theta = NormalNoise(sigma=0.15)
			obs_noise_v = NormalNoise(sigma=1)
			obs_noise_steer = NormalNoise(sigma=0.1)
		self.use_encoder = use_encoder
		self.use_decoder = use_decoder
		if self.use_encoder:
			if not self.use_decoder:
				self.encoder = Encoder(input_dim=env_obs_dim, state_dim=state_dim).to(device)
				self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=enc_lr)
			else:
				# AutoEncoder
				self.encoder = Encoder(input_dim=env_obs_dim, state_dim=state_dim, use_decoder=True).to(device)
				self.encoder_optimizer = torch.optim.Adam(self.encoder.encoder.parameters(), lr=enc_lr)
				self.autoencoder_criterion = nn.MSELoss()
				self.autoencoder_optimizer = torch.optim.Adam(self.encoder.decoder.parameters(), lr=enc_lr)

		# Actor-Critic Hyperparameters
		self.tau = tau
		self.target_update_interval = target_update_interval
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# High-level policy hyperparameters
		self.clip_v_function = clip_v_function
		self.Lambda = Lambda
		self.n_ensemble = n_ensemble

		# Utils
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.logger = logger
		self.total_it = 0

	def save(self, folder, save_optims=False):
		torch.save(self.actor.state_dict(),		 folder + "actor.pth")
		torch.save(self.critic.state_dict(),		folder + "critic.pth")
		if self.safety:
			torch.save(self.critic_cost.state_dict(),		folder + "critic_cost.pth")
		# Only save subgoal_net if it exists (i.e., not for SAC-Lagrangian)
		if self.subgoal_net is not None:
			torch.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
		if self.use_encoder:
			torch.save(self.encoder.state_dict(), folder + "encoder.pth")
		if self.use_lidar_predictor:
			torch.save(self.lidar_predictor.state_dict(), folder + "lidar.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
			if self.subgoal_optimizer is not None:
				torch.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
			if self.use_encoder:
				torch.save(self.encoder_optimizer.state_dict(), folder + "encoder_opti")

	def load(self, folder, old_version=False, best=True):
		if old_version:
			run_name = ""	
		else:
			run_name = "best_" if best else "last_"
		print(f"load run_name: {run_name}")
		self.actor.load_state_dict(torch.load(folder+run_name+"actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder+run_name+"critic.pth", map_location=self.device))
		if self.safety:
			self.critic_cost.load_state_dict(torch.load(folder+run_name+"critic_cost.pth", map_location=self.device))
		# Only load subgoal_net if it exists on disk (SAC/SAC-Lagrangian don't have it)
		subgoal_path = folder + run_name + "subgoal_net.pth"
		if self.subgoal_net is not None and os.path.isfile(subgoal_path):
			self.subgoal_net.load_state_dict(torch.load(subgoal_path, map_location=self.device))
		if self.use_encoder:
			self.encoder.load_state_dict(torch.load(folder+run_name+"encoder.pth", map_location=self.device))
		if self.use_lidar_predictor:
			self.lidar_predictor.load_state_dict(torch.load(folder+run_name+"lidar.pth", map_location=self.device))

	def select_action(self, state, goal):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			if self.use_encoder:
				state = self.encoder(state)
				goal = self.encoder(goal)
			action, _, _ = self.actor.sample(state, goal)
		return action.cpu().data.numpy().flatten()
	
	def select_deterministic_action(self, state, goal):
		# print("select_deterministic_action")
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			if self.use_encoder:
				state = self.encoder(state)
				goal = self.encoder(goal)
			_, _, mean_action = self.actor.sample(state, goal)

		return mean_action.cpu().data.numpy().flatten()
		
	def value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
		return V

	# if self.safety
	def safety_value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic_cost(state, action, goal).min(-1, keepdim=True)[0]
		return V

	def trajectory_length(self, x_array, y_array):
		length = 0.0
		for i in range(len(x_array) - 1):
			x1, y1 = x_array[i], y_array[i]
			x2, y2 = x_array[i + 1], y_array[i + 1]
			segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
			length += segment_length
		return length

	def dubins_distance(self, state, subgoal, goal):
		"""
		Function to compute dubins paths length
		"""
		_, lengths_to_subgoal = dubins_path_planner.get_dubins_path_length(
				state[0].item(), state[1].item(), state[2].item(), 
				subgoal[0].item(), subgoal[1].item(), subgoal[2].item(), self.curvature)
		_, lengths_to_goal = dubins_path_planner.get_dubins_path_length(
				subgoal[0].item(), subgoal[1].item(), subgoal[2].item(), 
				goal[0].item(), goal[1].item(), goal[2].item(), self.curvature)

		return sum(lengths_to_subgoal) + sum(lengths_to_goal)

	def dubins_filter_subgoals(self, state, subgoals, goal, num_subgoals):
		"""
		Function to take first num_subgoals subgoals with min dubins_distance

		Arguments
		- subgoals(torch.tensor()): dim = [B, N, L], 
									B = batch size
									N = num of subgoals for D_KL estimation
									L = encoded state dim
		"""
		assert len(subgoals.shape) == 3
		
		with torch.no_grad():
			# decode subgoals
			subgoals = self.encoder.decoder(subgoals) # 2048 x 10 x 176

			filtred_subgoals = torch.ones(subgoals.shape[0], num_subgoals, subgoals.shape[2]).to(self.device)
			init_dubins_distance = 0
			filtred_dubins_dinstance = 0
			for idx, subgoals_for_estimation in enumerate(subgoals):
				subgoals_distances_for_estimation = []
				subgoals_distances_for_estimation = [(idx_, self.dubins_distance(state[idx], subgoal, goal[idx])) for idx_, subgoal in enumerate(subgoals_for_estimation)]
				subgoals_distances_for_estimation = sorted(subgoals_distances_for_estimation, key=lambda x: x[1])
				sorted_subgoal_indexes = [x[0] for x in subgoals_distances_for_estimation]
				test_subgoals = torch.index_select(subgoals_for_estimation, 0, torch.tensor(sorted_subgoal_indexes).to(self.device))
				test_subgoals = test_subgoals[0:num_subgoals, :]
				filtred_subgoals[idx] = test_subgoals
				init_dubins_distance += sum(x[1] for x in subgoals_distances_for_estimation)
				filtred_dubins_dinstance += sum(x[1] for x in subgoals_distances_for_estimation[:num_subgoals])

			# encode subgoals
			filtred_subgoals = self.encoder(filtred_subgoals)
		
		return filtred_subgoals, init_dubins_distance, filtred_dubins_dinstance

	def filter_predicted_subgoal(self, subgoal):
		if len(subgoal.shape) == 4:
			subgoal[:, :, :, 2] = torch.clamp(subgoal[:, :, :, 2], min=self.env_state_bounds["theta"][0], max=self.env_state_bounds["theta"][1])
			subgoal[:, :, :, 3] = torch.clamp(subgoal[:, :, :, 3], min=self.env_state_bounds["v"][0], max=self.env_state_bounds["v"][1])
			subgoal[:, :, :, 4] = torch.clamp(subgoal[:, :, :, 4], min=self.env_state_bounds["steer"][0], max=self.env_state_bounds["steer"][1])
		elif len(subgoal.shape) == 3:
			subgoal[:, :, 2] = torch.clamp(subgoal[:, :, 2], min=self.env_state_bounds["theta"][0], max=self.env_state_bounds["theta"][1])
			subgoal[:, :, 3] = torch.clamp(subgoal[:, :, 3], min=self.env_state_bounds["v"][0], max=self.env_state_bounds["v"][1])
			subgoal[:, :, 4] = torch.clamp(subgoal[:, :, 4], min=self.env_state_bounds["steer"][0], max=self.env_state_bounds["steer"][1])
		else:
			assert 1 == 0
		return subgoal
	def sample_subgoal(self, state, goal):
		subgoal_distribution = self.subgoal_net(state, goal)
		subgoal = subgoal_distribution.rsample((self.n_ensemble,))
		subgoal = torch.transpose(subgoal, 0, 1) # 2048x10x20
		if self.high_level_without_frame: # subgoal = 2048x10x5
			subgoal = subgoal.repeat(1, 1, 4) # 2048x10x20
		if self.use_lidar_predictor:
			batch_size = state.shape[0]
			n_subgoals = subgoal.shape[1]
			subgoal = subgoal.view(batch_size, n_subgoals, self.frame_stack, self.subgoal_dim) # 2048x10x4x5
			subgoal = self.filter_predicted_subgoal(subgoal)
			new_subgoal_lidar_data = self.lidar_predictor(subgoal, 
						state.unsqueeze(1).unsqueeze(2).expand(batch_size, n_subgoals, self.frame_stack, self.agent_state_dim), 
						goal.unsqueeze(1).unsqueeze(2).expand(batch_size, n_subgoals, self.frame_stack, self.agent_state_dim)
			)
			subgoal = torch.cat([subgoal, new_subgoal_lidar_data], -1).reshape(batch_size, n_subgoals, self.agent_state_dim)
		return subgoal

	def sample_action_and_log_prob(self, state, goal):
		# Sample action and log_prob
		action, log_prob, _ = self.actor.sample(state, goal)
		return action, log_prob

	def sample_action_and_KL(self, state, goal):
		batch_size = state.size(0)
		# Sample action, subgoals and KL-divergence
		action_dist = self.actor(state, goal)
		action = action_dist.rsample()

		# Only use subgoals if subgoal_net exists (i.e., for RIS, not for SAC/SAC-Lagrangian)
		if self.subgoal_net is not None:
			with torch.no_grad():
				subgoal = self.sample_subgoal(state, goal)
			if self.use_dubins_filter:
				subgoal, init_dubins_distance, filtred_dubins_dinstance = self.dubins_filter_subgoals(state, subgoal, goal, 5)
			if self.logger is not None:
				self.logger.store(
					init_dubins_distance = init_dubins_distance if self.use_dubins_filter else 0,
					filtred_dubins_dinstance = filtred_dubins_dinstance if self.use_dubins_filter else 0,
				)
			
			prior_action_dist = self.actor_target(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), subgoal)
			prior_prob = prior_action_dist.log_prob(action.unsqueeze(1).expand(batch_size, subgoal.size(1), self.action_dim)).sum(-1, keepdim=True).exp()
			prior_log_prob = torch.log(prior_prob.mean(1) + self.epsilon)
			D_KL = action_dist.log_prob(action).sum(-1, keepdim=True) - prior_log_prob
		else:
			# For SAC/SAC-Lagrangian: D_KL = 0 (no subgoal-based prior)
			D_KL = torch.zeros(batch_size, 1).to(self.device)

		action = torch.tanh(action)
		return action, D_KL
	
	def train_lidar_predictor(self, env_state, env_subgoal, env_goal):
		subgoal_from_state = env_state[:, 0:self.subgoal_dim]
		subgoal_from_subgoal = env_subgoal[:, 0:self.subgoal_dim]
		subgoal_from_goal = env_goal[:, 0:self.subgoal_dim]
		lidar_data_state = env_state[:, self.subgoal_dim:self.subgoal_dim+self.lidar_data_dim]
		lidar_data_subgoal = env_subgoal[:, self.subgoal_dim:self.subgoal_dim+self.lidar_data_dim]
		lidar_data_goal = env_goal[:, self.subgoal_dim:self.subgoal_dim+self.lidar_data_dim]
		# state
		y = self.lidar_predictor(subgoal_from_state, env_state, env_goal)
		lidar_predictor_loss_state = self.lidar_predictor_criterion(lidar_data_state, y)
		self.lidar_predictor_optimizer.zero_grad()
		lidar_predictor_loss_state.backward()
		self.lidar_predictor_optimizer.step()
		# subgoal
		y = self.lidar_predictor(subgoal_from_subgoal, env_subgoal, env_goal)
		lidar_predictor_loss_target_subgoal = self.lidar_predictor_criterion(lidar_data_subgoal, y)
		self.lidar_predictor_optimizer.zero_grad()
		lidar_predictor_loss_target_subgoal.backward()
		self.lidar_predictor_optimizer.step()
		# goal
		y = self.lidar_predictor(subgoal_from_goal, env_state, env_goal)
		lidar_predictor_loss_goal = self.lidar_predictor_criterion(lidar_data_goal, y)
		self.lidar_predictor_optimizer.zero_grad()
		lidar_predictor_loss_goal.backward()
		self.lidar_predictor_optimizer.step()
		if self.logger is not None:
			self.logger.store(
				lidar_predictor_loss_state = lidar_predictor_loss_state.mean().item(),
				lidar_predictor_loss_target_subgoal = lidar_predictor_loss_target_subgoal.mean().item(),
				lidar_predictor_loss_goal = lidar_predictor_loss_goal.mean().item(),
			)
	
	def add_lidar_data_to_subgoals(self, new_subgoal, state, goal, output_info=False):
		if output_info:
			info = {}
		batch_size = state.shape[0] # 2048
		new_subgoal = new_subgoal.view(batch_size, self.frame_stack, self.subgoal_dim) # 2048 x 4 x 5
		new_subgoal = self.filter_predicted_subgoal(new_subgoal)
		if output_info:
			info["predicted_subgoal_x_min"] = new_subgoal[:, :, 0].min().item()
			info["predicted_subgoal_x_max"] = new_subgoal[:, :, 0].max().item()
			info["predicted_subgoal_y_min"] = new_subgoal[:, :, 1].min().item()
			info["predicted_subgoal_y_max"] = new_subgoal[:, :, 1].max().item()
			info["predicted_subgoal_theta_min"] = new_subgoal[:, :, 2].min().item()
			info["predicted_subgoal_theta_max"] = new_subgoal[:, :, 2].max().item()
			info["predicted_subgoal_v_min"] = new_subgoal[:, :, 3].min().item()
			info["predicted_subgoal_v_max"] = new_subgoal[:, :, 3].max().item()
			info["predicted_subgoal_steer_min"] = new_subgoal[:, :, 4].min().item()
			info["predicted_subgoal_steer_max"] = new_subgoal[:, :, 4].max().item()
		new_subgoal_lidar_data = self.lidar_predictor(new_subgoal, 
				state.unsqueeze(1).expand(batch_size, self.frame_stack, self.agent_state_dim), 
				goal.unsqueeze(1).expand(batch_size, self.frame_stack, self.agent_state_dim)
		)
		new_subgoal = torch.cat([new_subgoal, new_subgoal_lidar_data], -1)
		new_subgoal = new_subgoal.view(batch_size, self.agent_state_dim) # 2048 x 176
		if output_info:
			new_subgoal_lidar_data = new_subgoal_lidar_data.view(batch_size, self.frame_stack*self.lidar_data_dim)
			info["predicted_lidar_data_min"] = new_subgoal_lidar_data.min().item()
			info["predicted_lidar_data_max"] = new_subgoal_lidar_data.max().item()
			return new_subgoal, info
		return new_subgoal

	def train_highlevel_policy(self, state, goal, subgoal):
		# Compute subgoal distribution 
		batch_size = state.shape[0] # 2048
		subgoal_distribution = self.subgoal_net(state, goal)
		with torch.no_grad():
			# Compute target value
			new_subgoal = subgoal_distribution.loc # 2048 x 20
			if self.high_level_without_frame: # new_subgoal = 2048 x 5
				new_subgoal = new_subgoal.repeat(1, 4) # 2048 x 20
			if self.use_lidar_predictor:
				new_subgoal, info = self.add_lidar_data_to_subgoals(new_subgoal, state, goal, output_info=True)
			policy_v_1 = self.value(state, new_subgoal)
			policy_v_2 = self.value(new_subgoal, goal)
			policy_v = torch.cat([policy_v_1, policy_v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]
			if self.safety_add_to_high_policy:
				policy_sefety_v_1 = self.safety_value(state, new_subgoal)
				policy_sefety_v_2 = self.safety_value(new_subgoal, goal)
				policy_sefety_v = torch.cat([policy_sefety_v_1, policy_sefety_v_2], -1).max(-1)[0]
				policy_v += policy_sefety_v

			# Compute subgoal distance loss
			v_1 = self.value(state, subgoal)
			v_2 = self.value(subgoal, goal)
			v = torch.cat([v_1, v_2], -1).clamp(min=self.clip_v_function, max=0.0).abs().max(-1)[0]
			if self.safety_add_to_high_policy:
				safety_v_1 = self.safety_value(state, subgoal)
				safety_v_2 = self.safety_value(subgoal, goal)
				safety_v = torch.cat([safety_v_1, safety_v_2], -1).max(-1)[0]
				v += safety_v
			adv = - (v - policy_v)
			weight = F.softmax(adv/self.Lambda, dim=0)
		
		if self.use_lidar_predictor:
			# transform target subgoal
			subgoal = subgoal.view(batch_size, self.frame_stack, -1) # 2048 x 4 x 44
			subgoal = subgoal[:, :, 0:self.subgoal_dim] # 2048 x 4 x 5
			subgoal = subgoal.reshape(batch_size, -1) # 2048 x 20
			if self.high_level_without_frame: 
				subgoal = subgoal[:, 0:self.subgoal_dim] # 2048 x 5

		log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)
		subgoal_loss = - (log_prob * weight).mean()

		# Update network
		self.subgoal_optimizer.zero_grad()
		subgoal_loss.backward()
		self.subgoal_optimizer.step()

		with torch.no_grad():
			subgoal_grad_norm = (
            sum(p.grad.data.norm(2).item() ** 2 for p in self.subgoal_net.parameters() if p.grad is not None) ** 0.5
        	)
		
		# debug subgoal
		if self.additional_debug:
			train_subgoal = {"x_max": new_subgoal[:, 0].max().item(),
							"x_mean": new_subgoal[:, 0].mean().item(), 
							"x_min": new_subgoal[:, 0].min().item(),
							"y_max": new_subgoal[:, 1].max().item(),
							"y_mean": new_subgoal[:, 1].mean().item(), 
							"y_min": new_subgoal[:, 1].min().item(),
							}

		# Log variables
		if self.logger is not None:
			self.logger.store(
				subgoal_grad_norm = subgoal_grad_norm,
				subgoal_weight = weight.mean().item(),
				subgoal_weight_max = weight.max().item(),
				subgoal_weight_min = weight.min().item(),
				log_prob_target_subgoal = log_prob.mean().item(),
				adv = adv.mean().item(),
				ratio_adv = adv.ge(0.0).float().mean().item(),
				subgoal_loss = subgoal_loss.item(),
				high_policy_v = policy_v.mean().item(),
				high_v = v.mean().item(),
				v1_v2_diff = policy_v_1.mean().item() - policy_v_2.mean().item(),
			)

			if self.additional_debug:
				self.logger.store(
					train_subgoal_x_max = train_subgoal["x_max"],
					train_subgoal_x_mean = train_subgoal["x_mean"],
					train_subgoal_x_min = train_subgoal["x_min"],
					train_subgoal_y_max = train_subgoal["y_max"],
					train_subgoal_y_mean = train_subgoal["y_mean"],
					train_subgoal_y_min = train_subgoal["y_min"],
					predicted_lidar_data_min = info["predicted_lidar_data_min"] if self.use_lidar_predictor else 0,
					predicted_lidar_data_max = info["predicted_lidar_data_max"] if self.use_lidar_predictor else 0,
					predicted_subgoal_x_min = info["predicted_subgoal_x_min"] if self.use_lidar_predictor else 0,
					predicted_subgoal_x_max = info["predicted_subgoal_x_max"] if self.use_lidar_predictor else 0,
					predicted_subgoal_y_min = info["predicted_subgoal_y_min"] if self.use_lidar_predictor else 0,
					predicted_subgoal_y_max = info["predicted_subgoal_y_max"] if self.use_lidar_predictor else 0,
					predicted_subgoal_theta_min = info["predicted_subgoal_theta_min"] if self.use_lidar_predictor else 0,
					predicted_subgoal_theta_max = info["predicted_subgoal_theta_max"] if self.use_lidar_predictor else 0,
					predicted_subgoal_v_min = info["predicted_subgoal_v_min"] if self.use_lidar_predictor else 0,
					predicted_subgoal_v_max = info["predicted_subgoal_v_max"] if self.use_lidar_predictor else 0,
					predicted_subgoal_steer_min = info["predicted_subgoal_steer_min"] if self.use_lidar_predictor else 0,
					predicted_subgoal_steer_max = info["predicted_subgoal_steer_max"] if self.use_lidar_predictor else 0,
				)


	# if self.safety
	def train_lagrangian(self, state, action, goal):
		if self.use_encoder:
			with torch.no_grad():
				state = self.encoder(state)
				goal = self.encoder(goal)
		Q_cost = self.critic_cost(state, action, goal)
		Q_cost = torch.min(Q_cost, -1, keepdim=True)[0]
		Q_cost = torch.clamp(Q_cost, min=0.0)
		violation = Q_cost - self.timestep_cost_limit
		lambda_loss =  self.lambda_coefficient * violation.detach()
		#lambda_loss = -lambda_loss.sum(dim=-1)
		lambda_loss = -lambda_loss.mean()
		self.lambda_optimizer.zero_grad()
		lambda_loss.backward()
		self.lambda_optimizer.step()

		if self.logger is not None:
			self.logger.store(
				lambda_coef   = self.lambda_coefficient.item(),
				lambda_loss   = lambda_loss.item(),
			)

	def train(self, state, action, reward, cost, next_state, done, goal, subgoal):
		assert cost.min().item() >= 0, f"batch cost:{cost.min().item()}, cant be negative"
		assert done.min().item() >= 0, f"done{done.min().item()}"

		if self.use_lidar_predictor:
			env_state = state.clone().detach().to(self.device)
			env_subgoal = subgoal.clone().detach().to(self.device)
			env_goal = goal.clone().detach().to(self.device)
		if self.add_obs_noise:
				assert 1 == 0, "didnt implement"
				state = obs_noise_x.perturb_action(state, min_action=-np.inf, max_action=np.inf)
				obs_noise_y = NormalNoise(sigma=1)
				obs_noise_theta = NormalNoise(sigma=0.15)
				obs_noise_v = NormalNoise(sigma=1)
				obs_noise_steer = NormalNoise(sigma=0.1)
				#ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)
				#action = controller_policy.select_action(state, subgoal)
        		#action = ctrl_noise.perturb_action(action, -max_action, max_action)
		""" Encode images (if vision-based environment), use data augmentation """
		if self.use_encoder:
			# Stop gradient for subgoal goal and next state
			if self.use_decoder:
				env_state_decoder = state.clone().detach().to(self.device)
			state = self.encoder(state)
			with torch.no_grad():
				goal = self.encoder(goal)
				next_state = self.encoder(next_state)
				subgoal = self.encoder(subgoal)

		""" Critic """
		# Compute target Q
		with torch.no_grad():
			next_action, log_prob, _ = self.actor.sample(next_state, goal)
			target_Q = self.critic_target(next_state, next_action, goal)
			if self.sac_use_v_entropy or self.train_sac:
				target_Q = torch.min(target_Q, -1, keepdim=True)[0] - self.sac_alpha * log_prob
			else:
				target_Q = torch.min(target_Q, -1, keepdim=True)[0]
			target_Q = reward + (1.0-done) * self.gamma*target_Q
			if self.safety:
				target_Q_cost = self.critic_cost_target(next_state, next_action, goal)
				target_Q_cost = torch.min(target_Q_cost, -1, keepdim=True)[0]
				target_Q_cost = torch.clamp(target_Q_cost, min=0.0)
				target_Q_cost = cost + (1.0-done) * self.gamma*target_Q_cost
		if self.logger is not None:
			self.logger.store(
				log_entropy_critic = log_prob.mean().item() if self.sac_use_v_entropy or self.train_sac else 0,
		)

		# Compute critic loss
		Q = self.critic(state, action, goal)
		critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()
		if self.safety and self.test_case_soft_critic:
			Q_cost = self.critic_cost(state, action, goal)
			critic_cost_loss = 0.5 * (Q_cost - target_Q_cost).pow(2).sum(-1).mean()
			critic_loss += critic_cost_loss

		# Optimize the critic
		if self.use_encoder: self.encoder_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		if self.safety and self.test_case_soft_critic: self.critic_cost_optimizer.zero_grad()
		critic_loss.backward()
		if self.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
		if self.use_encoder: self.encoder_optimizer.step()
		self.critic_optimizer.step()
		if self.safety and self.test_case_soft_critic: self.critic_cost_optimizer.step()

		with torch.no_grad():
			critic_grad_norm = (
            sum(p.grad.data.norm(2).item() ** 2 for p in self.critic.parameters() if p.grad is not None) ** 0.5
        	)

		# Optimize autoencoder
		if self.use_decoder:
			y = self.encoder.autoencoder_forward(env_state_decoder)
			autoencoder_loss = self.autoencoder_criterion(env_state_decoder, y)
			self.autoencoder_optimizer.zero_grad()
			autoencoder_loss.backward()
			self.autoencoder_optimizer.step()

		if self.logger is not None:
			self.logger.store(
				critic_value   = Q.mean().item(),
				target_value  = target_Q.mean().item(),
				autoencoder_loss = autoencoder_loss.item() if self.use_decoder else 0,
			)

		# Stop backpropagation to encoder
		if self.use_encoder:
			state = state.detach()
			goal = goal.detach()
			subgoal = subgoal.detach()

		""" Safety Critic """
		if self.safety and not self.test_case_soft_critic:
			# Compute safety critic loss
			Q_cost = self.critic_cost(state, action, goal)
			critic_cost_loss = 0.5 * (Q_cost - target_Q_cost).pow(2).sum(-1).mean()

			# Optimize the safety critic
			self.critic_cost_optimizer.zero_grad()
			critic_cost_loss.backward()
			if self.max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(self.critic_cost.parameters(), max_norm=self.max_grad_norm)
			self.critic_cost_optimizer.step()

			with torch.no_grad():
				critic_cost_grad_norm = (
				sum(p.grad.data.norm(2).item() ** 2 for p in self.critic_cost.parameters() if p.grad is not None) ** 0.5
				)
			
		if self.safety:
			if self.logger is not None:
				lambda_multiplier = torch.nn.functional.softplus(self.lambda_coefficient)
				self.logger.store(
					safety_critic_value   = Q_cost.mean().item(),
					safety_target_value   = target_Q_cost.mean().item(),
					critic_cost_loss      = critic_cost_loss.item(),
					critic_cost_grad_norm      = critic_cost_grad_norm,
					lambda_multiplier = lambda_multiplier.item(),
				)

		""" High-level policy learning (only for RIS, not for SAC/SAC-Lagrangian) """
		if self.subgoal_net is not None:
			if self.use_lidar_predictor:
				self.train_lidar_predictor(env_state, env_subgoal, env_goal)
			self.train_highlevel_policy(state, goal, subgoal)

		""" Actor """
		# Sample action
		action, D_KL = self.sample_action_and_KL(state, goal)
		if self.train_sac or self.train_ris_with_sac:
			# Sample action and log_prob
			action, log_prob = self.sample_action_and_log_prob(state, goal)

		if self.safety:
			# Compute actor loss + safety
			Q = self.critic(state, action, goal)
			Q = torch.min(Q, -1, keepdim=True)[0]
			lambda_multiplier = torch.nn.functional.softplus(self.lambda_coefficient)
			Q_cost = self.critic_cost(state, action, goal)
			Q_cost = lambda_multiplier * torch.min(Q_cost, -1, keepdim=True)[0]
			if self.train_sac:
				actor_loss = (self.sac_alpha * log_prob - Q + Q_cost).mean()
			elif self.train_ris_with_sac:
				actor_loss = (self.alpha*D_KL + self.sac_alpha * log_prob - Q + Q_cost).mean()
			else:
				actor_loss = (self.alpha*D_KL - Q + Q_cost).mean()
		else:
			# Compute actor loss
			Q = self.critic(state, action, goal)
			Q = torch.min(Q, -1, keepdim=True)[0]
			if self.train_sac:
				actor_loss = (self.sac_alpha * log_prob - Q).mean()
			elif self.train_ris_with_sac:
				actor_loss = (self.alpha*D_KL + self.sac_alpha * log_prob - Q).mean()
			else:
				actor_loss = (self.alpha*D_KL - Q).mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		if self.actor_max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad_norm)
		self.actor_optimizer.step()

		with torch.no_grad():
			actor_grad_norm = (
            sum(p.grad.data.norm(2).item() ** 2 for p in self.actor.parameters() if p.grad is not None) ** 0.5
        	)

		# Update target networks
		self.total_it += 1
		if self.total_it % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			if self.safety:
				for param, target_param in zip(self.critic_cost.parameters(), self.critic_cost_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


		# debug
		if self.additional_debug:
			train_state = {"x_max": state[:, 0].max().item(),
						   "x_mean": state[:, 0].mean().item(), 
						   "x_min": state[:, 0].min().item(),
						   "y_max": state[:, 1].max().item(),
						   "y_mean": state[:, 1].mean().item(), 
						   "y_min": state[:, 1].min().item(),
						  }

			train_goal = {"x_max": goal[:, 0].max().item(),
						   "x_mean": goal[:, 0].mean().item(), 
						   "x_min": goal[:, 0].min().item(),
						   "y_max": goal[:, 1].max().item(),
						   "y_mean": goal[:, 1].mean().item(), 
						   "y_min": goal[:, 1].min().item(),
						  }
			train_subgoal_data = {"x_max": subgoal[:, 0].max().item(),
								  "x_mean": subgoal[:, 0].mean().item(), 
								  "x_min": subgoal[:, 0].min().item(),
								  "y_max": subgoal[:, 1].max().item(),
								  "y_mean": subgoal[:, 1].mean().item(), 
								  "y_min": subgoal[:, 1].min().item(),
								 }

			if self.logger is not None:
				self.logger.store(
					train_state_x_max = train_state["x_max"],
					train_state_x_mean = train_state["x_mean"],
					train_state_x_min = train_state["x_min"],
					train_state_y_max = train_state["y_max"],
					train_state_y_mean = train_state["y_mean"],
					train_state_y_min = train_state["y_min"],
				)
			
				self.logger.store(
					train_goal_x_max = train_goal["x_max"],
					train_goal_x_mean = train_goal["x_mean"],
					train_goal_x_min = train_goal["x_min"],
					train_goal_y_max = train_goal["y_max"],
					train_goal_y_mean = train_goal["y_mean"],
					train_goal_y_min = train_goal["y_min"],
				)

				self.logger.store(
					train_subgoal_data_x_max = train_subgoal_data["x_max"],
					train_subgoal_data_x_mean = train_subgoal_data["x_mean"],
					train_subgoal_data_x_min = train_subgoal_data["x_min"],
					train_subgoal_data_y_max = train_subgoal_data["y_max"],
					train_subgoal_data_y_mean = train_subgoal_data["y_mean"],
					train_subgoal_data_y_min = train_subgoal_data["y_min"],
				)

		train_reward = {"max": reward[:, 0].max().item(),
					  "mean": reward[:, 0].mean().item(), 
					  "min": reward[:, 0].min().item(),
					  }

		if self.logger is not None:
			self.logger.store(
				train_reward_max = train_reward["max"],
				train_reward_mean = train_reward["mean"],
				train_reward_min = train_reward["min"],
			)

		# Log variables
		if self.logger is not None:
			self.logger.store(
				actor_loss   = actor_loss.item(),
				critic_loss  = critic_loss.item(),
				D_KL		 = D_KL.mean().item(),
				alpha        = self.alpha,	
				log_entropy_sac = log_prob.mean().item() if self.train_sac or self.train_ris_with_sac else 0,
				critic_grad_norm = critic_grad_norm,
				actor_grad_norm = actor_grad_norm,
			)
