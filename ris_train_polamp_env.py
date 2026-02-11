import os
import time
import json
import copy
from math import pi, fabs

import torch
import numpy as np
import random
import argparse
import wandb
import comet_ml
import tensorflow as tf
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MPLPath

from polamp_env.lib.utils_operations import normalizeAngle
from utils.logger import Logger
from polamp_RIS import RIS
from polamp_RIS import normalize_state
from polamp_HER import HERReplayBuffer, PathBuilder
from polamp_env.lib.utils_operations import generateDataSet
from polamp_env.lib.structures import State
#from PythonRobotics.PathPlanning.DubinsPath import dubins_path_planner


def evalPolicy(policy, env, 
               plot_full_env=True, plot_subgoals=False, plot_value_function=False,
               value_function_angles=["theta_agent", np.pi, 0, np.pi/2, -np.pi/2],
               plot_safe_bound=False, 
               plot_lidar_predictor=False,
               plot_decoder_agent_states=False,
               plot_subgoal_dispertion=False,
               plot_agent_lidar_data_encoder=False,
               plot_subgoal_lidar_data=False,
               plot_dubins_curve=False,
               add_text=False,
               plot_only_agent_values=False, plot_actions=False, render_env=False, 
               video_validate_tasks = [],                
               data_to_plot={}, dataset_plot=True, 
               eval_strategy=None,
               skip_not_video_tasks=False,
               plot_only_start_position=False,
               dataset_validation=None,
               full_validation=False,
               plot_trajectory=False,
               with_noise=False,
               obs_noise_std=0.01,
               action_noise_std=0.01):
    """
        medium dataset: video_validate_tasks = [("map4", 8), ("map4", 13), ("map6", 5), ("map6", 18), ("map7", 19), ("map5", 7)]
        hard dataset: video_validate_tasks = [("map0", 2), ("map0", 5), ("map0", 10), ("map0", 15)]
    """
    #assert 1.0 * plot_full_env + 1.0 * render_env >= 1, "didnt implement other"
    assert type(video_validate_tasks) == type(list())
    assert not plot_subgoals or (plot_subgoals and policy.use_encoder and policy.use_decoder) or (plot_subgoals and not policy.use_encoder)
    assert (plot_decoder_agent_states and policy.use_decoder) or not plot_decoder_agent_states
    assert not plot_lidar_predictor or (plot_lidar_predictor and policy.use_lidar_predictor)
    print()

    plot_obstacles = env.static_env
    validation_info = {}
    if render_env:
        videos = []   
    if plot_full_env:
        videos = []
        if plot_value_function:
            if plot_only_agent_values:
                fig = plt.figure(figsize=[6.4 * 2, 4.8])
                ax_states = fig.add_subplot(121)
                ax_values_agent = fig.add_subplot(122)
                ax_values_s = [ax_values_agent]
                value_function_angles=["theta_agent"]
            else:
                if len(value_function_angles) == 2:
                    fig = plt.figure(figsize=[6.4, 4.8*3])
                    ax_states = fig.add_subplot(311)
                    ax_values_right = fig.add_subplot(312)
                    ax_values_down = fig.add_subplot(313)
                    ax_values_s = [ax_values_right, ax_values_down]
                elif len(value_function_angles) == 3:
                    fig = plt.figure(figsize=[6.4, 4.8*4])
                    ax_states = fig.add_subplot(411)
                    ax_values_agent = fig.add_subplot(412)
                    ax_values_right = fig.add_subplot(413)
                    ax_values_down = fig.add_subplot(414)
                    ax_values_s = [ax_values_agent, ax_values_right, ax_values_down] 
                else:
                    fig = plt.figure(figsize=[6.4*2, 4.8*3])
                    ax_states = fig.add_subplot(321)
                    ax_values_agent = fig.add_subplot(322)
                    ax_values_left = fig.add_subplot(323)
                    ax_values_right = fig.add_subplot(324)
                    ax_values_down = fig.add_subplot(325)
                    ax_values_up = fig.add_subplot(326)
                    ax_values_s = [ax_values_agent, 
                                ax_values_left, ax_values_right, 
                                ax_values_down, ax_values_up]
        else:
            fig = plt.figure(figsize=[6.4, 4.8])
            ax_states = fig.add_subplot(111)
    state_distrs = {"x": [], "start_x": [], "y": [], "theta": [], "v": [], "steer": []}
    goal_dists = {"goal_x": [], "goal_y": [], "goal_theta": [], "goal_v": [], "goal_steer": []}
    max_state_vals = {}
    min_state_vals = {}
    max_goal_vals = {}
    min_goal_vals = {}
    action_info = {"max_linear_acc": None, "max_steer_rate": None, "min_linear_acc": None, "min_steer_rate": None}
    mean_actions = {"a": [], "v_s": []}
    final_distances = []
    successes = [] 
    acc_rewards = []
    acc_costs = []
    acc_collisions = []
    episode_lengths = []
    task_statuses = []
    lst_min_clearance_distances = []
    lst_mean_clearance_distances = []
    lst_unsuccessful_tasks = []
    if False and (dataset_validation == "cross_dataset_simplified" or dataset_validation == "cross_dataset_balanced"):
        patern_nums = 12 if dataset_validation == "cross_dataset_simplified" else 2
        task_count = 15
        tasks_per_patern = 1
        video_validate_tasks = []
        if full_validation:
            for i in range(len(env.valTasks['map0'])):
               video_validate_tasks.append(("map0", i))
        else:
            for i in range(patern_nums):
                j = i * task_count
                for k in range(tasks_per_patern):
                    video_validate_tasks.append(("map0", j+k))
    dataset_plot_is_already_visualized = False
    # Track subgoal collisions across all episodes
    all_subgoals_collision = []  # List of collision status for all subgoals across all episodes
    for val_key in env.maps.keys():
        eval_tasks = len(env.valTasks[val_key])
        for task_id in range(eval_tasks):  
            need_to_plot_task = (plot_full_env or plot_value_function or render_env) \
                                and (val_key, task_id) in video_validate_tasks
            if skip_not_video_tasks and not need_to_plot_task:
                continue
            if need_to_plot_task:
                images = []
            print(f"map={val_key}", f"task={task_id}", end=" ")
            if need_to_plot_task:
                print("DO VIDEO", end=" ")
            obs = env.reset(id=task_id, val_key=val_key)
            info = {}
            agent = env.environment.agent.current_state
            goal_state = env.environment.agent.goal_state  # Save goal_state before it gets overwritten
            info["agent_state"] = [agent.x, agent.y, agent.theta, agent.v, agent.steer]
            info["goal_state"] = [goal_state.x, goal_state.y, goal_state.theta, goal_state.v, goal_state.steer]
            done = False
            state = obs["observation"]
            goal = obs["desired_goal"]
            
            # Add noise to initial observation if with_noise is enabled
            if with_noise:
                obs_noise = np.random.normal(0, obs_noise_std, size=state.shape)
                state = state + obs_noise
            
            t = 0
            acc_reward = 0
            acc_cost = 0
            acc_collision = 0
            min_clearance_distances = []
            state_distrs["start_x"].append(state[0])
            
            # Collect trajectory for plot_trajectory
            trajectory_x = []
            trajectory_y = []
            if plot_trajectory:
                trajectory_x.append(agent.x)
                trajectory_y.append(agent.y)
            
            # Collect subgoals generated during episode
            subgoals_x = []
            subgoals_y = []
            subgoals_theta = []
            subgoals_collision = []  # Track collision status for each subgoal

            while not done:
                if plot_full_env and need_to_plot_task:
                    if plot_only_start_position and t > 0:
                        done = True
                        break
                    env_min_x = env.dataset_info["min_x"]
                    env_max_x = env.dataset_info["max_x"]
                    env_min_y = env.dataset_info["min_y"]
                    env_max_y = env.dataset_info["max_y"]
                    grid_resolution_x = 20
                    grid_resolution_y = 20

                    with torch.no_grad():
                        to_torch_state = torch.FloatTensor(state).to(policy.device).unsqueeze(0)
                        to_torch_goal = torch.FloatTensor(goal).to(policy.device).unsqueeze(0)
                        if policy.use_encoder:
                            encoded_state = policy.encoder(to_torch_state)
                            encoded_goal = policy.encoder(to_torch_goal)
                        else:
                            encoded_state = to_torch_state
                            encoded_goal = to_torch_goal
                        if plot_subgoals:
                            def generate_subgoals(encoded_state, encoded_goal, subgoals, stds, K=2, add_to_end=True):
                                if K == 0:
                                    return
                                subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
                                subgoal = subgoal_distribution.loc
                                if policy.high_level_without_frame:
                                    subgoal = subgoal.repeat(1, 4)
                                if policy.use_lidar_predictor:
                                    subgoal = policy.add_lidar_data_to_subgoals(subgoal, encoded_state, encoded_goal)
                                std = subgoal_distribution.scale
                                if add_to_end:
                                    subgoals.append(subgoal)
                                    stds.append(std)
                                else:
                                    subgoals.insert(0, subgoal)
                                    stds.insert(0, std)
                                generate_subgoals(encoded_state, subgoal, subgoals, stds, K-1, add_to_end=False)
                                generate_subgoals(subgoal, encoded_goal, subgoals, stds, K-1, add_to_end=True)
                            subgoals = []
                            stds = []
                            generate_subgoals(encoded_state, encoded_goal, subgoals, stds, K=2)
                        
                        x_agent = info["agent_state"][0]
                        y_agent = info["agent_state"][1]
                        theta_agent = info["agent_state"][2]
                        x_goal = info["goal_state"][0]
                        y_goal = info["goal_state"][1]
                        theta_goal = info["goal_state"][2]
                        car_length = 2

                        current_state = env.environment.agent.current_state
                        center_state = env.environment.agent.dynamic_model.shift_state(current_state)
                        agentBB = env.environment.getBB(center_state, ego=True)
                        ax_states.scatter(np.linspace(agentBB[0][0].x, agentBB[0][1].x, 500), 
                                        np.linspace(agentBB[0][0].y, agentBB[0][1].y, 500), 
                                        color="green", s=1)
                        ax_states.scatter(np.linspace(agentBB[1][0].x, agentBB[1][1].x, 500), 
                                        np.linspace(agentBB[1][0].y, agentBB[1][1].y, 500), 
                                        color="green", s=1)
                        ax_states.scatter(np.linspace(agentBB[2][0].x, agentBB[2][1].x, 500), 
                                        np.linspace(agentBB[2][0].y, agentBB[2][1].y, 500), 
                                        color="green", s=1)
                        ax_states.scatter(np.linspace(agentBB[3][0].x, agentBB[3][1].x, 500), 
                                        np.linspace(agentBB[3][0].y, agentBB[3][1].y, 500), 
                                        color="green", s=1)
                        ax_states.set_ylim(bottom=env_min_y, top=env_max_y)
                        ax_states.set_xlim(left=env_min_x, right=env_max_x)
                        ax_states.scatter([x_agent], [y_agent], color="green", s=50)
                        ax_states.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
                                        [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
                                        color="green", s=5)
                        ax_states.scatter([x_goal], [y_goal], color="yellow", s=50)
                        ax_states.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
                                        [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
                                        color="yellow", s=5)
                        if plot_lidar_predictor:
                            lidar_data = policy.lidar_predictor(to_torch_state[:, 0:policy.subgoal_dim], to_torch_state, to_torch_goal).cpu().squeeze()
                            for angle, d in zip(env.environment.angle_space, lidar_data):
                                color_lidar = "green"
                                obst_x = x_agent + d.item() * np.cos(theta_agent + angle)
                                obst_y = y_agent + d.item() * np.sin(theta_agent + angle)
                                ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=5)
                        if add_text:
                            ax_states.text(x_agent + 0.05, y_agent + 0.05, "agent")
                            ax_states.text(x_goal + 0.05, y_goal + 0.05, "goal")
                        if plot_subgoals:
                            for ind, (subgoal, std) in enumerate(zip(subgoals, stds)):
                                if policy.use_encoder:
                                    cuda_decoded_subgoal = policy.encoder.decoder(subgoal)
                                    decoded_subgoal = cuda_decoded_subgoal.cpu()
                                else:
                                    cuda_decoded_subgoal = subgoal
                                    decoded_subgoal = subgoal.cpu()
                                x_subgoal = decoded_subgoal[0][0].item()
                                y_subgoal = decoded_subgoal[0][1].item()
                                theta_subgoal = decoded_subgoal[0][2].item()
                                lidar_data = decoded_subgoal[0][5:44]
                                if policy.use_lidar_predictor:
                                    x_subgoal_1 = decoded_subgoal[0][policy.subgoal_dim+policy.lidar_data_dim].item()
                                    y_subgoal_1 = decoded_subgoal[0][policy.subgoal_dim+policy.lidar_data_dim+1].item()
                                    x_subgoal_2 = decoded_subgoal[0][2*(policy.subgoal_dim+policy.lidar_data_dim)].item()
                                    y_subgoal_2 = decoded_subgoal[0][2*(policy.subgoal_dim+policy.lidar_data_dim)+1].item()
                                    x_subgoal_3 = decoded_subgoal[0][3*(policy.subgoal_dim+policy.lidar_data_dim)].item()
                                    y_subgoal_3 = decoded_subgoal[0][3*(policy.subgoal_dim+policy.lidar_data_dim)+1].item()
                                if plot_lidar_predictor and ind == len(subgoals) // 2:
                                    lidar_data = policy.lidar_predictor(cuda_decoded_subgoal[:, 0:policy.subgoal_dim], to_torch_state, to_torch_goal).cpu().squeeze()
                                    for angle, d in zip(env.environment.angle_space, lidar_data):
                                        color_lidar = "red"
                                        obst_x = x_subgoal + d.item() * np.cos(theta_subgoal + angle)
                                        obst_y = y_subgoal + d.item() * np.sin(theta_subgoal + angle)
                                        ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=10)
                                if plot_subgoal_lidar_data and ind == len(subgoals) // 2:
                                    for angle, d in zip(env.environment.angle_space, lidar_data):
                                        color_lidar = "orange"
                                        obst_x = x_subgoal + d * np.cos(theta_subgoal + angle)
                                        obst_y = y_subgoal + d * np.sin(theta_subgoal + angle)
                                        ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=10)
                                ax_states.scatter([x_subgoal], [y_subgoal], color="orange", s=50)
                                if policy.use_lidar_predictor and ind == len(subgoals) // 2:
                                    ax_states.scatter([x_subgoal_1], [y_subgoal_1], color="orange", s=15)
                                    ax_states.text(x_subgoal_1, y_subgoal_1, "1")
                                    ax_states.scatter([x_subgoal_2], [y_subgoal_2], color="orange", s=15)
                                    ax_states.text(x_subgoal_2, y_subgoal_2, "2")
                                    ax_states.scatter([x_subgoal_3], [y_subgoal_3], color="orange", s=15)
                                    ax_states.text(x_subgoal_3, y_subgoal_3, "3")
                                ax_states.scatter([np.linspace(x_subgoal, x_subgoal + car_length*np.cos(theta_subgoal), 100)], 
                                                [np.linspace(y_subgoal, y_subgoal + car_length*np.sin(theta_subgoal), 100)], 
                                                color="orange", s=5)
                                if plot_subgoal_dispertion:
                                    if ind == len(subgoals) // 2:   
                                        x_std = std[0][0].item()
                                        y_std = std[0][1].item()
                                        rect = patches.Rectangle((x_subgoal - x_std/2, y_subgoal - y_std/2), x_std, y_std, linewidth=1, edgecolor='r', facecolor='none')
                                        ax_states.add_patch(rect)

                                if plot_dubins_curve and ind == len(subgoals) // 2:
                                    R = env.environment.agent.dynamic_model.wheel_base / np.tan(env.environment.agent.dynamic_model.max_steer)
                                    curvature = 1 / R
                                    path_x, path_y, path_yaw, mode, _ = dubins_path_planner.plan_dubins_path(
                                            x_agent, y_agent, theta_agent, x_subgoal, y_subgoal, theta_subgoal, curvature)
                                    ax_states.plot(path_x, path_y)
                                    path_x, path_y, path_yaw, mode, _ = dubins_path_planner.plan_dubins_path(
                                            x_subgoal, y_subgoal, theta_subgoal, x_goal, y_goal, theta_goal, curvature)
                                    ax_states.plot(path_x, path_y)

                        if plot_obstacles:
                            for obstacle in env.environment.obstacle_segments:
                                ax_states.scatter(np.linspace(obstacle[0][0].x, obstacle[0][1].x, 500), 
                                                np.linspace(obstacle[0][0].y, obstacle[0][1].y, 500), 
                                                color="blue", s=1)
                                ax_states.scatter(np.linspace(obstacle[1][0].x, obstacle[1][1].x, 500), 
                                                np.linspace(obstacle[1][0].y, obstacle[1][1].y, 500), 
                                                color="blue", s=1)
                                ax_states.scatter(np.linspace(obstacle[2][0].x, obstacle[2][1].x, 500), 
                                                np.linspace(obstacle[2][0].y, obstacle[2][1].y, 500), 
                                                color="blue", s=1)
                                ax_states.scatter(np.linspace(obstacle[3][0].x, obstacle[3][1].x, 500), 
                                                np.linspace(obstacle[3][0].y, obstacle[3][1].y, 500), 
                                                color="blue", s=1)
                        if not dataset_plot_is_already_visualized and len(data_to_plot) != 0 and dataset_plot:
                            dataset_plot_is_already_visualized = True
                            if "dataset_x" in data_to_plot and "dataset_y" in data_to_plot:
                                ax_states.scatter(data_to_plot["dataset_x"], 
                                                data_to_plot["dataset_y"], 
                                                color="green", s=5)
                            if "train_step_x" in data_to_plot and "train_step_y" in data_to_plot:
                                ax_states.scatter(data_to_plot["train_step_x"], 
                                                data_to_plot["train_step_y"], 
                                                color="red", s=3)
                        ax_states.text(env_max_x - 46, env_max_y - 2, f"a:{round(env.environment.agent.action[0], 2)}")
                        ax_states.text(env_max_x - 39, env_max_y - 2, f"w:{round(env.environment.agent.action[1], 2)}")
                        ax_states.text(env_max_x - 32, env_max_y - 2, f"v:{round(env.environment.agent.current_state.v, 2)}")
                        ax_states.text(env_max_x - 25, env_max_y - 2, f"st:{round(env.environment.agent.current_state.steer * 180 / pi, 1)}")
                        ax_states.text(env_max_x - 18, env_max_y - 2, f"R:{int(acc_reward*10)/10}")
                        ax_states.text(env_max_x - 11, env_max_y - 2, f"C:{int(acc_cost*10)/10}")
                        ax_states.text(env_max_x - 4, env_max_y - 2, f"t:{t}")

                        if plot_decoder_agent_states:
                            decoded_state = policy.encoder.decoder(encoded_state).cpu()
                            x_decoded_state = decoded_state[0][0]
                            y_decoded_state = decoded_state[0][1]
                            theta_decoded_state = decoded_state[0][2]
                            lidar_data = decoded_state[0][5:44]
                            if plot_agent_lidar_data_encoder:
                                for angle, d in zip(env.environment.angle_space, lidar_data):
                                    color_lidar = "green"
                                    obst_x = x_decoded_state + d * np.cos(theta_decoded_state + angle)
                                    obst_y = y_decoded_state + d * np.sin(theta_decoded_state + angle)
                                    ax_states.scatter([obst_x], [obst_y], color=color_lidar, s=10)
                                ax_states.scatter([x_decoded_state], [y_decoded_state], color="red", s=50)
                                ax_states.scatter([np.linspace(x_decoded_state, x_decoded_state + car_length*np.cos(theta_decoded_state), 100)], 
                                                [np.linspace(y_decoded_state, y_decoded_state + car_length*np.sin(theta_decoded_state), 100)], 
                                                color="red", s=5)

                        # values plot
                        if plot_value_function:
                            def plot_values(ax_values, theta, set_lim_xy=True, return_cb=True):
                                if set_lim_xy:
                                    ax_values.set_ylim(bottom=env_min_y, top=env_max_y)
                                    ax_values.set_xlim(left=env_min_x, right=env_max_x)
                                grid_states = []              
                                grid_goals = []
                                grid_dx = (env_max_x - env_min_x) / grid_resolution_x
                                grid_dy = (env_max_y - env_min_y) / grid_resolution_y
                                for grid_state_y in np.linspace(env_min_y + grid_dy/2, env_max_y - grid_dy/2, grid_resolution_y):
                                    for grid_state_x in np.linspace(env_min_x + grid_dx/2, env_max_x - grid_dx/2, grid_resolution_x):
                                        if env.add_frame_stack:
                                            agent_state = [grid_state_x, grid_state_y, theta, 0, 0]
                                            grid_state = []
                                            if env.use_lidar_data:
                                                grid_state_struct = copy.deepcopy(env.environment.agent.current_state)
                                                grid_state_struct.x = grid_state_x
                                                grid_state_struct.y = grid_state_y
                                                grid_state_struct.theta = theta
                                                grid_state_struct.steer = 0
                                                grid_state_struct.v = 0
                                                beams_observation = env.environment.get_observation_without_env_change(grid_state_struct)
                                                agent_state.extend(beams_observation.tolist())
                                            for _ in range(env.frame_stack):
                                                grid_state.extend(agent_state)
                                        else:
                                            assert 1 == 0, "something incorrect here"
                                            grid_state = [grid_state_x, grid_state_y]
                                            grid_state.extend([theta])
                                            grid_state.extend([0 for _ in range(5 - 3)])
                                        grid_states.append(grid_state)
                                grid_states = torch.FloatTensor(np.array(grid_states)).to(policy.device)
                                assert type(grid_states) == type(encoded_state), f"{type(grid_states)} == {type(encoded_state)}"                
                                numpy_encoded_goal = encoded_goal.cpu().squeeze().numpy()
                                assert type(goal) == type(numpy_encoded_goal)
                                numpy_grid_goals = np.array([numpy_encoded_goal if policy.use_encoder else goal 
                                                for _ in range(grid_resolution_x * grid_resolution_y)])
                                grid_goals = torch.FloatTensor(numpy_grid_goals).to(policy.device)
                                if policy.use_encoder:
                                    grid_states = policy.encoder(grid_states)
                                assert grid_goals.shape == grid_states.shape, \
                                       f"doesnt equal state shape: {grid_states.shape}   to goal shape: {grid_goals.shape} "
                                grid_vs = policy.value(grid_states, grid_goals)
                                grid_vs = grid_vs.detach().cpu().numpy().reshape(grid_resolution_x, grid_resolution_y)[::-1]                
                                img = ax_values.imshow(grid_vs, extent=[env_min_x,env_max_x, env_min_y,env_max_y])
                                if return_cb:
                                    cb = fig.colorbar(img)
                                else:
                                    cb = None
                                ax_values.scatter([np.linspace(env_max_x - 3.5, env_max_x - 3.5 + car_length*np.cos(theta), 100)], 
                                                [np.linspace(env_max_y - 1.5, env_max_y - 1.5 + car_length*np.sin(theta), 100)], 
                                                color="black", s=5)
                                ax_values.scatter([env_max_x - 3.5], [env_max_y - 1.5], color="black", s=40)
                                ax_values.scatter([x_agent], [y_agent], color="green", s=100)
                                ax_values.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
                                                [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
                                                color="black", s=5)
                                if plot_subgoals:
                                    for ind, (subgoal, std) in enumerate(zip(subgoals, stds)):
                                        if policy.use_encoder:
                                            cuda_decoded_subgoal = policy.encoder.decoder(subgoal)
                                            decoded_subgoal = cuda_decoded_subgoal.cpu()
                                        else:
                                            cuda_decoded_subgoal = subgoal
                                            decoded_subgoal = subgoal.cpu()
                                        x_subgoal = decoded_subgoal[0][0]
                                        y_subgoal = decoded_subgoal[0][1]
                                        theta_subgoal = decoded_subgoal[0][2]
                                        ax_values.scatter([x_subgoal], [y_subgoal], color="orange", s=50)
                                        ax_values.scatter([np.linspace(x_subgoal, x_subgoal + car_length*np.cos(theta_subgoal), 100)], 
                                                        [np.linspace(y_subgoal, y_subgoal + car_length*np.sin(theta_subgoal), 100)], 
                                                        color="orange", s=5)
                                        if plot_subgoal_dispertion:
                                            if ind == len(subgoals) // 2:   
                                                x_std = std[0][0].item()
                                                y_std = std[0][1].item()
                                                rect = patches.Rectangle((x_subgoal - x_std/2, y_subgoal - y_std/2), x_std, y_std, linewidth=1, edgecolor='r', facecolor='none')
                                                ax_values.add_patch(rect)
                                ax_values.scatter([x_goal], [y_goal], color="yellow", s=100)
                                ax_values.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
                                                [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
                                                color="black", s=5)

                                return cb

                        if plot_value_function:
                            cbs = [plot_values(ax, theta=theta_agent) if theta=="theta_agent" else plot_values(ax, theta=theta) 
                                    for ax, theta in zip(ax_values_s, value_function_angles)]
                        fig.canvas.draw()
                        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        images.append(data)
                        if plot_value_function:
                            for cb in cbs:
                                cb.remove()
                            for ax_values in ax_values_s:
                                ax_values.clear()
                        ax_states.clear()

                state_distrs["x"].append(state[0])
                state_distrs["y"].append(state[1])
                state_distrs["theta"].append(state[2])
                state_distrs["v"].append(state[3])
                state_distrs["steer"].append(state[4])

                goal_dists["goal_x"].append(goal[0])
                goal_dists["goal_y"].append(goal[1])
                goal_dists["goal_theta"].append(goal[2])
                goal_dists["goal_v"].append(goal[3])
                goal_dists["goal_steer"].append(goal[4])
                
                # Generate and save subgoal position before selecting action
                # Only for RIS (not for SAC-Lagrangian)
                if hasattr(policy, 'subgoal_net') and policy.subgoal_net is not None:
                    with torch.no_grad():
                        to_torch_state = torch.FloatTensor(state).to(policy.device).unsqueeze(0)
                        to_torch_goal = torch.FloatTensor(goal).to(policy.device).unsqueeze(0)
                        if policy.use_encoder:
                            encoded_state = policy.encoder(to_torch_state)
                            encoded_goal = policy.encoder(to_torch_goal)
                        else:
                            encoded_state = to_torch_state
                            encoded_goal = to_torch_goal
                        
                        # Generate subgoal
                        subgoal_distribution = policy.subgoal_net(encoded_state, encoded_goal)
                        subgoal = subgoal_distribution.loc
                        if policy.high_level_without_frame:
                            subgoal = subgoal.repeat(1, 4)
                        if policy.use_lidar_predictor:
                            subgoal = policy.add_lidar_data_to_subgoals(subgoal, encoded_state, encoded_goal)
                        
                        # Decode subgoal to get position (same logic as in plot_subgoals)
                        if policy.use_encoder:
                            cuda_decoded_subgoal = policy.encoder.decoder(subgoal)
                            decoded_subgoal = cuda_decoded_subgoal.cpu()
                        else:
                            cuda_decoded_subgoal = subgoal
                            decoded_subgoal = subgoal.cpu()
                        
                        # Extract x, y, theta from decoded subgoal
                        subgoal_x = decoded_subgoal[0][0].item()
                        subgoal_y = decoded_subgoal[0][1].item()
                        subgoal_theta = decoded_subgoal[0][2].item()
                        
                        # Save positions only if plot_trajectory is True
                        if plot_trajectory:
                            subgoals_x.append(subgoal_x)
                            subgoals_y.append(subgoal_y)
                            subgoals_theta.append(subgoal_theta)
                        
                        # Check if subgoal is in collision with obstacles (only for RIS)
                        # Save current agent state
                        original_state = env.environment.agent.current_state
                        # Create subgoal state (use default v and steer from original state)
                        subgoal_state = State([subgoal_x, subgoal_y, subgoal_theta, original_state.v, original_state.steer])
                        # Temporarily set agent state to subgoal to check collision
                        env.environment.agent.current_state = subgoal_state
                        # Check collision
                        is_collision = env.environment.is_robot_in_collision()
                        # Restore original state
                        env.environment.agent.current_state = original_state
                        
                        subgoals_collision.append(is_collision)
                
                if eval_strategy is None:
                    action = policy.select_deterministic_action(state, goal)
                else:
                    print("EVAL ACTION = ", eval_strategy)
                    action = eval_strategy
                
                # Add noise to action if with_noise is enabled
                if with_noise:
                    action_noise = np.random.normal(0, action_noise_std, size=action.shape)
                    action = action + action_noise
                    # Clip action to valid range [-1, 1]
                    action = np.clip(action, -1.0, 1.0)
                
                if action_info["max_linear_acc"] is None:
                    action_info["max_linear_acc"] = action[0]
                    action_info["max_steer_rate"] = action[1]
                    action_info["min_linear_acc"] = action[0]
                    action_info["min_steer_rate"] = action[1]
                else:
                    action_info["max_linear_acc"] = max(action_info["max_linear_acc"], action[0])
                    action_info["max_steer_rate"] = max(action_info["max_steer_rate"], action[1])
                    action_info["min_linear_acc"] = min(action_info["min_linear_acc"], action[0])
                    action_info["min_steer_rate"] = min(action_info["min_steer_rate"], action[1])
                mean_actions["a"].append(action[0])
                mean_actions["v_s"].append(action[1])

                next_obs, reward, done, info = env.step(action)
                if full_validation:
                    min_clearance_distances.append(np.min(env.beams_observation))
                acc_reward += reward
                acc_cost += info["cost"]
                acc_collision += 1.0 * ("Collision" in info)
                
                next_state = next_obs["observation"]
                
                # Add noise to observation if with_noise is enabled
                if with_noise:
                    obs_noise = np.random.normal(0, obs_noise_std, size=next_state.shape)
                    next_state = next_state + obs_noise
                
                state = next_state
                
                # Collect trajectory coordinates
                if plot_trajectory:
                    current_agent = env.environment.agent.current_state
                    trajectory_x.append(current_agent.x)
                    trajectory_y.append(current_agent.y)

                if render_env and need_to_plot_task:
                    images.append(env.render())
                t += 1

            if need_to_plot_task:
                if plot_full_env and render_env:
                    images_full_env = [image for ind, image in enumerate(images) if (ind+1) % 2 != 0]
                    images_render = [image for ind, image in enumerate(images) if (ind+1) % 2 == 0]
                    videos.append((val_key+"_full_env", task_id, images_full_env))
                    videos.append((val_key+"_render", task_id, images_render))
                else:
                    videos.append((val_key, task_id, images))
            final_distances.append(info["EuclideanDistance"])
            success = 1.0 * info["goal_achieved"]
            if success:
                task_status = "success"
            else:
                task_status = "limit_reached"
            if env.static_env: 
                if "Collision" in info:
                    task_status = "collision"
                    success = 0.0
            successes.append(success)
            episode_lengths.append(info["last_step_num"])
            acc_rewards.append(acc_reward)
            acc_collisions.append(acc_collision)
            acc_costs.append(acc_cost)
            task_statuses.append((val_key, task_id, task_status))
            if full_validation:
                lst_min_clearance_distances.append(np.min(min_clearance_distances))
                lst_mean_clearance_distances.append(np.mean(min_clearance_distances))
                if acc_collision > 0.5 or success < 0.5:
                    lst_unsuccessful_tasks.append((val_key, task_id, task_status))
            
            # Plot trajectory if requested
            if plot_trajectory and len(trajectory_x) > 0:
                env_min_x = env.dataset_info["min_x"]
                env_max_x = env.dataset_info["max_x"]
                env_min_y = env.dataset_info["min_y"]
                env_max_y = env.dataset_info["max_y"]
                
                fig_traj = plt.figure(figsize=[6.4, 4.8])
                ax_traj = fig_traj.add_subplot(111)
                
                # Plot obstacles
                if plot_obstacles:
                    # Use maximum safe distance from safe_vehicle_array for expansion
                    # This defines the boundary where agent starts getting cost (clearance_is_enough = False)
                    if hasattr(env.environment.agent, 'safe_vehicle_array') and len(env.environment.agent.safe_vehicle_array) > 0:
                        expansion_distance = np.max(env.environment.agent.safe_vehicle_array)
                    else:
                        expansion_distance = env.environment.agent.dynamic_model.safe_eps
                    
                    # Reduce expansion by half - take distance between red and blue rectangles and divide by 2
                    expansion_distance = expansion_distance / 4
                    
                    for obstacle in env.environment.obstacle_segments:
                        # Get obstacle corners in order
                        corners = [
                            [obstacle[0][0].x, obstacle[0][0].y],
                            [obstacle[1][0].x, obstacle[1][0].y],
                            [obstacle[2][0].x, obstacle[2][0].y],
                            [obstacle[3][0].x, obstacle[3][0].y]
                        ]
                        
                        # Draw original obstacle boundaries (blue lines)
                        for i in range(4):
                            p1 = corners[i]
                            p2 = corners[(i + 1) % 4]
                            ax_traj.scatter(np.linspace(p1[0], p2[0], 500), 
                                            np.linspace(p1[1], p2[1], 500), 
                                            color="blue", s=1, zorder=2)
                        
                        # Calculate center of obstacle
                        center_x = np.mean([c[0] for c in corners])
                        center_y = np.mean([c[1] for c in corners])
                        
                        # Calculate width and height of original obstacle
                        # Find min and max x, y coordinates
                        x_coords = [c[0] for c in corners]
                        y_coords = [c[1] for c in corners]
                        min_x = np.min(x_coords)
                        max_x = np.max(x_coords)
                        min_y = np.min(y_coords)
                        max_y = np.max(y_coords)
                        
                        width = max_x - min_x
                        height = max_y - min_y
                        
                        # Expand width and height by 2 * expansion_distance (both sides)
                        # expansion_distance is already halved, so this creates a smaller expansion
                        expanded_width = width + 2 * expansion_distance
                        expanded_height = height + 2 * expansion_distance
                        
                        # Calculate expanded rectangle corners (centered at the same center)
                        expanded_min_x = center_x - expanded_width / 2
                        expanded_max_x = center_x + expanded_width / 2
                        expanded_min_y = center_y - expanded_height / 2
                        expanded_max_y = center_y + expanded_height / 2
                        
                        # Create expanded rectangle corners
                        expanded_corners = [
                            [expanded_min_x, expanded_min_y],
                            [expanded_max_x, expanded_min_y],
                            [expanded_max_x, expanded_max_y],
                            [expanded_min_x, expanded_max_y]
                        ]
                        
                        # Draw expanded rectangle boundaries (red lines) - dangerous zone
                        for i in range(4):
                            p1 = expanded_corners[i]
                            p2 = expanded_corners[(i + 1) % 4]
                            # Don't add label to scatter - will add separately for legend
                            ax_traj.scatter(np.linspace(p1[0], p2[0], 500), 
                                            np.linspace(p1[1], p2[1], 500), 
                                            color="red", s=1, zorder=2)
                
                # Add legend entry for Dangerous zone with large marker (only once)
                if plot_obstacles and len(env.environment.obstacle_segments) > 0:
                    ax_traj.plot([], [], marker='o', color='red', markersize=15, 
                               markeredgecolor='red', markeredgewidth=1, linestyle='None', label='Dangerous zone')
                
                # Plot trajectory as orange dots (without label)
                ax_traj.scatter(trajectory_x, trajectory_y, color="orange", s=10, alpha=0.6, zorder=3)
                
                # Add legend entry for Trajectory with large marker
                ax_traj.plot([], [], marker='o', color='orange', markersize=15, 
                           markeredgecolor='orange', markeredgewidth=1, linestyle='None', label='Trajectory')
                
                # Plot all subgoals generated during episode
                if len(subgoals_x) > 0:
                    ax_traj.scatter(subgoals_x, subgoals_y, color="purple", s=50, alpha=0.8, 
                                  marker='*', zorder=4, edgecolors='purple', linewidths=0.5)
                    # Draw direction arrows for subgoals
                    car_length = 0.5
                    for sg_x, sg_y, sg_theta in zip(subgoals_x, subgoals_y, subgoals_theta):
                        arrow_length = car_length * 0.8
                        ax_traj.arrow(sg_x, sg_y, 
                                     arrow_length * np.cos(sg_theta), 
                                     arrow_length * np.sin(sg_theta),
                                     head_width=0.1, head_length=0.1, 
                                     fc='purple', ec='purple', alpha=0.6, zorder=4)
                    
                    # Add legend entry for Subgoals
                    ax_traj.plot([], [], marker='*', color='purple', markersize=15, 
                               markeredgecolor='purple', markeredgewidth=0.5, linestyle='None', label='Subgoals')
                
                # Plot initial position - draw the car (agentBB)
                start_x = agent.x
                start_y = agent.y
                start_theta = agent.theta
                start_state = State([start_x, start_y, start_theta, agent.v, agent.steer])
                center_start_state = env.environment.agent.dynamic_model.shift_state(start_state)
                start_agentBB = env.environment.getBB(center_start_state, ego=True)
                # Draw car bounding box
                ax_traj.scatter(np.linspace(start_agentBB[0][0].x, start_agentBB[0][1].x, 500), 
                               np.linspace(start_agentBB[0][0].y, start_agentBB[0][1].y, 500), 
                               color="green", s=2, zorder=6)
                ax_traj.scatter(np.linspace(start_agentBB[1][0].x, start_agentBB[1][1].x, 500), 
                               np.linspace(start_agentBB[1][0].y, start_agentBB[1][1].y, 500), 
                               color="green", s=2, zorder=6)
                ax_traj.scatter(np.linspace(start_agentBB[2][0].x, start_agentBB[2][1].x, 500), 
                               np.linspace(start_agentBB[2][0].y, start_agentBB[2][1].y, 500), 
                               color="green", s=2, zorder=6)
                ax_traj.scatter(np.linspace(start_agentBB[3][0].x, start_agentBB[3][1].x, 500), 
                               np.linspace(start_agentBB[3][0].y, start_agentBB[3][1].y, 500), 
                               color="green", s=2, zorder=6)
                
                # Draw green arrow for agent direction at start position
                car_length = 1.5
                start_arrow_dx = car_length * np.cos(start_theta)
                start_arrow_dy = car_length * np.sin(start_theta)
                ax_traj.arrow(start_x, start_y, start_arrow_dx, start_arrow_dy, 
                             head_width=0.2, head_length=0.15, fc='green', ec='green', 
                             linewidth=2, zorder=7, length_includes_head=True)
                
                # Plot goal position - yellow circle with black border and green arrow (larger size)
                goal_x = info["goal_state"][0]
                goal_y = info["goal_state"][1]
                goal_theta = info["goal_state"][2]
                
                # Draw yellow circle with black border (larger size)
                circle_radius = 1.0
                circle = patches.Circle((goal_x, goal_y), circle_radius, color='yellow', ec='black', linewidth=2, zorder=7)
                ax_traj.add_patch(circle)
                # Add label separately using plot for proper circle display in legend
                ax_traj.plot([], [], marker='o', color='w', markerfacecolor='yellow', 
                            markeredgecolor='black', markersize=10, markeredgewidth=2, label='Goal')
                
                # Draw green arrow for goal direction
                car_length = 1.5
                goal_arrow_dx = car_length * np.cos(goal_theta)
                goal_arrow_dy = car_length * np.sin(goal_theta)
                ax_traj.arrow(goal_x, goal_y, goal_arrow_dx, goal_arrow_dy, 
                             head_width=0.2, head_length=0.15, fc='green', ec='green', 
                             linewidth=2, zorder=8, length_includes_head=True)
                
                ax_traj.set_ylim(bottom=env_min_y, top=env_max_y)
                ax_traj.set_xlim(left=env_min_x, right=env_max_x)
                ax_traj.set_xlabel("X")
                ax_traj.set_ylabel("Y")
                ax_traj.set_title(f"Trajectory - Map: {val_key}, Task: {task_id}, Status: {task_status}")
                ax_traj.legend()
                ax_traj.grid(False)
                
                # Convert figure to numpy array
                fig_traj.canvas.draw()
                trajectory_image = np.frombuffer(fig_traj.canvas.tostring_rgb(), dtype=np.uint8)
                trajectory_image = trajectory_image.reshape(fig_traj.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig_traj)
                
                # Store trajectory image in validation_info
                if "trajectory" not in validation_info:
                    validation_info["trajectory"] = []
                validation_info["trajectory"].append((val_key, task_id, trajectory_image))
            
            # Collect subgoal collision information for this episode
            if len(subgoals_collision) > 0:
                all_subgoals_collision.extend(subgoals_collision)

    eval_distance = np.mean(final_distances) 
    success_rate = np.mean(successes)
    eval_reward = np.mean(acc_rewards)
    eval_cost = np.mean(acc_costs)
    eval_collisions = np.mean(acc_collisions)
    eval_episode_length = np.mean(episode_lengths)
    eval_min_clearance = 0
    eval_mean_clearance = 0
    if full_validation:
        eval_min_clearance = np.mean(lst_min_clearance_distances)
        eval_mean_clearance = np.mean(lst_mean_clearance_distances)

    for key in state_distrs:
        max_state_vals["max_" + str(key)] = np.max(state_distrs[key])
        min_state_vals["min_" + str(key)] = np.min(state_distrs[key])
        state_distrs[key] = np.mean(state_distrs[key])
    for key in goal_dists:
        max_goal_vals["max_" + str(key)] = np.max(goal_dists[key])
        min_goal_vals["min_" + str(key)] = np.min(goal_dists[key])
        goal_dists[key] = np.mean(goal_dists[key])

    env.close()
    if plot_full_env:
        plt.close()
    if plot_full_env or render_env:
        videos = [(map_name, task_indx, np.transpose(np.array(video), axes=[0, 3, 1, 2])) for map_name, task_indx, video in videos]
    validation_info["task_statuses"] = task_statuses
    validation_info["unsuccessful_tasks"] = lst_unsuccessful_tasks
    if plot_full_env or render_env:
        validation_info["videos"] = videos
    validation_info["action_info"] = action_info
    validation_info["eval_cost"] = eval_cost
    validation_info["eval_collisions"] = eval_collisions
    validation_info["eval_min_clearance"] = eval_min_clearance
    validation_info["eval_mean_clearance"] = eval_mean_clearance
    
    # Compute subgoal collision metric (only for RIS, not for SAC-Lagrangian)
    # Only compute if subgoal_net exists (i.e., for RIS)
    if hasattr(policy, 'subgoal_net') and policy.subgoal_net is not None:
        # Metric is 1.0 if all subgoals are in collision, 0.0 if none are in collision
        if len(all_subgoals_collision) > 0:
            # Fraction of subgoals that are in collision
            subgoal_collision_rate = np.mean(all_subgoals_collision)
            validation_info["eval_subgoal_collision_rate"] = subgoal_collision_rate
            print(f"Subgoal collision metric: {subgoal_collision_rate:.4f} (total subgoals: {len(all_subgoals_collision)})")
        else:
            validation_info["eval_subgoal_collision_rate"] = 0.0
            print("Warning: No subgoals were generated during validation, eval_subgoal_collision_rate set to 0.0")
    # For SAC-Lagrangian, don't compute or add this metric

    return eval_distance, success_rate, eval_reward, \
           [state_distrs, max_state_vals, min_state_vals], \
           [goal_dists, max_goal_vals, min_goal_vals], mean_actions, eval_episode_length, validation_info


def sample_and_preprocess_batch(replay_buffer, env, batch_size=256, device=torch.device("cuda")):
    # Extract 
    batch = replay_buffer.random_batch(batch_size)
    state_batch         = batch["observations"]
    action_batch        = batch["actions"]
    next_state_batch    = batch["next_observations"]
    goal_batch          = batch["resampled_goals"]
    reward_batch        = batch["rewards"]
    done_batch          = batch["terminals"]
    agent_state_batch   = batch["state_observation"]
    goal_state_batch    = batch["state_desired_goal"]
    if env.static_env:
        clearance_is_enough_batch     = batch["clearance_is_enough"]
    if env.static_env and env.add_collision_reward: 
        current_step_batch  = batch["current_step"] 
        collision_batch     = batch["collision"]        
        if env.add_frame_stack:
            current_step_batch  = current_step_batch[:, 0:1]
            collision_batch     = collision_batch[:, 0:1]
    
    # Compute sparse rewards: -1 for all actions until the goal is reached
    reward_batch = np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True)) # distance: next_state to goal
    if env.static_env and env.add_collision_reward:
        if env.add_ppo_reward:
            assert 1 == 0, "didnt implement add ppo reward"
            done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
            reward_batch = env.HER_reward(state=state_batch, action=action_batch, 
                                          next_state=next_state_batch, goal=goal_batch, 
                                          collision=collision_batch, goal_was_reached=done_batch, 
                                          step_counter=current_step_batch)
        else:
            # if the state has zero velocity we can reward agent multiple times
            done_batch   = 1.0 * env.is_terminal_dist * (reward_batch < env.SOFT_EPS) + 1.0 * (np.array(next_state_batch)[:, 3:4] > 0.01)# terminal condition
            #done_batch   = 1.0 * env.is_terminal_dist * (reward_batch < env.SOFT_EPS)# terminal condition
            if env.is_terminal_angle:
                angle_batch = abs(np.vectorize(normalizeAngle)(abs(np.array(next_state_batch - goal_batch)[:, 2:3])))
                done_batch += 1.0 * env.is_terminal_angle * (angle_batch < env.ANGLE_EPS)
            done_batch = 1.0 * collision_batch + (1.0 - 1.0 * collision_batch) * (done_batch // (1.0 * env.is_terminal_dist + 1.0 * env.is_terminal_angle + 1.0))
            #done_batch = 1.0 * collision_batch + (1.0 - 1.0 * collision_batch) * (done_batch // (1.0 * env.is_terminal_dist + 1.0 * env.is_terminal_angle))
            reward_batch = (- np.ones_like(done_batch) * env.abs_time_step_reward) * (1.0 - collision_batch) \
                            + (env.collision_reward) * collision_batch
    else:
        done_batch   = 1.0 * (reward_batch < env.SOFT_EPS) # terminal condition
        reward_batch = - np.ones_like(done_batch) * env.abs_time_step_reward
    
    if env.static_env:
        velocity_array = np.abs(next_state_batch[:, 3:4])
        cost_collision = 100
        if env.use_velocity_constraint_cost:
            risk_velocity = env.risk_agent_velocity
            velocity_limit_exceeded = velocity_array >= risk_velocity
            updated_velocity_array = velocity_array * velocity_limit_exceeded
            cost_batch = (np.ones_like(done_batch) * updated_velocity_array) * (1.0 - clearance_is_enough_batch)
            cost_batch = (1.0 - collision_batch) * cost_batch + cost_collision * collision_batch
        else:
            # every timestep in risk zone will be penalized
            cost_batch = (np.ones_like(done_batch)) * (1.0 - clearance_is_enough_batch)
            cost_batch = (1.0 - collision_batch) * cost_batch + cost_collision * collision_batch
    else:
        cost_batch = (- np.ones_like(done_batch) * 0)
    
    # Scaling
    # if args.scaling > 0.0:
    #     reward_batch = reward_batch * args.scaling
    # check if (collision == 1) then (done == 1)
    if env.static_env and not env.teleport_back_on_collision:
        assert ( (1.0 - 1.0 * collision_batch) + (1.0 * collision_batch) * (1.0 * done_batch) ).all()

    # Convert to Pytorch
    state_batch         = torch.FloatTensor(state_batch).to(device)
    action_batch        = torch.FloatTensor(action_batch).to(device)
    reward_batch        = torch.FloatTensor(reward_batch).to(device)
    cost_batch        = torch.FloatTensor(cost_batch).to(device)
    next_state_batch    = torch.FloatTensor(next_state_batch).to(device)
    done_batch          = torch.FloatTensor(done_batch).to(device)
    goal_batch          = torch.FloatTensor(goal_batch).to(device)

    return state_batch, action_batch, reward_batch, cost_batch, next_state_batch, done_batch, goal_batch

def train(args=None):   
    # Initialize comet_ml_experiment to None
    comet_ml_experiment = None
    
    # chech if hyperparams tuning
    if type(args) == type(argparse.Namespace()):
        hyperparams_tune = False
        if args.using_wandb:
            wandb.init(project=args.wandb_project, config=args, 
                    name="RIS," 
                            + " Lambda: " + str(args.Lambda) + " alpha: " + str(args.alpha) 
                            + " enc_s: " + str(args.state_dim) + " n_ens: " + str(args.n_ensemble))
        if args.using_comet:
            comet_ml.login()
            comet_ml_experiment = comet_ml.start(project_name="speis")
            # Convert args to dict for logging parameters
            if hasattr(args, '__dict__'):
                comet_ml_experiment.log_parameters(vars(args))
            else:
                comet_ml_experiment.log_parameters(args)
    else:
        hyperparams_tune = True
        if args.using_wandb:
            wandb.init(config=args, name="hyperparams_tune_RIS")
            args = wandb.config
    
    print("**************")
    print("state_dim:", args.state_dim)
    print("max_timesteps:", args.max_timesteps)
    print("Lambda:", args.Lambda)
    print("alpha:", args.alpha)
    print("n_ensemble:", args.n_ensemble)

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
    goal_our_env_config["dataset"] = args.dataset
    goal_our_env_config["uniform_feasible_train_dataset"] = args.uniform_feasible_train_dataset
    goal_our_env_config["random_train_dataset"] = args.random_train_dataset
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
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if train_env_name in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    register(
        id=train_env_name,
        entry_point='goal_polamp_env.env:GCPOLAMPEnvironment',
        kwargs={'full_env_name': "polamp_env", "config": args.other_keys}
    )

    env         = gym.make(train_env_name)
    test_env    = gym.make(test_env_name)
    vectorized = True
    action_dim = env.action_space.shape[0]
    env_obs_dim = env.observation_space["observation"].shape[0]
    print(f"env_obs_dim: {env_obs_dim}")
    if args.use_encoder:
        state_dim = args.state_dim 
    else:
        state_dim = env_obs_dim
    folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
    load_results = os.path.isdir(folder)

    # Create logger
    logger = Logger(vars(args), save_git_head_hash=False)
    
    # Initialize policy
    env_state_bounds = {"x": 100, "y": 100, 
                        "theta": (-np.pi, np.pi),
                        "v": (env.environment.agent.dynamic_model.min_vel, 
                              env.environment.agent.dynamic_model.max_vel), 
                        "steer": (-env.environment.agent.dynamic_model.max_steer, 
                                 env.environment.agent.dynamic_model.max_steer)}
    R = env.environment.agent.dynamic_model.wheel_base / np.tan(env.environment.agent.dynamic_model.max_steer)
    curvature = 1 / R
    max_polamp_steps = our_env_config["max_polamp_steps"]
    print(f"max_polamp_steps: {max_polamp_steps}")
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
                 max_episode_steps = max_polamp_steps,
                 device=args.device, logger=logger if args.log_loss else None, 
                 env_obs_dim=env_obs_dim, add_ppo_reward=env.add_ppo_reward,
                 add_obs_noise=args.add_obs_noise,
                 curriculum_high_policy=args.curriculum_high_policy,
                 vehicle_curvature=curvature,
                 env_state_bounds=env_state_bounds,
                 lidar_max_dist=env.environment.MAX_DIST_LIDAR,
                 train_env=env,
    )

    # Initialize replay buffer and path_builder
    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        env=env,
        fraction_goals_are_rollout_goals = args.fraction_goals_are_rollout_goals,
        fraction_resampled_goals_are_env_goals = args.fraction_resampled_goals_are_env_goals,
        fraction_resampled_goals_are_replay_buffer_goals = args.fraction_resampled_goals_are_replay_buffer_goals,
        ob_keys_to_save     =["state_observation", "state_achieved_goal", "state_desired_goal", "current_step", "collision", "clearance_is_enough"],
        desired_goal_keys   =["desired_goal", "state_desired_goal"],
        observation_key     = 'observation',
        desired_goal_key    = 'desired_goal',
        achieved_goal_key   = 'achieved_goal',
        vectorized          = vectorized 
    )
    path_builder = PathBuilder()

    if load_results and not hyperparams_tune:
        policy.load(folder)
        print("weights is loaded")
    else:
        print("WEIGHTS ISN'T LOADED")

    # Initialize environment
    obs = env.reset()
    done = False
    state = obs["observation"]
    goal = obs["desired_goal"]
    
    # Add noise to initial observation if with_noise is enabled
    if args.with_noise:
        obs_noise = np.random.normal(0, args.obs_noise_std, size=state.shape)
        state = state + obs_noise
    
    episode_timesteps = 0
    episode_num = 0 
    old_success_rate = None
    save_policy_count = 0 
    if args.curriculum_alpha:
        saved_final_result = False

    assert args.eval_freq > 250, "logger is erased after each eval"
    logger.store(train_step_x = state[0])
    logger.store(train_step_y = state[1])
    logger.store(train_rate = 1.0*done)
    cumulative_reward = 0
    cumulative_cost = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        if t % 1e4 == 0:
            print("step:", t, end=" ")

        # Select action
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state, goal)
        
        # Add noise to action if with_noise is enabled
        if args.with_noise:
            action_noise = np.random.normal(0, args.action_noise_std, size=action.shape)
            action = action + action_noise
            # Clip action to valid range [-1, 1]
            action = np.clip(action, -1.0, 1.0)

        # Perform action
        next_obs, reward, done, train_info = env.step(action) 
        next_state = next_obs["observation"]
        next_agent_state = next_obs["state_observation"]
        
        # Add noise to observation if with_noise is enabled
        if args.with_noise:
            obs_noise = np.random.normal(0, args.obs_noise_std, size=next_state.shape)
            next_state = next_state + obs_noise
            # Update next_obs with noisy observation
            next_obs["observation"] = next_state
        
        cumulative_reward += reward
        cumulative_cost += train_info["cost"]

        path_builder.add_all(
            observations=obs,
            actions=action,
            rewards=reward,
            next_observations=next_obs,
            terminals=[1.0*done]
        )
        
        state = next_state
        agent_state = next_agent_state
        obs = next_obs
        logger.store(train_step_x = agent_state[0])
        logger.store(train_step_y = agent_state[1])

        # Train agent after collecting enough data
        if t >= args.batch_size and t >= args.start_timesteps:
            start_train = time.time()
            state_batch, action_batch, reward_batch, cost_batch, next_state_batch, done_batch, goal_batch = sample_and_preprocess_batch(
                replay_buffer, 
                env=env,
                batch_size=args.batch_size,
                device=args.device
            )
            # Sample subgoal candidates uniformly in the replay buffer
            subgoal_batch = torch.FloatTensor(replay_buffer.random_state_batch(args.batch_size)).to(args.device)
            policy.train(state_batch, action_batch, reward_batch, cost_batch, next_state_batch, done_batch, goal_batch, subgoal_batch)
            if t % 1e4 == 0:
                print("train", args.exp_name, end=" ")
            if args.safety and t % policy.update_lambda == 0:
                policy.train_lagrangian(state_batch, action_batch, goal_batch)
            end_train = time.time()
            logger.store(train_time = end_train - start_train)
            #print("train_step_time:", end_train - start_train)

        if done: 
            train_success = 1.0 * train_info["goal_achieved"]
            collision_end = 1.0 * ("Collision" in train_info)
            if collision_end:
                logger.store(collision_velocity=agent_state[3])
            # Add path to replay buffer and reset path builder
            replay_buffer.add_path(path_builder.get_all_stacked())
            path_builder = PathBuilder()
            logger.store(t=t, reward=reward)
            logger.store(cumulative_reward=cumulative_reward)
            logger.store(cumulative_cost=cumulative_cost)
            logger.store(train_rate=train_success)
            logger.store(collision_rate=collision_end)
            cumulative_reward = 0
            cumulative_cost = 0
            # Reset environment
            obs = env.reset()
            done = False
            state = obs["observation"]
            goal = obs["desired_goal"]
            
            # Add noise to observation if with_noise is enabled
            if args.with_noise:
                obs_noise = np.random.normal(0, args.obs_noise_std, size=state.shape)
                state = state + obs_noise
            
            episode_timesteps = 0
            episode_num += 1 
            logger.store(dataset_x = state[0])
            logger.store(dataset_y = state[1])
            logger.store(dataset_x = goal[0])
            logger.store(dataset_y = goal[1])

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # Eval policy
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
                                with_noise=args.with_noise,
                                obs_noise_std=args.obs_noise_std,
                                action_noise_std=args.action_noise_std,
                                plot_trajectory=args.plot_trajectory,
                                data_to_plot={"train_step_x": logger.data["train_step_x"], 
                                              "train_step_y": logger.data["train_step_y"],
                                              "dataset_x": logger.data["dataset_x"],
                                              "dataset_y": logger.data["dataset_y"]},
                                #video_validate_tasks = [("map0", 0), ("map0", 1), ("map1", 0), ("map1", 1)],
                                video_validate_tasks = [],
                                #video_validate_tasks = [("map0", 0), ("map0", 1), ("map0", 2), ("map1", 3)],
                                value_function_angles=["theta_agent", 0, -np.pi/2],
                                dataset_plot=True,
                                skip_not_video_tasks=False,
                                dataset_validation=args.dataset)
            
            # Print validation metrics to console
            print("=" * 60)
            print("VALIDATION METRICS (Training):")
            print("=" * 60)
            print(f"Success Rate: {success_rate:.4f}")
            print(f"Collision Rate: {validation_info.get('eval_collisions', 0.0):.4f}")
            print(f"Cost: {validation_info.get('eval_cost', 0.0):.4f}")
            if "eval_subgoal_collision_rate" in validation_info:
                print(f"Subgoal Collision Rate: {validation_info['eval_subgoal_collision_rate']:.4f}")
            print(f"Episode Length: {eval_episode_length:.4f}")
            print(f"Distance: {eval_distance:.4f}")
            print(f"Reward: {eval_reward:.4f}")
            print("=" * 60)
            
            train_success_rate = sum(logger.data["train_rate"]) / len(logger.data["train_rate"])
            train_collision_rate = sum(logger.data["collision_rate"]) / len(logger.data["collision_rate"])
            log_dict = {
                    'steps': logger.data["t"][-1],
                    'train_time': sum(logger.data["train_time"]) / len(logger.data["train_time"]),
                    'train/train_rate': train_success_rate, 
                    'train/collision_rate': train_collision_rate,    
                    'train/collision_velocity': sum(logger.data["collision_velocity"]) / len(logger.data["collision_velocity"]) if "collision_velocity" in logger.data else 0,    
                    'train/avg_cumulative_reward': sum(logger.data["cumulative_reward"]) / len(logger.data["cumulative_reward"]),
                    'train/max_cumulative_reward': max(logger.data["cumulative_reward"]),
                    'train/min_cumulative_reward': min(logger.data["cumulative_reward"]),
                    'train/avg_cumulative_cost': sum(logger.data["cumulative_cost"]) / len(logger.data["cumulative_cost"]),
                    'train/autoencoder_loss': sum(logger.data["autoencoder_loss"][-args.eval_freq:]) / args.eval_freq if policy.use_decoder and "autoencoder_loss" in logger.data and len(logger.data["autoencoder_loss"]) > 0 else 0,
                    # High-level policy metrics (only for RIS, not for SAC/SAC-Lagrangian)
                    'train/v1_v2_diff': sum(logger.data["v1_v2_diff"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "v1_v2_diff" in logger.data and len(logger.data["v1_v2_diff"]) > 0 else 0,
                    'train/high_policy_v': sum(logger.data["high_policy_v"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "high_policy_v" in logger.data and len(logger.data["high_policy_v"]) > 0 else 0,    
                    'train/high_v': sum(logger.data["high_v"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "high_v" in logger.data and len(logger.data["high_v"]) > 0 else 0,   
                    'train/train_adv': sum(logger.data["adv"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "adv" in logger.data and len(logger.data["adv"]) > 0 else 0,    
                    'train/train_D_KL': sum(logger.data["D_KL"][-args.eval_freq:]) / args.eval_freq,
                    'train/subgoal_loss': sum(logger.data["subgoal_loss"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "subgoal_loss" in logger.data and len(logger.data["subgoal_loss"]) > 0 else 0,
                    'train/train_critic_loss': sum(logger.data["critic_loss"][-args.eval_freq:]) / args.eval_freq,
                    'train/critic_cost_loss': sum(logger.data["critic_cost_loss"][-args.eval_freq:]) / args.eval_freq if policy.safety and "critic_cost_loss" in logger.data and len(logger.data["critic_cost_loss"]) > 0 else 0,
                    'train/critic_value': sum(logger.data["critic_value"][-args.eval_freq:]) / args.eval_freq,
                    'train/critic_grad_norm': sum(logger.data["critic_grad_norm"][-args.eval_freq:]) / args.eval_freq,
                    'train/target_value': sum(logger.data["target_value"][-args.eval_freq:]) / args.eval_freq,
                    'train/actor_loss': sum(logger.data["actor_loss"][-args.eval_freq:]) / args.eval_freq,
                    'train/actor_grad_norm': sum(logger.data["actor_grad_norm"][-args.eval_freq:]) / args.eval_freq,
                    'train/safety_critic_value': sum(logger.data["safety_critic_value"][-args.eval_freq:]) / args.eval_freq if policy.safety and "safety_critic_value" in logger.data and len(logger.data["safety_critic_value"]) > 0 else 0,
                    'train/safety_target_value': sum(logger.data["safety_target_value"][-args.eval_freq:]) / args.eval_freq if policy.safety and "safety_target_value" in logger.data and len(logger.data["safety_target_value"]) > 0 else 0,
                    'train/critic_cost_grad_norm': sum(logger.data["critic_cost_grad_norm"][-args.eval_freq:]) / args.eval_freq if policy.safety and "critic_cost_grad_norm" in logger.data and len(logger.data["critic_cost_grad_norm"]) > 0 else 0,
                    'train/lambda_loss': sum(logger.data["lambda_loss"][-args.eval_freq:]) / args.eval_freq if policy.safety and "lambda_loss" in logger.data and len(logger.data["lambda_loss"]) > 0 else 0,
                    # Subgoal metrics (only for RIS, not for SAC/SAC-Lagrangian)
                    'train/subgoal_weight': sum(logger.data["subgoal_weight"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "subgoal_weight" in logger.data and len(logger.data["subgoal_weight"]) > 0 else 0,
                    'train/subgoal_weight_max': sum(logger.data["subgoal_weight_max"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "subgoal_weight_max" in logger.data and len(logger.data["subgoal_weight_max"]) > 0 else 0,
                    'train/subgoal_weight_min': sum(logger.data["subgoal_weight_min"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "subgoal_weight_min" in logger.data and len(logger.data["subgoal_weight_min"]) > 0 else 0,
                    'train/log_prob_target_subgoal': sum(logger.data["log_prob_target_subgoal"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "log_prob_target_subgoal" in logger.data and len(logger.data["log_prob_target_subgoal"]) > 0 else 0,    
                    'train/subgoal_grad_norm': sum(logger.data["subgoal_grad_norm"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "subgoal_grad_norm" in logger.data and len(logger.data["subgoal_grad_norm"]) > 0 else 0,
                     
                     # additional
                     'train/alpha': sum(logger.data["alpha"][-args.eval_freq:]) / args.eval_freq,
                     'train/lambda_coef': sum(logger.data["lambda_coef"]) / len(logger.data["lambda_coef"]) if policy.safety and "lambda_coef" in logger.data else 0,
                     'train/lambda_multiplier': sum(logger.data["lambda_multiplier"]) / len(logger.data["lambda_multiplier"]) if policy.safety and "lambda_multiplier" in logger.data else 0,
                     'train/fraction_goals_rollout_goals': replay_buffer.fraction_goals_rollout_goals,
                     'train/fraction_resampled_goals_replay_buffer_goals': replay_buffer.fraction_resampled_goals_replay_buffer_goals,

                     # SAC
                     'train/log_entropy_sac': sum(logger.data["log_entropy_sac"][-args.eval_freq:]) / args.eval_freq,
                     'train/log_entropy_critic': sum(logger.data["log_entropy_critic"][-args.eval_freq:]) / args.eval_freq,

                     # dubins filter (only for RIS with dubins filter, not for SAC/SAC-Lagrangian)
                     'extra/init_dubins_distance': sum(logger.data["init_dubins_distance"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "init_dubins_distance" in logger.data and len(logger.data["init_dubins_distance"]) > 0 else 0,
                     'extra/filtred_dubins_dinstance': sum(logger.data["filtred_dubins_dinstance"][-args.eval_freq:]) / args.eval_freq if not args.train_sac and "filtred_dubins_dinstance" in logger.data and len(logger.data["filtred_dubins_dinstance"]) > 0 else 0,

                     # validate logging
                     f'validation/val_distance({args.n_eval} episodes)': eval_distance,
                     f'validation/eval_reward({args.n_eval} episodes)': eval_reward,
                     f'validation/eval_cost({args.n_eval} episodes)': validation_info["eval_cost"],
                     f'validation/val_rate({args.n_eval} episodes)': success_rate,
                     f'validation/eval_collisions({args.n_eval} episodes)': validation_info["eval_collisions"],
                     "validation/val_episode_length": eval_episode_length,
                     f'validation/eval_subgoal_collision_rate({args.n_eval} episodes)': validation_info.get("eval_subgoal_collision_rate", 0.0), 
                    } if (args.using_wandb or args.using_comet) else {}
            cur_step = logger.data["t"][-1]
            if args.using_wandb:
                for dict_ in val_state + val_goal:
                    for key in dict_:
                        log_dict[f"{key}"] = dict_[key]
                for map_name, task_indx, video in validation_info["videos"]:
                    log_dict["validation_video"+"_"+map_name+"_"+f"{task_indx}"] = \
                        wandb.Video(video, fps=10, format="gif", caption=f"steps: {cur_step}")
                # Log trajectory images to wandb
                if args.plot_trajectory and "trajectory" in validation_info:
                    import PIL.Image
                    for map_name, task_indx, trajectory_image in validation_info["trajectory"]:
                        if len(trajectory_image.shape) == 3:
                            pil_image = PIL.Image.fromarray(trajectory_image)
                            log_dict[f"validation_trajectory_{map_name}_{task_indx}_step_{cur_step}"] = wandb.Image(pil_image)
                wandb.log(log_dict)
            if args.using_comet and comet_ml_experiment is not None:
                # Add val_state and val_goal metrics for comet_ml
                for dict_ in val_state + val_goal:
                    for key in dict_:
                        log_dict[f"{key}"] = dict_[key]
                
                # Remove 'steps' from metrics dict (it's not a metric, it's the step number)
                # Also remove wandb.Video objects (they're wandb-specific)
                comet_metrics = {k: v for k, v in log_dict.items() 
                                if k != 'steps' 
                                and not (hasattr(v, '__class__') and 'wandb' in str(type(v)))}
                
                # Log metrics to comet_ml with step number
                if len(comet_metrics) > 0:
                    try:
                        comet_ml_experiment.log_metrics(comet_metrics, step=cur_step)
                    except Exception as e:
                        print(f"Warning: Failed to log metrics to Comet ML: {e}")
                # Also log videos if available
                if not args.not_visual_validation and "videos" in validation_info:
                    for map_name, task_indx, video in validation_info["videos"]:
                        cur_step = logger.data["t"][-1]
                        # Convert video format if needed
                        if len(video.shape) == 4:
                            if video.shape[1] == 3 or video.shape[1] == 1:  # (T, C, H, W)
                                video_to_save = np.transpose(video, (0, 2, 3, 1))  # (T, H, W, C)
                            else:
                                video_to_save = video
                        else:
                            video_to_save = video
                        
                        # Ensure video is uint8
                        if video_to_save.dtype != np.uint8:
                            if video_to_save.max() <= 1.0:
                                video_to_save = (video_to_save * 255).astype(np.uint8)
                            else:
                                video_to_save = np.clip(video_to_save, 0, 255).astype(np.uint8)
                        
                        # Save video to temporary file
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                            tmp_video_path = tmp_file.name
                        
                        try:
                            try:
                                import imageio
                                if imageio is not None:
                                    imageio.mimwrite(tmp_video_path, video_to_save, fps=10, codec='libx264')
                                else:
                                    raise ImportError
                            except (ImportError, Exception):
                                import PIL.Image
                                if len(video_to_save.shape) == 3:
                                    frames = [PIL.Image.fromarray(frame, mode='L') for frame in video_to_save]
                                else:
                                    frames = [PIL.Image.fromarray(frame) for frame in video_to_save]
                                frames[0].save(tmp_video_path.replace('.mp4', '.gif'), 
                                             save_all=True, append_images=frames[1:], 
                                             duration=100, loop=0)
                                tmp_video_path = tmp_video_path.replace('.mp4', '.gif')
                            
                            comet_ml_experiment.log_video(
                                tmp_video_path,
                                name=f"validation_video_{map_name}_{task_indx}_step_{cur_step}",
                                overwrite=True
                            )
                        finally:
                            if os.path.exists(tmp_video_path):
                                os.unlink(tmp_video_path)
                # Log trajectory images to comet_ml
                if args.plot_trajectory and "trajectory" in validation_info:
                    for map_name, task_indx, trajectory_image in validation_info["trajectory"]:
                        # Save trajectory image to temporary file
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            tmp_image_path = tmp_file.name
                        
                        try:
                            import PIL.Image
                            pil_image = PIL.Image.fromarray(trajectory_image)
                            pil_image.save(tmp_image_path)
                            
                            comet_ml_experiment.log_image(
                                tmp_image_path,
                                name=f"validation_trajectory_{map_name}_{task_indx}_step_{cur_step}",
                                overwrite=True
                            )
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_image_path):
                                os.unlink(tmp_image_path)
            del log_dict
     
            if args.curriculum_high_policy:
                if train_success_rate >= 0.95:
                    policy.stop_train_high_policy = True

            # stop high policy influence on low policy
            if args.curriculum_alpha:
                if (t + 1) >= args.curriculum_alpha_treshold:
                    policy.alpha = args.curriculum_alpha_val
                    # save results after change alpha
                    if not saved_final_result:
                        folder = "results/{}/RIS/{}_final/".format(args.env, args.exp_name)
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        if not hyperparams_tune:
                            logger.save(folder + "log.pkl")
                            policy.save(folder)
                        saved_final_result = True

            # Save (current) results
            folder = "results/{}/RIS/{}/".format(args.env, args.exp_name) + "last_"
            if not os.path.exists(folder):
                os.makedirs(folder)
            if not hyperparams_tune:
                logger.save(folder + "log.pkl")
                policy.save(folder)
            # Save (best) results
            if old_success_rate is None or success_rate >= old_success_rate:
                old_success_rate = success_rate
                save_policy_count += 1
                folder = "results/{}/RIS/{}/".format(args.env, args.exp_name) + "best_"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                if not hyperparams_tune:
                    logger.save(folder + "log.pkl")
                    policy.save(folder)
                if args.using_wandb:
                    wandb.log({"save_policy_count": save_policy_count})
 
            # clean log buffer
            logger.clear()
            print("eval", end=" ")
        if t % 1e4 == 0 or (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            print()

if __name__ == "__main__":	
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--env",                  default="polamp_env")
    parser.add_argument("--test_env",             default="polamp_env")
    parser.add_argument("--dataset",              default="cross_dataset_test_level_1")
    parser.add_argument("--uniform_feasible_train_dataset", default=False)
    parser.add_argument("--random_train_dataset",           default=False)

    # sac
    parser.add_argument("--train_sac",            default=False, type=bool)

    # validate
    parser.add_argument("--eval_freq",          default=int(3e4), type=int) # 3e4
    parser.add_argument('--not_visual_validation', default=False, action='store_true')
    parser.add_argument('--plot_trajectory', default=False, action='store_true', help='Plot trajectory with subgoals during validation')
    parser.add_argument('--with_noise', default=False, action='store_true', help='Add noise to observations and actions during training')
    parser.add_argument('--obs_noise_std', default=0.01, type=float, help='Standard deviation for observation noise')
    parser.add_argument('--action_noise_std', default=0.01, type=float, help='Standard deviation for action noise')

    # ris
    parser.add_argument("--epsilon",            default=1e-16, type=float)
    parser.add_argument("--n_critic",           default=2, type=int) # 1
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--batch_size",         default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=5e5, type=int) # 5e5
    parser.add_argument("--n_eval",             default=5, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int) # 42
    parser.add_argument("--exp_name",           default="RIS_ant")
    parser.add_argument("--alpha",              default=1.76, type=float)
    parser.add_argument("--Lambda",             default=0.29, type=float) # 0.1
    parser.add_argument("--n_ensemble",         default=10, type=int) # 10
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
    parser.add_argument("--lambda_initialization",  default=1.0, type=float)
    
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
    parser.add_argument("--safety",                    default=True, type=bool)
    parser.add_argument("--cost_limit",                default=5.0, type=float)
    parser.add_argument("--update_lambda",             default=1000, type=int)
    
    # logging
    parser.add_argument("--using_wandb",        default=False, type=bool)
    parser.add_argument("--using_comet",        default=True, type=bool)
    parser.add_argument("--wandb_project",      default="safety_ris", type=str)
    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)

    args = parser.parse_args()

    train(args)
