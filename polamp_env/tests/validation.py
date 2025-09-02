from asyncio import tasks
import sys
import time
from collections import deque

import numpy as np
import torch
import wandb
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict


def enjoy(cfg, max_num_frames=250, use_wandb=False):
    if use_wandb:
        wandb.init( project='validate_polamp', entity='brian-angulo')
    cfg = load_from_checkpoint(cfg)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    # env.seed(0)

    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    # obs = env.reset()
    # val_key = env.valTasks
    # print(f"val_key: {val_key}")
    successed_tasks = 0
    collision_tasks = 0
    total_tasks = 0
    min_distance_array = []
    avg_distance_array = []
    trajectory_length = []
    # print(f"env.valTasks: {env.valTasks}")
    # print(f"env.maps: {env.maps}")
    start_time = time.time()
    for val_key in env.valTasks:
        eval_tasks = len(env.valTasks[val_key])
        total_tasks += eval_tasks
        print(f"val_key: {val_key}")
        for id in range(eval_tasks):
            obs = env.reset(id=id, val_key=val_key)
            print("##### Initializing #####")
            rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
            episode_reward = np.zeros(env.num_agents)
            finished_episode = [False] * env.num_agents
            print(f"max_num_frames: {max_num_frames}")
            done = [False]
            images = []
            num_frames = 0
            min_trajectory_distance = []
            with torch.no_grad():
                # or not max_frames_reached(num_frames)
                while not done[0]:
                    if max_frames_reached(num_frames):
                        break
                    # print(f"num_frames: {num_frames}")
                    # print(f"done: {done}")
                    obs_torch = AttrDict(transform_dict_observations(obs))
                    for key, x in obs_torch.items():
                        obs_torch[key] = torch.from_numpy(x).to(device).float()
                    # print(f"obs_torch: {obs_torch}")
                    # print(f"rnn_states: {rnn_states}")
                    policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

                    # sample actions from the distribution by default
                    actions = policy_outputs.actions
                    # print(f"actions: {actions}")

                    action_distribution = policy_outputs.action_distribution
                    # if isinstance(action_distribution, ContinuousActionDistribution):
                    #     if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                    actions = action_distribution.means
                    # print(f"actions: {actions}")

                    actions = actions.cpu().numpy()

                    rnn_states = policy_outputs.rnn_states

                    for _ in range(render_action_repeat):
                        if not cfg.no_render:
                            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                            current_delay = time.time() - last_render_start
                            time_wait = target_delay - current_delay

                            if time_wait > 0:
                                # log.info('Wait time %.3f', time_wait)
                                time.sleep(time_wait)

                            last_render_start = time.time()

                        obs, rew, done, infos = env.step(actions)
                        min_trajectory_distance.append(env.environment.min_beam)
                        # if use_wandb:
                        episode_reward += rew
                        num_frames += 1
                        # print(episode_reward)
                        # image = env.render(episode_reward[0], save_image=False)
                        # images.append(image)
                        # print(f"infos: {infos}")
                        # print(f"done: {done}")

                        for agent_i, done_flag in enumerate(done):
                            if done_flag:
                                finished_episode[agent_i] = True
                                episode_rewards[agent_i].append(episode_reward[agent_i])
                                true_rewards[agent_i].append(infos[agent_i].get('true_reward', episode_reward[agent_i]))
                                log.info('Episode finished for agent %d at %d frames. Reward: %.3f, true_reward: %.3f', agent_i, num_frames, episode_reward[agent_i], true_rewards[agent_i][-1])
                                rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)], dtype=torch.float32, device=device)
                                episode_reward[agent_i] = 0

                        # if episode terminated synchronously for all agents, pause a bit before starting a new one
                        # if all(done):
                        #     if not cfg.no_render:
                        #         image = env.render()
                        #         images.append(image)
                            # time.sleep(0.05)

                        if all(finished_episode):
                            print(f"finished_episode: {finished_episode}")
                            print(f"infos: {infos}")
                            if "Collision" in infos[0]:
                                # collision = True
                                # isDone = False
                                print("$$ Collision $$")
                                print(f"val_key: {val_key}")
                                print(f"id: {id}")
                                collision_tasks += 1
                            elif "SoftEps" in infos[0]:
                                print("$$ SoftEps $$")
                            else:
                                successed_tasks += 1
                            finished_episode = [False] * env.num_agents
                            avg_episode_rewards_str, avg_true_reward_str = '', ''
                            for agent_i in range(env.num_agents):
                                avg_rew = np.mean(episode_rewards[agent_i])
                                avg_true_rew = np.mean(true_rewards[agent_i])
                                if not np.isnan(avg_rew):
                                    if avg_episode_rewards_str:
                                        avg_episode_rewards_str += ', '
                                    avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                                if not np.isnan(avg_true_rew):
                                    if avg_true_reward_str:
                                        avg_true_reward_str += ', '
                                    avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'

                            log.info('Avg episode rewards: %s, true rewards: %s', avg_episode_rewards_str, avg_true_reward_str)
                            log.info('Avg episode reward: %.3f, avg true_reward: %.3f', np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]), np.mean([np.mean(true_rewards[i]) for i in range(env.num_agents)]))

                        # VizDoom multiplayer stuff
                        # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                        #     key = f'PLAYER{player}_FRAGCOUNT'
                        #     if key in infos[0]:
                        #         log.debug('Score for player %d: %r', player, infos[0][key])

            env.close()
            avg_distance_array.append(np.mean(min_trajectory_distance))
            min_distance_array.append(np.min(min_trajectory_distance))
            trajectory_length.append(len(min_trajectory_distance))
            if (len(images) > 0):
                images = np.transpose(np.array(images), axes=[0, 3, 1, 2])
                if use_wandb and (id == 5 or id ==7):
                    wandb.log({f"task_{key}_{id}": wandb.Video(images, fps=10, format="gif")})
            print("##### Ending #####")
    
    end_time = time.time()
    print("final time: ", end_time - start_time)
    success_rate = successed_tasks / total_tasks * 100
    collision_rate = collision_tasks / total_tasks * 100

    print(f"success_rate: {success_rate}")
    print(f"collision_rate: {collision_rate}")
    
    print(f"avg min distance: {np.mean(avg_distance_array)}")
    print(f"avg median distance: {np.median(avg_distance_array)}")
    print(f"min avg min distance: {np.min(min_distance_array)}")
    
    print(f"avg len trajectory: {np.mean(trajectory_length)}")
    print(f"median len trajectory: {np.median(trajectory_length)}")
    print(f"max len trajectory: {np.max(trajectory_length)}")
    print(f"min len trajectory: {np.min(trajectory_length)}")

    if use_wandb:
        # wandb.log({f"success rate on validation": wandb.Video(val_traj, fps=10, format="gif")})
        wandb.log(
                {
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                })

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


# import sys
# from .enjoy_validation import register_custom_components, custom_parse_args

# def main():
#     """Script entry point."""
#     register_custom_components()
#     cfg = custom_parse_args(evaluation=True)
#     # print(f"config {cfg}")
#     status = enjoy(cfg)
#     return status


# if __name__ == '__main__':
#     sys.exit(main())