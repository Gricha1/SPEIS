import gym
import matplotlib.pyplot as plt
import math
import numpy as np
from .utils_operations import degToRad, kmToM
from .grid import GridEnvironment
from .polygon import PolygonEnvironment
from .structures import State
from gym import Wrapper


class POLAMPEnvironment(gym.Env):
    def __init__(self, full_env_name, config):
        # print(f"config {config}")
        # config = config["other_keys"]
        self.name = full_env_name
        env_config = config["our_env_config"]
        # self.evaluation = config["evaluation"]
        self.reward_config = config["reward_config"]
        self.goal = None
        self.current_state = None
        self.old_state = None
        # self.last_action = [0., 0.]
        # self.obstacle_segments = []
        # self.dyn_obstacle_segments = []
        self.last_observations = []
        self.hardGoalReached = False
        self.step_counter = 0
        # self.vehicle = VehicleConfig(config['vehicle_config'])
        self.trainTasks = config['tasks']
        self.valTasks = config['valTasks']
        self.maps_init = config['maps']
        self.maps = dict(config['maps'])
        self.alpha = env_config['alpha']
        self.max_steer = env_config['max_steer']
        self.max_dist = env_config['max_dist']
        self.min_dist = env_config['min_dist']
        self.min_vel = env_config['min_vel']
        self.max_vel = env_config['max_vel']
        self.min_obs_v = env_config['min_obs_v']
        self.max_obs_v = env_config['max_obs_v']
        self.HARD_EPS = env_config['HARD_EPS']
        self.SOFT_EPS = env_config['SOFT_EPS']
        self.ANGLE_EPS = degToRad(env_config['ANGLE_EPS'])
        self.SPEED_EPS = kmToM(env_config['SPEED_EPS'])
        self.STEERING_EPS = degToRad(env_config['STEERING_EPS'])
        self.MAX_DIST_LIDAR = env_config['MAX_DIST_LIDAR']
        self.UPDATE_SPARSE = env_config['UPDATE_SPARSE']
        self._max_episode_steps = env_config['max_polamp_steps']
        # self._max_episode_steps = 250
        # print(f"self._max_episode_steps {self._max_episode_steps}")
        # print(f"env_config {env_config}")
        # print(f"self._max_episode_steps {self.max_polamp_steps}")
        self.view_angle = degToRad(env_config['view_angle'])
        self.hard_constraints = env_config['hard_constraints']
        self.soft_constraints = env_config['soft_constraints']
        self.frame_stack = env_config['frame_stack']
        self.bias_beam = env_config['bias_beam']
        self.n_beams = env_config['n_beams']
        self.dynamic_obstacles = []
        # self.dyn_acc = 0
        # self.dyn_ang_vel = 0
        self.collision_time = 0
        # self.angle_space = np.linspace(-self.view_angle, self.view_angle, self.n_beams + 1)[:-1]
        self.reward_weights = [
            self.reward_config["collision"],
            self.reward_config["goal"],
            self.reward_config["timeStep"],
            self.reward_config["distance"],
            self.reward_config["reverse"],
            self.reward_config["overSpeeding"],
            self.reward_config["overSteering"]
        ]
        # self.environment = GridEnvironment(config)
        # print(f"env_config['is_polygon_env']: {env_config['is_polygon_env']}")
        self.is_polygon_env = env_config['is_polygon_env']
        if self.is_polygon_env:
            self.environment = PolygonEnvironment(config)
        else:
            self.environment = GridEnvironment(config)

        self.other_features = len(self.environment.agent.getDiff())
        state_min_box = [-np.inf for _ in range(self.other_features + self.n_beams)] * self.frame_stack
        state_max_box = [np.inf for _ in range(self.other_features + self.n_beams)] * self.frame_stack
        obs_min_box = np.array(state_min_box, dtype=np.float32)
        obs_max_box = np.array(state_max_box, dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_min_box, obs_max_box, dtype=np.float32)
        self.action_space = gym.spaces.Box(
                                        low=np.array([-self.environment.agent.dynamic_model.max_acc, -self.environment.agent.dynamic_model.max_ang_vel], dtype=np.float32),\
                                        high=np.array([self.environment.agent.dynamic_model.max_acc, self.environment.agent.dynamic_model.max_ang_vel], dtype=np.float32)
                                        )

        if (len(self.maps.keys()) > 0):
            self.lst_keys = list(self.maps.keys())
            index = np.random.randint(len(self.lst_keys))
            self.map_key = self.lst_keys[index]
            self.obstacle_map = self.maps[self.map_key]

    def reset(self, task=None, grid_map=None, id=None, val_key=None):
        # self.maps = dict(self.maps_init)
        self.hardGoalReached = False
        # if not grid_map is None:
            # print(f"grid_map.resolution: {grid_map.resolution}")
            # self.environment.reset(grid_map.resolution)

        self.step_counter = 0
        self.last_observations = []
        # self.last_action = [0., 0.]
        # self.obstacle_segments = []
        # self.dyn_obstacle_segments = []
        self.dynamic_obstacles = []
        # self.dyn_acc = 0
        # self.dyn_ang_vel = 0
        self.collision_time = 0
        # print(f"self.step_counter: {self.step_counter}")
        # set new tasks
        if id is None or val_key is None:
            if task is None or grid_map is None:
                self.map_key = self.lst_keys[np.random.randint(len(self.lst_keys))]
                polygon_map = self.maps[self.map_key]
                tasks = self.trainTasks[self.map_key]
                id = np.random.randint(len(tasks))
                current_task = {}
                current_task["start"] = tasks[id][0]
                current_task["goal"] = tasks[id][1]
                
                # Checking if the task is correct
                while not self.environment.set_polygon_task(current_task, polygon_map):
                    print("------one more time------")
                    self.map_key = self.lst_keys[np.random.randint(len(self.lst_keys))]
                    polygon_map = self.maps[self.map_key]
                    tasks = self.trainTasks[self.map_key]
                    id = np.random.randint(len(tasks))
                    current_task = {}
                    current_task["start"] = tasks[id][0]
                    current_task["goal"] = tasks[id][1]
                # print(f"polygon_map: {polygon_map}")
                # print(f"current_task: {current_task}")
            else:
                self.environment.set_task(task, grid_map)
        else:
            print("---------- Validating ----------")
            self.map_key = val_key
            polygon_map = self.maps[self.map_key]
            tasks = self.valTasks[self.map_key]
            current_task = {}
            current_task["start"] = tasks[id][0]
            current_task["goal"] = tasks[id][1]
            self.environment.set_polygon_task(current_task, polygon_map)

        
        self.environment.reset(self.environment.occupancy_grid.resolution)
        # We need to include debug for inference?
        self.environment.agent.action = [0., 0.]
        beams_observation = self.environment.get_observation(self.environment.agent.current_state)
        extra_observation = self.environment.agent.getDiff()
        
        if len(self.last_observations) == 0:
            for _ in range(self.frame_stack - 1):
                self.last_observations.extend(beams_observation)
                self.last_observations.extend(extra_observation)
        observation = list(self.last_observations)
        observation.extend(beams_observation)
        observation.extend(extra_observation)

        return np.array(observation, dtype=np.float32)
    
    def set_new_goal(self, new_goal=None):
        self.environment.set_new_goal(new_goal)

    def step(self, action, next_dyn_states=[]):
        info = {}
        isDone = False
        self.environment.move_agent(action)
        self.beams_observation = self.environment.get_observation(self.environment.agent.current_state)
        extra_observation = self.environment.agent.getDiff()

        # frame stack!!
        self.last_observations.extend(self.beams_observation)
        # print(f"beams_observation: {beams_observation}")
        self.last_observations.extend(extra_observation)
        # print(f"extra_observation: {extra_observation}")
        observation = self.last_observations
        self.last_observations = self.last_observations[self.other_features + self.n_beams:]

        collision = self.environment.is_robot_in_collision()
        distance_to_goal = self.environment.get_goal_distance()

        info["EuclideanDistance"] = distance_to_goal * self.environment.agent.resolution

        goal_was_reached = self.environment.was_goal_reached()

        if not self.hardGoalReached and distance_to_goal < self.SOFT_EPS:
            self.hardGoalReached = True
        if self.hardGoalReached:
            if distance_to_goal > self.SOFT_EPS:
                info["SoftEps"] = False

        reward = self.environment.reward(collision, goal_was_reached, self.step_counter)
        info["cost"] = self.environment.constrained_cost(self.step_counter)
        # print(f"cost: {info['cost']}")
        self.step_counter += 1
        
        if goal_was_reached or collision or (self._max_episode_steps == self.step_counter):
            isDone = True
            if collision:
                info["Collision"] = True
            if (self._max_episode_steps == self.step_counter):
                info["SoftEps"] = False
        
        return np.array(observation, dtype=np.float32), reward, isDone, info
    
    def drawObstacles(self, vertices, color="-b"):
        a = vertices
        plt.plot([a[(i + 1) % len(a)][0].x for i in range(len(a) + 1)], [a[(i + 1) % len(a)][0].y for i in range(len(a) + 1)], color, linewidth=2)

    def render(self, mode="human", save_image=True):
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(4)
        resolution = self.environment.occupancy_grid.resolution
        # self.MAX_DIST_LIDAR = 20
        x_delta = self.MAX_DIST_LIDAR / resolution
        y_delta = self.MAX_DIST_LIDAR / resolution
        # print(f"x_delta: {x_delta}")

        x_min = self.environment.agent.current_state.x - x_delta
        x_max = self.environment.agent.current_state.x + x_delta
        ax.set_xlim(x_min, x_max)

        y_min = self.environment.agent.current_state.y - y_delta
        y_max = self.environment.agent.current_state.y + y_delta
        ax.set_ylim(y_min, y_max)

        current_state = self.environment.agent.current_state
        center_state = self.environment.agent.dynamic_model.shift_state(current_state)
        agentBB = self.environment.getBB(center_state, ego=True)
        self.drawObstacles(agentBB)
        ax.arrow(current_state.x, current_state.y, 3 * math.cos(current_state.theta), 3 * math.sin(current_state.theta), head_width=0.5,
                 color='red', linewidth=4)

        center_state = self.environment.agent.dynamic_model.shift_state(self.environment.agent.goal_state)
        agentBB = self.environment.getBB(center_state, ego=True)
        self.drawObstacles(agentBB, "-g")
        ax.arrow(self.environment.agent.goal_state.x, self.environment.agent.goal_state.y, 3 * math.cos(self.environment.agent.goal_state.theta), 3 * math.sin(self.environment.agent.goal_state.theta), head_width=0.5,
                 color='green', linewidth=4)

        ax.plot([current_state.x, self.environment.agent.goal_state.x], [current_state.y, self.environment.agent.goal_state.y], '--r')
        
        tuple_rays = self.environment.get_observation(current_state, debug=True)
        
        counter = 0
        for distance, goal in zip(tuple_rays[0], tuple_rays[1]):
            if counter == 0:
                ax.plot([current_state.x, goal[0]], [current_state.y, goal[1]], '-', linewidth = 4, color='green')
                counter += 1
                continue
            ax.plot([current_state.x, goal[0]], [current_state.y, goal[1]], '-', linewidth = 2, color='orange')
        
        counter = 0
        for distance, angle in zip(self.environment.agent.safe_vehicle_array, self.environment.angle_space):
            if counter == 0:
                ax.plot([current_state.x, current_state.x + distance * math.cos(angle + current_state.theta)],\
                        [current_state.y, current_state.y + distance * math.sin(angle + current_state.theta)],\
                        '-', linewidth = 4, color='red')
                counter += 1
                continue
            ax.plot([current_state.x, current_state.x + distance * math.cos(angle + current_state.theta)],\
                    [current_state.y, current_state.y + distance * math.sin(angle + current_state.theta)],\
                    '-', linewidth = 2, color='cyan')

        if self.is_polygon_env:
            for obstacle_segment in self.environment.obstacle_segments:
                self.drawObstacles(obstacle_segment, "-b")
        else:
            print(f"self.environment.occupancy_grid.resolution: {self.environment.occupancy_grid.resolution}")
            # print(f"self.environment.occupancy_grid.width: {self.environment.occupancy_grid.width}")
            # print(f"self.environment.occupancy_grid.height: {self.environment.occupancy_grid.height}")
            for id, cell in enumerate(self.environment.occupancy_grid.map):
                if cell >= 80:
                    # print(f"cell: {cell}")
                    x = id % self.environment.occupancy_grid.width
                    y = id // self.environment.occupancy_grid.height
                    if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
                        ax.plot(x, y, ".k")
            
            contour = self.environment.get_grid_robot()
            for id, point in enumerate(contour):
                ax.plot(point[0], point[1], "Hg")
        
        action = self.environment.agent.action
        linear_velocity = current_state = self.environment.agent.current_state.v
        steering_angle = current_state = self.environment.agent.current_state.steer * 180 / math.pi
        # dx = self.goal.x - self.current_state.x
        # dy = self.goal.y - self.current_state.y
        # theta = radToDeg(self.current_state.theta)
        # v = mToKm(self.current_state.v)
        # delta = radToDeg(self.current_state.steer)
        
        ax.set_title('lin-acc: {:.2f}, ang-vel: {:.2f}, lin-vel: {:.2f}, steer: {:.1f}'.format(action[0], action[1], linear_velocity, steering_angle))
          
        if save_image:
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return image
        else:
            # plt.pause(0.1)
            plt.show()
            # plt.pause(0.1)
            # plt.close('all')
            # plt.close('all')

    def close(self):
        pass


class MultiPolampAgentWrapper(Wrapper):
    """
    This wrapper allows us to treat a single-agent environment as multi-agent with 1 agent.
    That is, the data (obs, rewards, etc.) is converted into lists of length 1

    """

    def __init__(self, env):
        super().__init__(env)

        self.num_agents = 1
        self.is_multiagent = True

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return [obs]

    def step(self, action):
        action = action[0]
        obs, rew, done, info = self.env.step(action)
        # if done:
        #     obs = self.env.reset()
        return [obs], [rew], [done], [info]