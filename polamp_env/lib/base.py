# from .env_config import EnvConfig
from .structures import Agent, State, CustomOccupancyGrid
import math
import numpy as np
from .utils_operations import Point

class BaseEnvironment:
    def __init__(self, env_config):
        self.config = env_config
        self.env_config = env_config["our_env_config"]
        self.occupancy_grid = CustomOccupancyGrid()
        self.agent = Agent(self.config["vehicle_config"])
        self.reward_config = self.config["reward_config"]
        self.reward_weights = [
            self.reward_config["collision"],
            self.reward_config["goal"],
            self.reward_config["clearance"],
            self.reward_config["timeStep"],
            self.reward_config["distance"],
            self.reward_config["reverse"],
            self.reward_config["overSpeeding"],
            self.reward_config["overSteering"]
        ]
        self.n_beams = self.env_config['n_beams']
        self.angle_space = np.linspace(0, 2 * math.pi, self.n_beams + 1)[:-1]
        self.MAX_DIST_LIDAR = self.env_config['MAX_DIST_LIDAR']
        self.UPDATE_SPARSE = self.env_config['UPDATE_SPARSE']
        self.HARD_EPS = self.env_config['HARD_EPS']
        self.SOFT_EPS = self.env_config['SOFT_EPS']
        self.ANGLE_EPS = self.env_config['ANGLE_EPS'] * math.pi / 180
        self.soft_constraints = self.env_config['soft_constraints']
        self.goal_array = []
        self.clearance_is_enough = True

    def reset(self, resolution):
        self.agent.resolution = resolution
        self.agent.dynamic_model.resolution = resolution

    def set_polygon_task(self, task, polygon_map):
        raise NotImplementedError

    def set_task(self, task, map):
        self.occupancy_grid = map
        self.agent.set_task(task)
        self.agent.old_state = State(task["start"])
        self.agent.current_state = State(task["start"])
        self.agent.goal_state = State(task["goal"])

    def set_new_goal(self, new_goal):
        self.agent.goal_state = State(new_goal)

    def move_agent(self, action):
        self.agent.move(action)
    
    def get_observation(self, current_state, debug=False):
        raise NotImplementedError
    
    def is_robot_in_collision(self):
        raise NotImplementedError
    
    def getBB(self, state, width=2.0, length=3.8, ego=True):
        x = state.x
        y = state.y
        angle = state.theta
        if ego:
            w = self.agent.dynamic_model.width / (2 * self.occupancy_grid.resolution)
            l = self.agent.dynamic_model.length / (2 * self.occupancy_grid.resolution)
        else:
            w = width / self.occupancy_grid.resolution
            l = length / self.occupancy_grid.resolution
        BBPoints = [(-l, -w), (l, -w), (l, w), (-l, w)]
        vertices = []
        sinAngle = math.sin(angle)
        cosAngle = math.cos(angle)
        for i in range(len(BBPoints)):
            new_x = cosAngle * (BBPoints[i][0]) - sinAngle * (BBPoints[i][1])
            new_y = sinAngle * (BBPoints[i][0]) + cosAngle * (BBPoints[i][1])
            vertices.append(Point(new_x + x, new_y + y))
            
        segments = [(vertices[(i) % len(vertices)], \
                    vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
        
        return segments

    def is_robot_in_collision(self):
        raise NotImplementedError
    
    def get_goal_distance(self):
        # for every agent 
        return self.agent.get_goal_distance()
    
    def was_goal_reached(self):
        if self.soft_constraints:
            goalReached = self.agent.get_goal_distance() < self.HARD_EPS
        else:
            goalReached = self.agent.get_goal_distance() < self.HARD_EPS and self.agent.get_orientation_distance() < self.ANGLE_EPS
        return goalReached
    
    def reward(self, collision, goal_was_reached, step_counter):
        previous_delta = self.agent.get_previous_goal_distance()
        new_delta = self.agent.get_goal_distance()
        reward = []
        reward.append(-1 if collision else 0)

        if goal_was_reached:
            reward.append(1)
        else:
            reward.append(0)

        if self.clearance_is_enough:
            reward.append(0)
        else:
            reward.append(-1)

        if not ((step_counter + 1) % self.UPDATE_SPARSE):
            if (new_delta < 0.5):
                new_delta = 0.5
            reward.append(-1)
            # if self.with_potential:
            #     reward.append((previous_delta - new_delta) / new_delta)
            # else:
            # print(f"previous_delta - new_delta: {previous_delta - new_delta}")
            reward.append((previous_delta - new_delta) * self.occupancy_grid.resolution)
            reward.append(0 if self.agent.current_state.v >= 0 else -1)
            reward.append(-1 if self.agent.overSpeeding else 0)
            reward.append(-1 if self.agent.overSteering else 0)
            #Penalizing only for acceleration
            # reward.append(0)
            # Updating the current state
            self.agent.old_state = self.agent.current_state
        else:
            reward.append(0)
            reward.append(0)
            reward.append(0)
            reward.append(0)
            reward.append(0)

        # print(f"self.reward_weights: {self.reward_weights}")
        # print(f"reward: {reward}")
        return np.matmul(self.reward_weights, reward)
    
    def constrained_cost(self, step_counter):
        if not ((step_counter + 1) % self.UPDATE_SPARSE):
            if self.clearance_is_enough:
                return 0
            else:
                return self.agent.current_state.v
        else:
            return 0


