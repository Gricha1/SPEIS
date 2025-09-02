import math
import numpy as np
from .utils_operations import normalizeAngle, degToRad, kmToM, Transformation

class State:
    def __init__(self, state):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]
        self.steer = state[4]
        self.width = 0
        self.length = 0

class Vehicle:
    def __init__(self, car_config):
        self.resolution = 1.0
        self.length = car_config["length"]
        self.width = car_config["width"]
        self.wheel_base = car_config["wheel_base"]
        self.safe_eps = car_config["safe_eps"]
        self.max_steer = degToRad(car_config["max_steer"])
        self.max_vel = kmToM(car_config["max_vel"])
        self.min_vel = kmToM(car_config["min_vel"])
        self.max_acc = car_config["max_acc"]
        self.max_ang_vel = degToRad(car_config["max_ang_vel"])
        self.delta_t = car_config["delta_t"]
        self.rear_to_center = (self.length - self.wheel_base) / 2.
        self.min_dist_to_check_collision = math.hypot(self.wheel_base / 2. + self.length / 2., self.width / 2.)

    def dynamic(self, state, action):
        acceleration = np.clip(action[0], -self.max_acc, self.max_acc)
        # print(f"acceleration: {acceleration}")
        # print(f"action[0]: {action[0]}")
        # acceleration = action[0]
        v_steering = np.clip(action[1], -self.max_ang_vel, self.max_ang_vel)
        # print(f"self.max_ang_vel: {self.max_ang_vel}")
        # print(f"action[1]: {action[1]}")
        # v_steering = action[1]
        overSpeeding = math.fabs(action[0]) > self.max_acc
        overSteering = math.fabs(action[1]) > self.max_ang_vel
        new_v = state.v + self.delta_t * acceleration
        new_phi = normalizeAngle(state.steer + v_steering * self.delta_t)
        # overSpeeding = new_v > self.max_vel or new_v < self.min_vel
        # overSteering = abs(new_phi) > self.max_steer
        new_v = np.clip(new_v, self.min_vel, self.max_vel)
        new_phi = np.clip(new_phi, -self.max_steer, self.max_steer)
        new_x = state.x + new_v * math.cos(state.theta + new_phi) * self.delta_t  / self.resolution
        new_y = state.y + new_v * math.sin(state.theta + new_phi) * self.delta_t  / self.resolution
        new_theta = normalizeAngle(state.theta + new_v / self.wheel_base * math.sin(new_phi) * self.delta_t)
        new_state = State([new_x, new_y, new_theta, new_v, new_phi])
        true_action = [acceleration, v_steering]
        # true_action = action

        return new_state, overSpeeding, overSteering, true_action
    
    def shift_state(self, state, toCenter=True):
        # Is needed to consider the resolution
        l = self.length / 2
        # shift to grid
        shift = (l - self.rear_to_center) / self.resolution 
        shift = -shift if toCenter else shift
        new_state = State([state.x + shift * math.cos(state.theta), state.y + shift * math.sin(state.theta), state.theta, state.v, state.steer])
        return new_state
    
class Agent():
    def __init__(self, agent_config):
        self.dynamic_model = Vehicle(agent_config)
        self.overSpeeding = False
        self.overSteering = False
        self.action = [0., 0.]
        self.resolution = 1.0
        empty_list = [0 for i in range(5)]
        self.old_state = State(empty_list)
        self.current_state = State(empty_list)
        self.goal_state = State(empty_list)
        self.safe_vehicle_array = []
    
    def move(self, action):
        # print("Move")
        self.current_state, self.overSpeeding, self.overSteering, true_action = self.dynamic_model.dynamic(self.current_state, action)
        self.action = true_action
        # print(f"true_action: {true_action}")

    def set_task(self, task):
        self.old_state = task["start"]
        self.current_state = task["start"]
        self.goal_state = task["goal"]
    
    def get_goal_distance(self):
        return math.hypot(self.current_state.x - self.goal_state.x, self.current_state.y - self.goal_state.y)
    
    def get_previous_goal_distance(self):
        return math.hypot(self.old_state.x - self.goal_state.x, self.old_state.y - self.goal_state.y)
    
    def get_orientation_distance(self):
        return normalizeAngle(self.current_state.theta - self.goal_state.theta)

    def getDiff(self):
        # if self.goal is None:
        #     self.goal = state
        self.transform = Transformation()
        current_transform, goal_transform = self.transform.rotate([self.current_state.x, self.current_state.y, self.current_state.theta], \
                                                                [self.goal_state.x, self.goal_state.y, self.goal_state.theta])

        delta = []
        # dx = (self.goal_state.x - self.current_state.x) * self.resolution
        # dy = (self.goal_state.y - self.current_state.y) * self.resolution
        # dtheta = self.goal_state.theta - self.current_state.theta
        dx = (goal_transform[0] - current_transform[0]) * self.resolution
        dy = (goal_transform[1] - current_transform[1]) * self.resolution
        dtheta = normalizeAngle(current_transform[2] - goal_transform[2])
        # dtheta = current_transform[2] - goal_transform[2]

        dv = self.goal_state.v - self.current_state.v
        dsteer = self.goal_state.steer - self.current_state.steer

        # theta = self.current_state.theta
        theta = current_transform[2]
        v = self.current_state.v
        steer = self.current_state.steer
        # dy always is 0 so we deleted it
        # delta.extend([dx, dtheta, dv, dsteer, theta, v, steer, self.action[0], self.action[1]])
        delta.extend([dx, dy, dtheta, dv, dsteer, theta, v, steer, self.action[0], self.action[1]])
        # print(f"delta: {delta}")
        return delta


class CustomOccupancyGrid():
    def __init__(self):
        self.map = []
        self.height = 600
        self.width = 600
        self.resolution = 1.0
    def set_height_width(self, height, width):
        self.height = height
        self.width = width
    
    def set_resolution(self, resolution):
        self.resolution = resolution
    
    def set_resolution(self, map):
        self.map = map

