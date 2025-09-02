# from .env_config import EnvConfig
from .structures import Agent, State, CustomOccupancyGrid
import math
import numpy as np
from .utils_operations import normalizeAngle, Point
from .base import BaseEnvironment

class GridEnvironment(BaseEnvironment):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.goal_array = []
        self.generation_min_rays()

    def set_polygon_task(self, task, polygon_map):
        self.polygon_to_grid(polygon_map)
        task = self.task_to_grid(task)
        self.agent.set_task(task)
        self.agent.old_state = State(task["start"])
        self.agent.current_state = State(task["start"])
        self.agent.goal_state = State(task["goal"])

        return True

    def task_to_grid(self, task):
        new_task = {}
        new_task["start"] = task["start"]
        new_task["start"][0] /= self.occupancy_grid.resolution
        new_task["start"][1] /= self.occupancy_grid.resolution
        new_task["goal"] = task["goal"]
        new_task["goal"][0] /= self.occupancy_grid.resolution
        new_task["goal"][1] /= self.occupancy_grid.resolution

        return new_task

    def polygon_to_grid(self, polygon_map):
        self.occupancy_grid.resolution = 0.2
        height = int(40 / self.occupancy_grid.resolution)
        width = int(40 /self.occupancy_grid.resolution)
        self.occupancy_grid.set_height_width(height, width)
        self.occupancy_grid.map = np.zeros(self.occupancy_grid.width * self.occupancy_grid.height)
        for polygon in polygon_map:
            # print(f"polygon: {polygon}")
            x, y, theta, polygon_width, polygon_length = polygon
            x_min, y_min = x - polygon_length, y - polygon_width
            x_max, y_max = x + polygon_length, y + polygon_width
            # print(f"x_min, y_min: {x_min, y_min}")
            # print(f"x_max, y_max: {x_max, y_max}")
            x_min = int(x_min / self.occupancy_grid.resolution)
            y_min = int(y_min / self.occupancy_grid.resolution)
            x_max = int(x_max / self.occupancy_grid.resolution)
            y_max = int(y_max / self.occupancy_grid.resolution)
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    if i >= 0 and i <= self.occupancy_grid.width - 1 and j >= 0 and j <= self.occupancy_grid.height - 1:
                        key = j * self.occupancy_grid.width + i
                        self.occupancy_grid.map[key] = 80
    
    def get_observation(self, current_state, debug=False):
        # self.angle_space = np.linspace(-math.pi, math.pi, 39)
        x = int(current_state.x)
        y = int(current_state.y)
        theta = current_state.theta
        # print(f"debug: {debug}")
        # print(f"current_state.y: {current_state.y}")
        # occupancy_grid.resolution = 0.2
        max_check_radius = 10.
        radius = max_check_radius / self.occupancy_grid.resolution
        distance_array = []
        self.goal_array = []
        # print(f"get_observation")
        for orientation in self.angle_space:
            beam_orientation = normalizeAngle(orientation + theta)
            # print(f"get_bresenham_distance beam_orientation: {beam_orientation * 180 / math.pi}")
            distance, goal = self.get_bresenham_distance(radius, beam_orientation, x, y)
            # print(f"distance: {distance}")
            if True or debug:
                # print("--------- debug ---------")
                self.goal_array.append(goal)
                new_goal = []
                new_goal.append(x + distance * math.cos(beam_orientation))
                new_goal.append(y + distance * math.sin(beam_orientation))
                max_available_diff = 3.0
                # checking if the goal is right
                if (math.fabs(new_goal[0] - goal[0]) > max_available_diff or math.fabs(new_goal[1] - goal[1]) > max_available_diff):
                    print("------- Failed with goal -------")
                    print(f"bresenham goal: {goal}")
                    print(f"new_goal: {new_goal}")
                # distance_goal = math.hypot(x - goal[0], y - goal[1])
                # if (math.fabs(distance_goal - distance) > 0.1):
                #     print(f"distance_goal: {distance_goal}")
                #     print(f"calculated_distance: {distance}")
                # else:
                #     print("ok!!")
            distance_array.append(distance * self.occupancy_grid.resolution)

        if debug:
            # print(f"np.array(distance_array): {distance_array}")
            # print(f"goal_array: {self.goal_array}")
            return np.array(distance_array), self.goal_array
            # return np.array(distance_array), orientations + theta
        else:
            return np.array(distance_array)
    
    def get_bresenham_line(self, radius, angle, x0, y0, xg=None, yg=None):
        line = []
        init_x = x0
        init_y = y0
        if xg is None or yg is None:
            init_x2 = int(x0 + radius * math.cos(angle))
            init_y2 = int(y0 + radius * math.sin(angle))
        else:
            init_x2 = xg
            init_y2 = yg 
        x2 = int(init_x2)
        y2 = int(init_y2)
        dx = int(math.fabs(x2 - x0))
        dy = int(math.fabs(y2 - y0))
        pk = int(2 * dy - dx)
        decidion = 0

        if (dx <= dy):
            x0, y0 = y0, x0
            dx, dy = dy, dx
            x2, y2 = y2, x2
            init_x, init_y = init_y, init_x
            init_x2, init_y2 = init_y2, init_x2
            decidion = 1

        for i in range(int(dx)):
            if decidion:
                x_ = x0
                y_ = y0
            else:
                x_ = y0
                y_ = x0
            
            if (x_ < 0 or x_ >= self.occupancy_grid.width or y_ < 0  or y_ >= self.occupancy_grid.height):
                if (decidion):
                    return []
                else:
                    return []

                # print(f"decidion: {decidion}")
            line.append([y_, x_])

            # print(f"x0: {x0}")
            # print(f"x2: {x2}")
            if x0 < x2:
                x0 += 1
            else:
                x0 -= 1  

            if (pk < 0):
                pk = pk + 2 * dy
            else:
                if y0 < y2:
                    y0 += 1
                else:
                    y0 -= 1
                pk = pk + 2 * dy - 2 * dx

        if (decidion):
            line.append([init_y2, init_x2])
            return line
        else:
            line.append([init_x2, init_y2])
            return line

    # The code of bresenham doesn not depend of decicion? _x is y and y_ is x
    def get_bresenham_distance(self, radius, angle, x0, y0, xg=None, yg=None, check_value=60):
        init_x = x0
        init_y = y0
        if xg is None or yg is None:
            init_x2 = int(x0 + radius * math.cos(angle))
            init_y2 = int(y0 + radius * math.sin(angle))
        else:
            init_x2 = xg
            init_y2 = yg 
        x2 = int(init_x2)
        y2 = int(init_y2)
        dx = int(abs(x2 - x0))
        dy = int(abs(y2 - y0))
        pk = int(2 * dy - dx)
        decidion = 0
        check_value = check_value

        if (dx <= dy):
            x0, y0 = y0, x0
            dx, dy = dy, dx
            x2, y2 = y2, x2
            init_x, init_y = init_y, init_x
            init_x2, init_y2 = init_y2, init_x2
            decidion = 1

        for i in range(int(dx)):
            if decidion:
                x_ = x0
                y_ = y0
            else:
                x_ = y0
                y_ = x0
            
            if (x_ < 0 or x_ >= self.occupancy_grid.width or y_ < 0  or y_ >= self.occupancy_grid.height):
                if (decidion):
                    return math.hypot(init_x - x_, init_y - y_), [y_, x_]
                else:
                    return math.hypot(init_y - x_, init_x - y_), [y_, x_]

            if self.occupancy_grid.map[x_ * self.occupancy_grid.width + y_] >= check_value:
                # print(f"decidion: {decidion}")
                if (decidion):
                    return math.hypot(init_x - x_, init_y - y_), [y_, x_]
                else:
                    return math.hypot(init_y - x_, init_x - y_), [y_, x_]

            if x0 < x2:
                x0 += 1
            else:
                x0 -= 1  

            if (pk < 0):
                pk = pk + 2 * dy
            else:
                if y0 < y2:
                    y0 += 1
                else:
                    y0 -= 1
                pk = pk + 2 * dy - 2 * dx

        # print(f"final decidion: {decidion}")
        if (decidion):
            return math.hypot(init_x - init_x2, init_y - init_y2), [init_y2, init_x2]
        else:
            return math.hypot(init_x - init_x2, init_y - init_y2), [init_x2, init_y2]
        # return math.hypot(init_x - init_x2, init_y - init_y2), [init_x2, init_y2]
    
    def generation_min_rays(self):
        zero_state = State([0. for i in range(5)])
        self.dyn_obstacle_segments = []
        self.obstacle_segments = []
        shifted_state = self.agent.dynamic_model.shift_state(zero_state)
        width = (self.agent.dynamic_model.width) / (2 * self.occupancy_grid.resolution)\
                            + self.agent.dynamic_model.safe_eps
        length = (self.agent.dynamic_model.length) / (2 * self.occupancy_grid.resolution)\
                            + self.agent.dynamic_model.safe_eps
        self.obstacle_segments.append(self.getBB(shifted_state, width=width, length=length, ego=False))
        # distance_array = []
        # for angle in self.angle_space:
        #     beam_orientation = normalizeAngle(zero_state.theta + angle)
        #     beam = self.get_current_beam(zero_state, beam_orientation)
        #     distance_array.append(beam)
        self.clearance_is_enough = True
        self.agent.safe_vehicle_array = np.ones(self.n_beams)

    def is_robot_in_collision(self):
        center_state = self.agent.dynamic_model.shift_state(self.agent.current_state)
        segments = self.getBB(center_state)
        for segment in segments:
            v1, v2 = segment
            v1_x = int(v1.x)
            v1_y = int(v1.y)
            v2_x = int(v2.x)
            v2_y = int(v2.y)
            distance = math.hypot(v2_y - v1_y, v2_x - v1_x)
            angle = math.atan2(v2_y - v1_y, v2_x - v1_x)
            # the checking should be with the 95
            _, goal = self.get_bresenham_distance(distance, angle, v1_x, v1_y, v2_x, v2_y, check_value=95)
            distance_collision = math.hypot(goal[1] - v2_y, goal[0] - v2_x)
            # print(f"goal: {goal}")
            # print(f"actual goal: {[v2_x, v2_y]}")
            # print(f"distance_collision: {distance_collision}")
            if distance_collision > 0.1:
                return True
        return False
    
    def get_grid_robot(self):
        robot_contour = []
        center_state = self.agent.dynamic_model.shift_state(self.agent.current_state)
        print(f"self.occupancy_grid.resolution: {self.occupancy_grid.resolution}")
        segments = self.getBB(center_state)
        for id, segment in enumerate(segments):
            v1, v2 = segment
            v1_x = int(v1.x)
            v1_y = int(v1.y)
            v2_x = int(v2.x)
            v2_y = int(v2.y)
            distance = math.hypot(v2_y - v1_y, v2_x - v1_x)
            angle = math.atan2(v2_y - v1_y, v2_x - v1_x)
            # the checking should be with the 95
            # if id % 2 == 1:
            line = self.get_bresenham_line(distance, angle, v1_x, v1_y, v2_x, v2_y)
            robot_contour.extend(line)
        
        return robot_contour
    
    def get_goal_distance(self):
        # for every agent 
        return self.agent.get_goal_distance()
    
    def was_goal_reached(self):
        if self.soft_constraints:
            goalReached = self.agent.get_goal_distance() < self.HARD_EPS
        else:
            goalReached = self.agent.get_goal_distance() < self.HARD_EPS and self.agent.get_orientation_distance() < self.ANGLE_EPS
        return goalReached