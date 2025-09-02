from .structures import Agent, State, CustomOccupancyGrid
import math
import numpy as np
from .utils_operations import normalizeAngle, angleIntersection, doIntersect, Point, Line, intersectPolygons
from .base import BaseEnvironment

class PolygonEnvironment(BaseEnvironment):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.generation_min_rays()

    def set_polygon_task(self, task, polygon_map):
        # self.polygon_to_grid(polygon_map)
        task = self.task_to_grid(task)
        self.agent.set_task(task)
        self.agent.old_state = State(task["start"])
        self.agent.current_state = State(task["start"])
        self.agent.goal_state = State(task["goal"])
        
        # set the polygon map
        self.dyn_obstacle_segments = []
        self.obstacle_segments = []
        self.lst_indexes = []
        growing_obstacles = np.random.randint(0, 11)
        for id, obstacle in enumerate(polygon_map):
            obs = State([obstacle[0], obstacle[1], obstacle[2], 0, 0])
            width = obstacle[3] 
            length = obstacle[4]
            # if growing_obstacles > 3:
            #     width += np.random.randint(1, 6) / 2.
            #     length += np.random.randint(1, 6) / 2.
            self.obstacle_segments.append(self.getBB(obs, width=width, length=length, ego=False))
            self.lst_indexes.append(id)
        self.min_beam = 0
        task_is_correct = not self.is_robot_in_collision()
        # print(f"task_is_correct: {task_is_correct}")

        return task_is_correct

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
        self.occupancy_grid.set_height_width(40, 40)
        # self.height = int(40 / self.occupancy_grid.resolution)
        # self.width = int(40 /self.occupancy_grid.resolution)
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
                        self.occupancy_grid[key] = 80
    
    def get_relevant_segment(self, state, with_angles=False):
        relevant_obstacles = []
        obstacles = list(self.obstacle_segments)
        obstacles.extend(self.dyn_obstacle_segments)
        for obst in obstacles:
            new_segments = []
            for segment in obst:
                d1 = math.hypot(state.x - segment[0].x, state.y - segment[0].y)
                d2 = math.hypot(state.x - segment[1].x, state.y - segment[1].y)
                new_segments.append((min(d1, d2), segment)) 
            new_segments.sort(key=lambda s: s[0])
            new_segments = [pair[1] for pair in new_segments[:2]]
            if not with_angles:
                relevant_obstacles.append(new_segments)
            else:
                new_segments_with_angle = []
                angles = []
                for segment in new_segments:
                    angle1 = math.atan2(segment[0].y - state.y, segment[0].x - state.x)
                    angle2 = math.atan2(segment[1].y - state.y, segment[1].x - state.x)
                    min_angle = min(angle1, angle2)
                    max_angle = max(angle1, angle2)
                    new_segments_with_angle.append(((min_angle, max_angle), segment))
                    angles.append((min_angle, max_angle))
                if angleIntersection(angles[0][0], angles[0][1], angles[1][0]) and \
                    angleIntersection(angles[0][0], angles[0][1], angles[1][1]):
                    relevant_obstacles.append([new_segments_with_angle[0]])
                elif angleIntersection(angles[1][0], angles[1][1], angles[0][0]) and \
                    angleIntersection(angles[1][0], angles[1][1], angles[0][1]):
                    relevant_obstacles.append([new_segments_with_angle[1]])
                else:
                    relevant_obstacles.append(new_segments_with_angle)
                    
        return relevant_obstacles

    def get_current_beam(self, state, angle, nearestObstacles=None, with_angles=False, lst_indexes=[]):
        if nearestObstacles is None:
            nearestObstacles = list(self.obstacle_segments)
            nearestObstacles.extend(self.dyn_obstacle_segments)
        
        # angle = normalizeAngle(angle + state.theta)
        new_x = state.x + self.MAX_DIST_LIDAR * math.cos(angle)
        new_y = state.y + self.MAX_DIST_LIDAR * math.sin(angle)
        p1 = Point(state.x, state.y)
        q1 = Point(new_x, new_y)
        min_dist = self.MAX_DIST_LIDAR
        for i, obstacles in enumerate(nearestObstacles):
            for obst_with_angles in obstacles:
                if with_angles:
                    angle1, angle2 = obst_with_angles[0]
                    p2, q2 = obst_with_angles[1]
                    if not angleIntersection(angle1, angle2, angle):
                        continue
                else:
                    p2, q2 = obst_with_angles

                if(doIntersect(p1, q1, p2, q2)):
                    beam = Line(p1, q1)
                    segment = Line(p2, q2)
                    intersection = beam.isIntersect(segment)
                    distance = math.hypot(p1.x - intersection.x, p1.y - intersection.y)
                    min_dist = min(min_dist, distance)
                    if (distance < self.agent.dynamic_model.min_dist_to_check_collision):
                        if i not in lst_indexes and i < len(self.obstacle_segments):
                            lst_indexes.append(i)
                    
        return min_dist

    def get_observation(self, current_state, debug=False):
        distance_array = []
        lst_indexes = []
        # self.MAX_DIST_LIDAR = 10
        self.bias_beam = 0
        goal_array = []
        if len(self.obstacle_segments) > 0 or len(self.dyn_obstacle_segments) > 0:
            with_angles=True
            nearest_obstacles = self.get_relevant_segment(current_state, with_angles=with_angles)
            for angle in self.angle_space:
                beam_orientation = normalizeAngle(current_state.theta + angle)
                beam = self.get_current_beam(current_state, beam_orientation, nearest_obstacles, with_angles=with_angles, lst_indexes=lst_indexes)
                distance_array.append(beam - self.bias_beam)
                new_goal = []
                new_goal.append(current_state.x + beam * math.cos(beam_orientation))
                new_goal.append(current_state.y + beam * math.sin(beam_orientation))
                goal_array.append(new_goal)
        else:
            for angle in self.angle_space:
                distance_array.append(self.MAX_DIST_LIDAR - self.bias_beam)

        # this information is for collision checking
        self.lst_indexes = lst_indexes
        self.min_beam = np.min(distance_array)
        np_distance_array = np.array(distance_array)
        self.clearance_is_enough = (np_distance_array > self.agent.safe_vehicle_array).all()
        
        # print(f"np_distance_array: {np_distance_array}")
        # print(f"self.agent.safe_vehicle_array: {self.agent.safe_vehicle_array}")
        # print(f"np_distance_array > self.agent.safe_vehicle_array: {np_distance_array > self.agent.safe_vehicle_array}")
        # print(f"self.clearance_is_enough: {self.clearance_is_enough}")

        if debug:
            return np_distance_array, goal_array
        else:
            return np_distance_array    
        
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
        distance_array = []
        for angle in self.angle_space:
            beam_orientation = normalizeAngle(zero_state.theta + angle)
            beam = self.get_current_beam(zero_state, beam_orientation)
            distance_array.append(beam)
        
        self.agent.safe_vehicle_array = np.array(distance_array)

    def is_robot_in_collision(self):
        center_state = self.agent.dynamic_model.shift_state(self.agent.current_state)
        
        if (self.agent.dynamic_model.min_dist_to_check_collision < self.min_beam):
            return False

        if len(self.obstacle_segments) > 0 or len(self.dyn_obstacle_segments) > 0:
            bounding_box = self.getBB(center_state)
            for i, obstacle in enumerate(self.obstacle_segments):
                if i in self.lst_indexes:
                    if (intersectPolygons(obstacle, bounding_box)):
                        return True
                    
            for obstacle in self.dyn_obstacle_segments:
                mid_x = (obstacle[0][0].x + obstacle[1][1].x) / 2.
                mid_y = (obstacle[0][0].y + obstacle[1][1].y) / 2.
                distance = math.hypot(mid_x - center_state.x, mid_y - center_state.y)
                if (distance > (self.agent.dynamic_model.min_dist_to_check_collision * 2)):
                    continue
                if (intersectPolygons(obstacle, bounding_box)):
                    return True
            
        return False
    
    def get_grid_robot(self):
        robot_contour = []
        center_state = self.agent.dynamic_model.shift_state(self.agent.current_state)
        segments = self.getBB(center_state)
        for id, segment in enumerate(segments):
            v1, v2 = segment
            v1_x = int(v1[0])
            v1_y = int(v1[1])
            v2_x = int(v2[0])
            v2_y = int(v2[1])
            distance = math.hypot(v2_y - v1_y, v2_x - v1_x)
            angle = math.atan2(v2_y - v1_y, v2_x - v1_x)
            # the checking should be with the 95
            # if id % 2 == 1:
            line = self.get_bresenham_line(distance, angle, v1_x, v1_y, v2_x, v2_y)
            robot_contour.extend(line)
        
        return robot_contour
        # x = int(self.agent.current_state.x)
        # y = int(self.agent.current_state.y)
        # hard_buffer = 80
        # if self.occupancy_grid[x + self.width * y] >= hard_buffer:
        #     return True
        # return False
        # raise NotImplementedError