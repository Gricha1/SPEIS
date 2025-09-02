import math
import numpy as np
from lib.envs import POLAMPEnvironment
from lib.utils_operations import generateDataSet
import json
import matplotlib.pyplot as plt
from lib.structures import State

# with open("configs/train_configs.json", 'r') as f:
#     train_config = json.load(f)

with open("configs/environment_configs.json", 'r') as f:
    our_env_config = json.load(f)
    # print(our_env_config)

with open("configs/reward_weight_configs.json", 'r') as f:
    reward_config = json.load(f)

with open("configs/car_configs.json", 'r') as f:
    car_config = json.load(f)

dataSet = generateDataSet(our_env_config)

maps, trainTask, valTasks = dataSet["obstacles"]
environment_config = {
        'vehicle_config': car_config,
        'tasks': trainTask,
        'valTasks': valTasks,
        'maps': maps,
        'our_env_config' : our_env_config,
        'reward_config' : reward_config,
        'evaluation': {},
    }

model_config = {
    "critic": [256, 256],
    "actor": [256, 256],
    "gamma": 0.99,
    "eps_clip": 0.2,
    "K_epochs": 10,
    "lr": 1e-4,
    "betas": [0.9, 0.999],
    "constrained_ppo": 1,
    "cost_limit": 0.2,
    "penalty_init": 1.0,
    "penalty_lr": 5e-4,
    "max_penalty": 1.0,
    "penalty_clip": 0.6,
}

env = POLAMPEnvironment("polamp", environment_config) 

obstacle = [ 
            [8, 9, 0 * math.pi/2, 5.5, 8],
            [28, 9, 0 * math.pi/2, 5.5, 8],
            [8, 25, 0 * math.pi/2, 5.5, 8],
            [28, 25, 0 * math.pi/2, 5.5, 8],
            ]
for obst in obstacle:
    state = State([obst[0], obst[1], obst[2], 0, 0])
    width = obst[3] 
    length = obst[4]
# plt.show()

def generateArc(pose1, pose2, pose3, radius):
    print(f"pose1.x: {pose1.x}")
    print(f"pose1.y: {pose1.y}")
    print(f"pose2.x: {pose2.x}")
    print(f"pose2.y: {pose2.y}")
    print(f"pose3.x: {pose3.x}")
    print(f"pose3.y: {pose3.y}")

    v1_x = pose2.x - pose1.x
    v1_y = pose2.y - pose1.y
    v2_x = pose2.x - pose3.x
    v2_y = pose2.y - pose3.y
    distance_pose2_pose1 = math.hypot(pose1.x - pose2.x, pose1.y - pose2.y)
    distance_pose2_pose3 = math.hypot(pose3.x - pose2.x, pose3.y - pose2.y)
    angle3 = math.acos((v1_x * v2_x + v1_y * v2_y) / (distance_pose2_pose1 * distance_pose2_pose3))
    if math.fabs(angle3 - math.pi / 2.) > 1e-4:
        return []

    angle1 = math.atan2(pose1.y - pose2.y, pose1.x - pose2.x)
    angle2 = math.atan2(pose3.y - pose2.y, pose3.x - pose2.x)
    length = radius / math.tan(angle3 / 2. + 1e-5)

    new_pose1 = State([0 for i in range(5)])
    new_pose1.x = pose2.x + length * math.cos(angle1)
    new_pose1.y = pose2.y + length * math.sin(angle1)
    
    new_pose2 = State([0 for i in range(5)])
    new_pose2.x = pose2.x + length * math.cos(angle2)
    new_pose2.y = pose2.y + length * math.sin(angle2)
    
    center = State([0 for i in range(5)])
    center.x = new_pose1.x + new_pose2.x - pose2.x
    center.y = new_pose1.y + new_pose2.y - pose2.y

    angle_space = np.linspace(0., angle3, 10)
    arc_trajectory = []
    for angle in angle_space:
        state = State([0 for i in range(5)])
        state.x = center.x + radius * math.cos(angle)
        state.y = center.y + radius * math.sin(angle)
        state.theta = angle - math.pi / 2.
        arc_trajectory.append(state)
    
    return arc_trajectory

def save_map(obstacle, name):
    with open("dataset/" + name, 'w') as output:
    # for obstacle in obstacles:
        output.write(str(len(obstacle)) + '\n')
        for polygon in obstacle:
            for element in polygon:
                output.write(str(element) + '\t')
            output.write("\n")

def save_dataset(lst_starts, lst_goals, name):
    with open("dataset/" + name, 'w') as output:
        output.write(str(len(lst_starts)) + '\n')
        for i in range(len(lst_starts)):
            # print(f"i : {i}")
            start = lst_starts[i]
            goal = lst_goals[i]
            for element in start:
                output.write(str(element) + '\t')
            for element in goal:
                output.write(str(element) + '\t')
            output.write("\n")

def visualization_map(obstacle, env):
    for obst in obstacle:
        state = State([obst[0], obst[1], obst[2], 0, 0])
        width = obst[3] 
        length = obst[4]
        segments = env.environment.getBB(state,  width=width, length=length, ego=False)
        env.drawObstacles(segments)

def visualization_dataset(lst_starts, lst_goals, env):
    for i in range(len(lst_starts)):
        start = lst_starts[i]
        goal = lst_goals[i]
        plt.plot(start[0], start[1], "H")
        plt.arrow(start[0], start[1], 3 * math.cos(start[2]), 3 * math.sin(start[2]), head_width=0.5, color='red', linewidth=4)
        plt.plot(goal[0], goal[1], "H")
        plt.arrow(goal[0], goal[1], 3 * math.cos(goal[2]), 3 * math.sin(goal[2]), head_width=0.5, color='green', linewidth=4)
        start = State([start[0], start[1], start[2], 0, 0])
        center_start = env.environment.agent.dynamic_model.shift_state(start)
        segments = env.environment.getBB(center_start, ego=True)
        env.drawObstacles(segments)
        goal = State([goal[0], goal[1], goal[2], 0, 0])
        center_goal = env.environment.agent.dynamic_model.shift_state(goal)
        segments = env.environment.getBB(center_goal, ego=True)
        env.drawObstacles(segments)
        if i >= 40 and i < 50:
            intersection_state = State([0 for i in range(5)])
            intersection_state.x = goal.x
            intersection_state.y = start.y
            trajectory = generateArc(start, intersection_state, goal, 2.5)
            for t in trajectory:
                state = env.environment.agent.dynamic_model.shift_state(t)
                segments = env.environment.getBB(state, ego=True)
                env.drawObstacles(segments)
                plt.plot(t.x, t.y, "b*")
    plt.show()

# number_of_tasks = 10
def task_generator(number_of_tasks=10):
    number_of_elem = 5
    lst_starts = []
    lst_goals = []
    for type in range(1, 8):
        if type == 1:
            x_min = 4
            x_max = 12
            y_min = 0.5
            y_max = 1.5
            theta = math.pi
        elif type == 2:
            x_min = 4
            x_max = 12
            y_min = 32.5
            y_max = 33.5
            theta = math.pi
        elif type == 3:
            x_min = 24
            x_max = 32
            y_min = 0.5
            y_max = 1.5
            theta = 0
        elif type == 4:
            x_min = 24
            x_max = 32
            y_min = 32.5
            y_max = 33.5
            theta = 0
        elif type == 5:
            x_min = 17.8
            x_max = 18.2
            y_min = 6
            y_max = 12
            theta = -math.pi / 2.
        elif type == 6:
            x_min = 17.8
            x_max = 18.2
            y_min = 22
            y_max = 28
            theta = math.pi / 2.
        elif type == 7:
            x_min = 24
            x_max = 32
            y_min = 16.5
            y_max = 17.5
            theta = 0
        
        start = [0 for _ in range(number_of_elem)]
        start[0] = np.random.uniform(4, 12)
        start[1] = 17
        for i in range(number_of_tasks):
            goal = [0 for _ in range(number_of_elem)]
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            goal[0] = x
            goal[1] = y
            goal[2] = theta
            lst_starts.append(start)
            lst_goals.append(goal)
        
    return lst_starts, lst_goals

lst_starts, lst_goals = task_generator()

print(" Saving dataset!!! ")
save_map(obstacle, "obstacle_map0.txt")
save_dataset(lst_starts, lst_goals, "train_map0.txt")
visualization_map(obstacle, env)
visualization_dataset(lst_starts, lst_goals, env)

lst_starts, lst_goals = task_generator(number_of_tasks=5)
save_dataset(lst_starts, lst_goals, "val_map0.txt")
visualization_map(obstacle, env)
visualization_dataset(lst_starts, lst_goals, env)
