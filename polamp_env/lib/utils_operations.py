from math import pi, fmod
import math
import numpy as np

ONE_RAD_GRAD = pi / 180
ONE_GRAD_RAD = 180. / pi

def normalizeAngle(angle, symmetric=True):
    begin = pi if symmetric else 0
    norm_angle = fmod(angle + begin, 2*pi)
    if norm_angle < 0:
        norm_angle += 2*pi
    return norm_angle - begin

def degToRad(deg):
    return deg * ONE_RAD_GRAD


def radToDeg(rad):
    return rad * ONE_GRAD_RAD


def mToKm(vel):
    return vel * 3.6


def kmToM(vel):
    return vel / 3.6

def angleIntersection(angle1, angle2, angle):
    if angle1 == angle or angle2 == angle:
        return True

    if angle1 * angle2 > 0:
        if (angle <= angle1 or angle >= angle2):
            return False
    else:
        dist1 = angle2 - angle1
        # dist2 = math.pi - angle2 + angle1 + math.pi
        if dist1 > math.pi:
            if (angle < 0 and angle >= angle1):
                return False
            if (angle > 0 and angle <= angle2):
                return False
        else:
            if (angle <= angle1 or angle >= angle2):
                return False
    return True

class Transformation():
    def __init__(self):     
        self.diff_x = 0
        self.diff_y = 0
        self.theta = 0
        self.cos_theta = 0
        self.sin_theta = 0

    def rotate(self, start, goal):
        x_start = start[0]
        y_start = start[1]
        theta_start = start[2]
        x_goal = goal[0]
        y_goal = goal[1]
        theta_goal = goal[2]
        self.theta = math.atan2(y_goal - y_start, x_goal - x_start)
        self.diff_x = x_start
        self.diff_y = y_start
        x_start = 0
        y_start = 0
        x_goal -= self.diff_x
        y_goal -= self.diff_y
        self.cos_theta = math.cos(self.theta)
        self.sin_theta = math.sin(self.theta)
        new_x_goal = self.cos_theta * x_goal + self.sin_theta * y_goal
        new_y_goal = - self.sin_theta * x_goal + self.cos_theta * y_goal
        theta_start -= self.theta
        theta_goal -= self.theta

        return [x_start, y_start, normalizeAngle(theta_start)], [new_x_goal, new_y_goal, normalizeAngle(theta_goal)]
    
    def rotateState(self, state):
        sx = state[0]
        sy = state[1]
        stheta = state[2]
        sx -= self.diff_x
        sy -= self.diff_y
        stheta -= self.theta
        new_sx = self.cos_theta * sx + self.sin_theta * sy
        new_sy = - self.sin_theta * sx + self.cos_theta * sy
        
        return [new_sx, new_sy, normalizeAngle(stheta)]

    def inverseRotate(self, state):
        x = state[0]
        y = state[1]
        theta = state[2]
        new_x = self.cos_theta * x - self.sin_theta * y
        new_y = self.sin_theta * x + self.cos_theta * y
        new_x += self.diff_x
        new_y += self.diff_y
        theta += self.theta
        return new_x, new_y, normalizeAngle(theta)


def generateValidateTasks(config):
    min_dist = config['min_dist']
    max_dist = config['max_dist']
    max_val_vel = config['max_vel']
    alpha = config['alpha']
    discrete_alpha = config['discrete_alpha']
    max_steer = config['max_steer']
    alpha = degToRad(alpha)
    alphas = np.linspace(-alpha, alpha, discrete_alpha)
    valTasks = []

    for angle in alphas:
        for angle1 in alphas:
            valTasks.append(([0., 0., angle, 0., degToRad(np.random.randint(-max_steer, max_steer + 1))], [np.random.randint(min_dist, max_dist + 1), 0., angle1, kmToM(np.random.randint(0, max_val_vel + 1)), 0.]))


def readObstacleMap(file):
    obstacles = []
    with open(file, "r") as f:
        j = -1
        for line in f.readlines():
            if(j == -1):
                j += 1
                continue
            parameters = line.split('\t')
            obst = []
            for i in range(len(parameters) - 1):
                obst.append(float(parameters[i]))

            obstacles.append(obst)
        return obstacles

def readTasks(file):
    tasks = []
    with open(file, "r") as f:
        j = -1
        for line in f.readlines():
            if(j == -1):
                j += 1
                continue
            parameters = line.split('\t')
            # print(parameters)
            start = []
            goal = []
            for i in range(len(parameters) - 1):
                # print(parameters[i])
                if i > 4:
                    goal.append(float(parameters[i]))
                else:
                    start.append(float(parameters[i]))
            tasks.append((start, goal))
        #     plt.plot([start[0], goal[0]], [start[1], goal[1]], '-r')
        
        # plt.show()
        return tasks

def readDynamicTasks(file):
    tasks = []
    with open(file, "r") as f:
        j = -1
        for line in f.readlines():
            if(j == -1):
                j += 1
                continue
            parameters = line.split('\t')
            # print(parameters)
            start = []
            goal = []
            dynamic_obst = []
            for i in range(len(parameters) - 1):
                # print(parameters[i])
                if i < 5:
                    start.append(float(parameters[i]))
                elif i < 10:
                    goal.append(float(parameters[i]))
                else:
                    dynamic_obst.append(float(parameters[i]))
                # print(f"start {start}")
                # print(f"goal {goal}")
                # print(f"dynamic_obst {dynamic_obst}")
            tasks.append((start, goal, [dynamic_obst]))
        #     plt.plot([start[0], goal[0]], [start[1], goal[1]], '-r')
        
        # plt.show()
        return tasks


def generateDataSet(our_env_config, name_folder="maps", total_maps=12, dynamic=True):
    dataSet = {}
    valTasks = generateValidateTasks(our_env_config)
    maps = {"map1": []}
    trainTask = {"map1": []}
    valTasks = {"map1": valTasks}
    dataSet["empty"] = (maps, trainTask, valTasks)
    
    maps_obst = {}
    valTasks_obst = {}
    trainTask_obst = {}
    for index in range(total_maps):
        maps_obst["map" + str(index)] = readObstacleMap(name_folder + "/obstacle_map" + str(index)+ ".txt")
        valTasks_obst["map" + str(index)] = readTasks(name_folder + "/val_map" + str(index)+ ".txt")
        trainTask_obst["map" + str(index)] = readTasks(name_folder + "/train_map" + str(index)+ ".txt")
    dataSet["obstacles"] = (maps_obst, trainTask_obst, valTasks_obst)
    
    if dynamic:
        maps_dyn_obst = {}
        valTasks_dyn_obst = {}
        trainTask_dyn_obst = {}
        for index in range(total_maps):
            maps_dyn_obst["map" + str(index)] = readObstacleMap(name_folder + "/obstacle_map" + str(index)+ ".txt")
            valTasks_dyn_obst["map" + str(index)] = readDynamicTasks(name_folder + "/dyn_val_map" + str(index)+ ".txt")
            trainTask_dyn_obst["map" + str(index)] = readDynamicTasks(name_folder + "/dyn_train_map" + str(index)+ ".txt")
        dataSet["dyn_obstacles"] = (maps_dyn_obst, trainTask_dyn_obst, valTasks_dyn_obst)

    # print(f"dataSet {dataSet}")
    return dataSet

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False
 
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
     
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.
     
    val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y))
    if (val > 0): 
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0
 
# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
     
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
 
    # Special Cases
 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
 
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
 
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
 
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
 
    # If none of the cases
    return False

class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        delta_x = end.x - start.x
        delta_y = end.y - start.y
        
        if delta_x == 0:
            b = 0
            a = 1
        elif delta_y == 0:
            a = 0
            b = 1
        else:
            b = 1
            a = -delta_y / delta_x
        c = - a * start.x - b * start.y
        
        self.a = a
        self.b = b
        self.c = c
        
    def isIntersect(self, other):
        det = self.a * other.b - self.b * other.a
        if det == 0:
            #exception
            return Point(other.start.x, other.start.y)
        
        det1 = -self.c * other.b + self.b * other.c
        det2 = -self.a * other.c + self.c * other.a
        x = det1 / det
        y = det2 / det

        return Point(x, y)


def separatingAxes(a, axes):
    for i in range(len(a)):
        current = a[i]
        next = a[(i + 1) % len(a)]
        edge = np.array(next) - np.array(current) 
        new_edge = edge / (np.sqrt(np.sum(edge ** 2)) + 1e-6)
        # print(f"new_edge : {new_edge}")
        axes.append([-new_edge[1], new_edge[0]])

def project(a, axis):
    maxProj = -math.inf
    minProj = math.inf
    for v in a:
        proj = np.dot(axis, v)
        if proj < minProj:
            minProj = proj
        if proj > maxProj:
            maxProj = proj
    
    return minProj, maxProj

def intersectPolygons(a, b, rl=True):
    axes = []
    new_a = []
    if rl:
        for pair in a:
            # print(pair)
            new_a.append((pair[0].x, pair[0].y))
        a = new_a
        new_b = []
        for pair in b:
            new_b.append((pair[0].x, pair[0].y))
        b = new_b 
    separatingAxes(a, axes)
    separatingAxes(b, axes)
    for axis in axes:
        aMinProj, aMaxProj, bMinProj, bMaxProj = 0., 0., 0., 0.
        aMinProj, aMaxProj = project(a, axis)
        bMinProj, bMaxProj = project(b, axis)
        if (aMinProj > bMaxProj) or (bMinProj > aMaxProj):
            return False 
    return True