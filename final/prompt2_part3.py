# import lxr as love

import numpy as np
import matplotlib.pyplot as plt
import copy

def input_map(map_number):
    if map_number == 1:
        _map = open('map1.txt', 'rt')
    elif map_number == 2:
        _map = open('map2.txt', 'rt')
    elif map_number == 3:
        _map = open('map3.txt', 'rt')
    byt = _map.readlines()
    obstacle = []
    agent_start = np.zeros(shape = (6,2), dtype=int)
    agent_goal = np.zeros(shape = (6,2), dtype=int)
    a = len(byt)
    b = len(byt[0])-1
    map_size = [a,b]
    for i in range(a):
        for j in range(b):
            if byt[i][j] == '#':
                obstacle.append([i,j])
            if byt[i][j] == 'A':
                agent_start[0][0] = i
                agent_start[0][1] = j
            if byt[i][j] == 'B':
                agent_start[1][0] = i
                agent_start[1][1] = j
            if byt[i][j] == 'C':
                agent_start[2][0] = i
                agent_start[2][1] = j
            if byt[i][j] == 'D':
                agent_start[3][0] = i
                agent_start[3][1] = j
            if byt[i][j] == 'E':
                agent_start[4][0] = i
                agent_start[4][1] = j
            if byt[i][j] == 'F':
                agent_start[5][0] = i
                agent_start[5][1] = j
            if byt[i][j] == '1':
                agent_goal[0][0] = i
                agent_goal[0][1] = j
            if byt[i][j] == '2':
                agent_goal[1][0] = i
                agent_goal[1][1] = j
            if byt[i][j] == '3':
                agent_goal[2][0] = i
                agent_goal[2][1] = j
            if byt[i][j] == '4':
                agent_goal[3][0] = i
                agent_goal[3][1] = j
            if byt[i][j] == '5':
                agent_goal[4][0] = i
                agent_goal[4][1] = j
            if byt[i][j] == '6':
                agent_goal[5][0] = i
                agent_goal[5][1] = j
    return map_size, obstacle, agent_start, agent_goal

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(map_size, obstacle, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (map_size[0]-1) or node_position[0] < 0 or node_position[1] > (map_size[1]-1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            is_obstacle = 0
            for i in range(len(obstacle)):
                if (int(obstacle[i][0]) == int(node_position[0])) and (int(obstacle[i][1]) == int(node_position[1])):
                    is_obstacle = 1
            if is_obstacle == 1:
                is_obstacle = 0
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def original_path(map_size, obstacle, agent_start, agent_goal):
    path = []
    path_no_start = []
    for i in range(len(agent_start)):
        start = (agent_start[i][0], agent_start[i][1])
        end = (agent_goal[i][0], agent_goal[i][1])
        path.append(astar(map_size, obstacle, start, end))
        path_no_start.append(astar(map_size, obstacle, start, end)[1:])
    return path, path_no_start

# multi astar algorithm
def collision_check(occupied_matrix, newnode):
    for i in range(occupied_matrix.shape[0]):
        if newnode[0] == occupied_matrix[i][0] and newnode[1] == occupied_matrix[i][1]:
            return i
    return None

def final_path(agent_start, agent_goal, original_path, original_path_no_start):
    occupied_matrix = copy.deepcopy(agent_start)
    occupied_matrix_old = copy.deepcopy(agent_start)
    _robot_position = copy.deepcopy(occupied_matrix)
    _robot_position = _robot_position.tolist()
    robot_position = []
    robot_position.append(_robot_position)
    _is_move_flag = np.zeros(shape = 6)
    while(original_path_no_start != [[],[],[],[],[],[]]):#需要修改
        for i in range(6):
            if original_path_no_start[i] != []:#需要修改
                # original_path[i].pop[0]
                # check collision before move
                index_1 = collision_check(occupied_matrix, original_path[i][1])
                index_2 = collision_check(occupied_matrix_old, original_path[i][1])
                if index_1 != None:
                    index = index_1
                elif index_2 != None:
                    index = index_2
                else:
                    index = None
                # print('collision_check:',collision_check(occupied_matrix, original_path[i][1]))
                # print('old:', collision_check(occupied_matrix_old, original_path[i][1]))
                # print('index:', index)
                if index != None:
                    # there is a collsion, check the collsion pattern then decide the move
                    if occupied_matrix[index][0] == agent_goal[index][0] and occupied_matrix[index][1] == agent_goal[index][1]:
                        collision_type = 1
                    elif original_path[index][1][0] == original_path[i][0][0] and original_path[index][1][1] == original_path[i][0][1]:
                        collision_type = 2
                    else:
                        collision_type = 3
                    if collision_type == 1 or collision_type == 2:
                        # re path planning
                        start = (original_path[i][0][0], original_path[i][0][1])
                        end = (agent_goal[i][0], agent_goal[i][1])
                        obstacle_new = copy.deepcopy(obstacle)
                        # print('original_path:',original_path)
                        obstacle_new.append([original_path[index][0][0], original_path[index][0][1]])
                        original_path[i] = astar(map_size, obstacle_new, start, end)
                        original_path_no_start[i] = astar(map_size, obstacle, start, end)[1:]
                        # # move
                        # occupied_matrix[i][0] = original_path[i][1][0]
                        # occupied_matrix[i][1] = original_path[i][1][1]
                        # original_path[i].pop(0)
                        # original_path_no_start[i].pop(0)
                else:
                    _is_move_flag[i] = 1
                    occupied_matrix[i][0] = original_path[i][1][0]
                    occupied_matrix[i][1] = original_path[i][1][1]
        occupied_matrix_old = copy.deepcopy(occupied_matrix)
        for i in range(6):
            if _is_move_flag[i] == 1:
                _is_move_flag[i] = 0
                # print('yes')
                # move
                # occupied_matrix[i][0] = original_path[i][1][0]
                # occupied_matrix[i][1] = original_path[i][1][1]
                # occupied_matrix[i][0] = original_path_no_start[i][0][0]
                # occupied_matrix[i][1] = original_path_no_start[i][0][1]
                original_path[i].pop(0)
                original_path_no_start[i].pop(0)
        new_occupied_matrix = copy.deepcopy(occupied_matrix)
        print('new_occupied_matrix:', new_occupied_matrix)
                # new_occupied_matrix.tolist()
        robot_position.append(new_occupied_matrix.tolist())
                # print('original_path_no_start:', original_path_no_start)
    return robot_position

def visualize_robotposition(robot_position, goal, map_size, obstacle):
    fig, ax = plt.subplots()
    for i in range(len(robot_position)):
        ax.cla()
        ax.scatter(np.array(goal)[:, 0], np.array(goal)[:, 1], color='r', marker='*')
        ax.scatter(np.array(robot_position)[i, :, 0], np.array(robot_position)[i, :, 1], marker='v')
        if obstacle != []:
            ax.scatter(np.array(obstacle)[:, 0], np.array(obstacle)[:, 1], color='g', marker='*')
        plt.axis([-1, map_size[0]+1, -1, map_size[1]+1])
        ax.legend()
        plt.pause(2)


if __name__ == '__main__':
    map_number = 2
    map_size, obstacle, agent_start, agent_goal = input_map(map_number)
    original_path,original_path_no_start = original_path(map_size, obstacle, agent_start, agent_goal)
    robot_position = final_path(agent_start, agent_goal, original_path, original_path_no_start) # only can be used for map 2
    for i in range(len(robot_position)):
        print(robot_position[i])
    visualize_robotposition(robot_position, agent_goal, map_size, obstacle)

    # [[0 6]
    # [2 6]
    # [0 7]
    # [2 8]
    # [1 6]
    # [1 7]]