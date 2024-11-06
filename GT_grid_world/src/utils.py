import numpy as np
from typing import Union

from .graph import Graph

def get_assigned_task_id(Ra : list, robot_id : int):
    for assignment in Ra:
        if assignment[-1] == robot_id:
            return assignment[0]

def get_robot_state(Rs : list, robot_id : int) -> tuple:
    for robot in Rs:
        if robot[0] == robot_id:
            return robot[1]
        
def get_task_start_location(J : set, task_id : int) -> tuple:
    for task in J:
        if task[0] == task_id:
            return task[1]
        
def get_task_goal_location(J : set, task_id : int) -> tuple:
    for task in J:
        if task[0] == task_id:
            return task[2]
        
def euclidian_distance(p1 : tuple, p2 : tuple):
    return ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5

def manhattan_distance(p1 : tuple, p2 : tuple): 
    return np.abs(p2[1] - p1[1]) + np.abs(p2[0] - p1[0])

def a_star(G : Graph, start : tuple, goal : tuple) -> list:
    astar = AStar(G, start, goal)
    return astar.search()


class AStar():
    def __init__(self, G : Graph, start : tuple, goal : tuple) -> None:
        self.admissible_heuristic = manhattan_distance
        self.goal = goal
        self.start = start
        self.get_neighbors = G.get_neighbors

    def reconstruct_path(self, came_from : dict, current : tuple) -> list:
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self) -> Union[list, bool]:
        """
        low level search 
        """
        initial_state = self.start
        step_cost = 1
        
        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        g_score = {} 
        g_score[initial_state] = 0

        f_score = {} 

        f_score[initial_state] = self.admissible_heuristic(initial_state, initial_state)

        while open_set:
            temp_dict = {open_item:f_score.setdefault(open_item, float("inf")) for open_item in open_set}
            current = min(temp_dict, key=temp_dict.get)

            # If current state is goal, return path
            if self.goal == current:
                return self.reconstruct_path(came_from, current)

            # Add current to closed and remove from open sets
            open_set -= {current}
            closed_set |= {current}

            # Get list of neighbors from current node
            neighbor_list = self.get_neighbors(current, True)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                if neighbor not in open_set:
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, current)
        print(len(closed_set))
        return False