import numpy as np
from typing import Union

from .graph import Graph
from .agent import Agent

def get_assigned_task_id(Ra : list, robot_id : int) -> int:
    for assignment in Ra:
        if assignment[-1] == robot_id:
            return assignment[0]

def get_robot_state(Rs : list, robot_id : int) -> tuple:
    for robot in Rs:
        if robot[0] == robot_id:
            return robot[1]
        
def get_unassigned_task_ids(J : set, Ra : list) -> set:
    """Return the set of all task_ids which have not been assigned

    Args:
        J (set): Set of tasks in the form {(task_id, start_loc, goal_loc)}
        Ra (list): Robot-Task Allocation in the form of [(task_id, robot_id), ...]

    Returns:
        set: Set of all unassigned task ids
    """
    assigned_task_ids = set([x[0] for x in Ra])
    task_ids = set([x[0] for x in J])

    unassigned_task_ids = task_ids - assigned_task_ids

    return unassigned_task_ids

def get_assigned_task_ids(Ra : list) -> set:
    """Return the set of all assigned task_ids

    Args:
        Ra (list): Robot-Task Allocation in the form of [(task_id, robot_id), ...]

    Returns:
        set: Set of all assigned task ids
    """
    return set([x[0] for x in Ra])
        
def euclidian_distance(p1 : tuple, p2 : tuple) -> float:
    return ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5

def manhattan_distance(p1 : tuple, p2 : tuple) -> float:
    return np.abs(p2[1] - p1[1]) + np.abs(p2[0] - p1[0])

def compute_agent_idle_steps(agent : Agent) -> int:
    """Count the number of steps in an agent's planned path where it remains stationary.

    Args:
        agent (Agent): The agent whose path_sequence to inspect

    Returns:
        int: Number of steps where the agent's location does not change from the previous step
    """
    if not agent.path_sequence:
        return 0

    idle_steps = 0
    prev_loc = agent.state
    for loc in agent.path_sequence:
        if loc == prev_loc:
            idle_steps += 1
        prev_loc = loc
    return idle_steps

def get_agent_goal_location(agent : Agent) -> tuple:
    """Get an agent's current goal location based on its status.

    Args:
        agent (Agent): The agent whose goal location to look up

    Returns:
        tuple: The pickup location if going to pickup, the delivery location if going to delivery,
               or None if the agent is free or has no task sequence
    """
    if not agent.task_sequence:
        return None

    if agent.status == 1:
        return agent.task_sequence[0][1]
    elif agent.status == 2:
        return agent.task_sequence[0][2]
    return None

def compute_agent_backtrack_steps(agent : Agent) -> int:
    """Count the number of steps in an agent's planned path that increase its L1 distance to its current target.

    Args:
        agent (Agent): The agent whose path_sequence to inspect

    Returns:
        int: Number of steps where the agent's distance to its current task target location increases
    """
    if not agent.path_sequence or not agent.task_sequence:
        return 0

    if agent.status == 1:
        target = agent.task_sequence[0][1]
    elif agent.status == 2:
        target = agent.task_sequence[0][2]
    else:
        return 0

    backtrack_steps = 0
    prev_loc = agent.state
    for loc in agent.path_sequence:
        if manhattan_distance(target, loc) > manhattan_distance(target, prev_loc):
            backtrack_steps += 1
        prev_loc = loc
    return backtrack_steps

def a_star(G : Graph, start : tuple, goal : tuple) -> list:
    astar = AStar(G, start, goal)
    return astar.search()

def construct_distance_hashmap(G : Graph):
    spaces = G.get_all_non_obstacles()
    
    hash_map = {}
    for i, start in enumerate(spaces):
        if i%10 == 0:
            output = "iteration " + str(i) + " out of " + str(len(spaces))
            print(output)
        hash_map[start] = {}
        for goal in spaces:
            if goal in hash_map:
                if start in hash_map[goal]:
                    hash_map[start][goal] = hash_map[goal][start]
                    continue
            hash_map[start][goal] = a_star(G, start, goal)
            
    return hash_map


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
        return False