import numpy as np
import random
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...graph import Graph
from ...utils import manhattan_distance

def shaw_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int, 
                 G: Graph, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], 
                 method: str = "manhattan", cost_lookup: Dict[Tuple[int, int, int, int], int] = None, task_id_to_idx: Dict[int, int] = None,
                 omega_1: float = 9.0, omega_2: float = 3.0, agent_start_cost_tensor: np.ndarray = None,
                 start_goal_dist: np.ndarray = None, task_start_mask: np.ndarray = None, task_goal_mask: np.ndarray = None,
                 original_task_start_mask: np.ndarray = None, original_task_goal_mask: np.ndarray = None,
                 agent_task_sequence_time: np.ndarray = None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int, int], int]]:
    """
    Shaw removal operator: randomly choose one task, then remove N-1 tasks in decreasing order of relatedness.
    
    Relatedness is defined as:
    r(task_i, task_j) = omega_1 * (d(g_i, g_j) + d(s_i, s_j)) + omega_2 * (|t(s_i) - t(s_j)| + |t(g_i) - t(g_j)|)
    
    Args:
        Rs: Current solution as AgentLoader
        allocations: List of (agent_idx, task_idx, start_idx, goal_idx) tuples for all allocations
        num_to_remove: Number of allocations to remove
        G: Graph object for shortest path calculations
        start_locs: List of start locations
        goal_locs: List of goal locations
        method: Cost calculation method ("manhattan" or "shortest_path")
        cost_lookup: Dictionary to store allocation costs
        omega_1: Spatial distance weight hyperparameter
        omega_2: Temporal distance weight hyperparameter
        
    Returns:
        Tuple containing:
        - Updated AgentLoader with removed allocations
        - Updated allocations list
        - Updated cost_lookup dictionary
    """
    if len(allocations) == 0:
        return Rs, allocations, cost_lookup, task_start_mask, task_goal_mask, agent_start_cost_tensor, agent_task_sequence_time
    
    # print(f"cost lookup: {cost_lookup}")
    # print(f"number of cost lookup: {len(cost_lookup.keys())}")
    # print(f"start_locs: {start_locs}")
    # print(f"number of start locations: {len(start_locs)}")
    # Randomly choose one task as the seed
    seed_agent_idx, seed_task_id, seed_start_idx, seed_goal_idx = random.choice(list(cost_lookup.keys()))
    
    # Get seed task locations from agent's task sequence
    seed_s_j = start_locs[seed_start_idx]
    seed_g_j = goal_locs[seed_goal_idx]
    
    if seed_s_j is None or seed_g_j is None:
        print(f"Warning: Could not find seed task {seed_task_id} in agent {seed_agent_idx}'s sequence")
        return Rs, allocations, cost_lookup, task_start_mask, task_goal_mask, agent_start_cost_tensor, agent_task_sequence_time
    
    # Calculate relatedness scores for all other tasks
    relatedness_scores = []
    
    for agent_idx, task_id, start_idx, goal_idx in list(cost_lookup.keys()):
        if task_id == seed_task_id:
            continue
        
        # Get task locations from agent's task sequence
        s_i = start_locs[start_idx]
        g_i = goal_locs[goal_idx]
        
        if s_i is None or g_i is None:
            print(f"Warning: Could not find task {task_id} in agent {agent_idx}'s sequence")
            continue
        
        # Calculate spatial distances
        if method == "manhattan":
            d_s = manhattan_distance(s_i, seed_s_j)
            d_g = manhattan_distance(g_i, seed_g_j)
        elif method == "shortest_path":
            d_s = G.get_distance(s_i, seed_s_j)
            d_g = G.get_distance(g_i, seed_g_j)
        else:
            raise ValueError(f"Invalid cost calculation method: {method}")
        
        # Calculate temporal distances
        t_s_i = _calculate_time_at_location(Rs, agent_idx, task_id, start_idx, goal_idx, cost_lookup, is_start=True)
        t_g_i = _calculate_time_at_location(Rs, agent_idx, task_id, start_idx, goal_idx, cost_lookup, is_start=False)
        t_s_j = _calculate_time_at_location(Rs, seed_agent_idx, seed_task_id, seed_start_idx, seed_goal_idx, cost_lookup, is_start=True)
        t_g_j = _calculate_time_at_location(Rs, seed_agent_idx, seed_task_id, seed_start_idx, seed_goal_idx, cost_lookup, is_start=False)
        
        # Calculate relatedness score
        spatial_component = omega_1 * (d_g + d_s)
        temporal_component = omega_2 * (abs(t_s_i - t_s_j) + abs(t_g_i - t_g_j))
        relatedness = spatial_component + temporal_component
        
        relatedness_scores.append((agent_idx, task_id, relatedness))
    
    # Sort by relatedness in decreasing order (highest relatedness first)
    relatedness_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N-1 most related tasks (plus the seed task)
    tasks_to_remove = [(seed_agent_idx, seed_task_id)]  # Start with seed task
    tasks_to_remove.extend([(agent_id, task_id) for agent_id, task_id, _ in relatedness_scores[:num_to_remove-1]])
    
    # Remove selected allocations and all subsequent tasks in each agent's sequence
    removed_start_locs = []
    removed_goal_locs = []
    changed_agents = []

    tasks_to_update = []
    # print(f"tasks to remove: {tasks_to_remove}")
    # print(f"cost lookup: {cost_lookup}")
    # print(f"Current task sequences: {[agent.task_sequence for agent in Rs.agents]}")
    for agent_id, task_id in tasks_to_remove:
        if task_id in tasks_to_update:
            continue
        changed_agents.append(agent_id)

        # print(f"agent id: {agent_id}")
        # print(f"task id: {task_id}")

        # Find the index of the task to remove
        for i, task in enumerate(reversed(Rs.agents[agent_id].task_sequence.copy())):
            # print(f"task: {task}")
            if task[0] == task_id:
                # Rs.agents[agent_id].task_sequence.pop(len(Rs.agents[agent_id].task_sequence) - i - 1)
                Rs.agents[agent_id].task_sequence.pop(-1)
                cost_lookup.pop((agent_id, task[0], start_locs.index(task[1]), goal_locs.index(task[2])))
                removed_start_locs.append(start_locs.index(task[1]))
                removed_goal_locs.append(goal_locs.index(task[2]))
                tasks_to_update.append(task_id)
                break
            else:
                Rs.agents[agent_id].task_sequence.pop(-1)
                # print(f"task start location: {task[1]}")
                # print(f"start_locs: {start_locs}")
                # print(f"start_locs: {start_locs.index(task[1])}")
                # print(f"number of start locations: {len(start_locs)}")
                cost_lookup.pop((agent_id, task[0], start_locs.index(task[1]), goal_locs.index(task[2])))
                removed_start_locs.append(start_locs.index(task[1]))
                removed_goal_locs.append(goal_locs.index(task[2]))
                tasks_to_update.append(task[0])

    # Update cost elements so that locations are usable for re-allocation
    for start_idx in removed_start_locs:
        task_start_mask[:, start_idx] = original_task_start_mask[:, start_idx]
    for goal_idx in removed_goal_locs:
        task_goal_mask[:, goal_idx] = original_task_goal_mask[:, goal_idx]

    # Update cost elements of task_start_mask and task_goal_mask for removed task idx
    # Need to ensure that the specific start and goal locations that are used are still marked in the updated mask
    for task_id in tasks_to_update:
        task_start_mask[task_id_to_idx[task_id], :] = original_task_start_mask[task_id_to_idx[task_id], :]
        task_goal_mask[task_id_to_idx[task_id], :] = original_task_goal_mask[task_id_to_idx[task_id], :]

    # Iterate over allocated tasks and edit masks to not use those locations
    for agent_idx, task_id, start_idx, goal_idx in cost_lookup.keys():
        task_start_mask[:, start_idx] = 0.0
        task_goal_mask[:, goal_idx] = 0.0

    # Iterate over all agents in Rs and for the first task in each agent's task sequence, edit the masks to not use those locations
    for agent in Rs.agents:
        if len(agent.task_sequence) == 0:
            continue
        if agent.task_sequence[0][1] in start_locs:
            task_start_mask[:, start_locs.index(agent.task_sequence[0][1])] = 0.0
        if agent.task_sequence[0][2] in goal_locs:
            task_goal_mask[:, goal_locs.index(agent.task_sequence[0][2])] = 0.0

    for agent_idx in changed_agents:
        if len(Rs.agents[agent_idx].task_sequence) == 0:
            agent_pos = Rs.agents[agent_idx].state
        else:
            agent_pos = Rs.agents[agent_idx].task_sequence[-1][2]  # goal location of most recent task
        
        for i, s in enumerate(start_locs):
            if method == "manhattan":
                cost = manhattan_distance(agent_pos, s)
            elif method == "shortest_path":
                cost = G.get_distance(agent_pos, s)
            agent_start_cost_tensor[agent_idx, i] = cost

    # Update agent_task_sequence_time for changed agents
    for agent_idx in changed_agents:
        agent = Rs.agents[agent_idx]
        if len(agent.task_sequence) == 0:
            # If agent has no tasks, reset to initial state
            agent_task_sequence_time[agent_idx] = 0.0
        else:
            # Recalculate agent_task_sequence_time based on current task sequence
            total_time = 0.0
            if len(agent.task_sequence) == 1:
                # Single task: distance from agent state to start + start to goal
                total_time = G.get_distance(agent.state, agent.task_sequence[0][1]) + G.get_distance(agent.task_sequence[0][1], agent.task_sequence[0][2])
            else:
                # Multiple tasks: include all transitions
                total_time = G.get_distance(agent.state, agent.task_sequence[0][1]) + G.get_distance(agent.task_sequence[0][1], agent.task_sequence[0][2])
                for i in range(1, len(agent.task_sequence)):
                    total_time += G.get_distance(agent.task_sequence[i-1][2], agent.task_sequence[i][1]) + G.get_distance(agent.task_sequence[i][1], agent.task_sequence[i][2])
            agent_task_sequence_time[agent_idx] = total_time

    # Return updated cost elements
    return Rs, allocations, cost_lookup, task_start_mask, task_goal_mask, agent_start_cost_tensor, agent_task_sequence_time

def _calculate_time_at_location(Rs: AgentLoader, agent_idx: int, task_id: int, start_idx: int, goal_idx: int, 
                               cost_lookup: Dict[Tuple[int, int, int, int], int], is_start: bool) -> float:
    """
    Calculate the time when the agent reaches a specific location (start or goal).
    
    Args:
        Rs: AgentLoader containing all agents
        agent_idx: Index of the agent
        task_id: ID of the task
        start_idx: Index of start location
        goal_idx: Index of goal location
        cost_lookup: Dictionary mapping allocations to costs
        is_start: True if calculating time at start location, False for goal location
        
    Returns:
        Time when agent reaches the specified location
    """
    agent = Rs.agents[agent_idx]
    total_time = 0.0
    
    # Calculate time for all tasks before the current task
    for task in agent.task_sequence:
        if task[0] == task_id:
            break
        
        # Find the allocation for this task
        for allocation in cost_lookup.keys():
            if allocation[0] == agent_idx and allocation[1] == task[0]:
                total_time += cost_lookup[allocation]
                break
    
    # If calculating time at goal location, add the cost of the current task
    if not is_start:
        # Find the allocation for the current task
        for allocation in cost_lookup.keys():
            if allocation[0] == agent_idx and allocation[1] == task_id:
                total_time += cost_lookup[allocation]
                break
    
    return total_time
