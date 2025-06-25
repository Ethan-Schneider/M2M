import numpy as np
import random
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...graph import Graph
from ...utils import manhattan_distance

def shaw_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int, 
                 G: Graph, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], 
                 method: str = "manhattan", cost_lookup: Dict[Tuple[int, int, int, int], int] = None,
                 omega_1: float = 9.0, omega_2: float = 3.0) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
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
        - List of removed allocations
    """
    if len(allocations) == 0:
        return Rs, allocations, []
    
    # Randomly choose one task as the seed
    seed_allocation = random.choice(allocations)
    seed_agent_idx, seed_task_id, seed_start_idx, seed_goal_idx = seed_allocation
    
    # Get seed task locations from agent's task sequence
    seed_s_j = None
    seed_g_j = None
    for task in Rs.agents[seed_agent_idx].task_sequence:
        if task[0] == seed_task_id:
            seed_s_j = task[1]  # start location
            seed_g_j = task[2]  # goal location
            break
    
    if seed_s_j is None or seed_g_j is None:
        print(f"Warning: Could not find seed task {seed_task_id} in agent {seed_agent_idx}'s sequence")
        return Rs, allocations, []
    
    # Calculate relatedness scores for all other tasks
    relatedness_scores = []
    
    for allocation in allocations:
        if allocation == seed_allocation:
            continue
            
        agent_idx, task_id, start_idx, goal_idx = allocation
        
        # Get task locations from agent's task sequence
        s_i = None
        g_i = None
        for task in Rs.agents[agent_idx].task_sequence:
            if task[0] == task_id:
                s_i = task[1]  # start location
                g_i = task[2]  # goal location
                break
        
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
        
        relatedness_scores.append((allocation, relatedness))
    
    # Sort by relatedness in decreasing order (highest relatedness first)
    relatedness_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N-1 most related tasks (plus the seed task)
    tasks_to_remove = [seed_allocation]  # Start with seed task
    tasks_to_remove.extend([allocation for allocation, _ in relatedness_scores[:num_to_remove-1]])
    
    # Remove selected allocations and all subsequent tasks in each agent's sequence
    all_removed_allocations = []
    
    for agent_idx, task_id, start_idx, goal_idx in tasks_to_remove:
        agent = Rs.agents[agent_idx]
        tasks_to_remove_from_agent = []
        
        # Find the index of the task to remove
        task_index = -1
        for i, task in enumerate(agent.task_sequence):
            if task[0] == task_id:
                task_index = i
                break
        
        if task_index != -1:
            # Remove the selected task and all tasks that come after it in the sequence
            tasks_to_remove_from_agent = agent.task_sequence[task_index:]
            agent.task_sequence = agent.task_sequence[:task_index]
            
            # Add these tasks to the list of removed allocations
            for task in tasks_to_remove_from_agent:
                # Find the corresponding allocation in the allocations list
                for allocation in allocations[:]:  # Create a copy to iterate
                    if allocation[0] == agent_idx and allocation[1] == task[0]:
                        all_removed_allocations.append(allocation)
                        allocations.remove(allocation)
                        if cost_lookup is not None:
                            cost_lookup.pop(allocation, None)
                        break
    return Rs, allocations

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
