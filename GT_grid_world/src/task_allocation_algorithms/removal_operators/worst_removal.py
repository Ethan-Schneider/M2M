import numpy as np
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...graph import Graph
from ...utils import manhattan_distance

def worst_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int, 
                 G: Graph, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], 
                 method: str = "manhattan", cost_lookup: Dict[Tuple[int, int, int, int], int] = None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Remove the worst (highest cost) allocations from the solution.
    
    Args:
        Rs: Current solution as AgentLoader
        allocations: List of (agent_idx, task_idx, start_idx, goal_idx) tuples for all allocations
        num_to_remove: Number of allocations to remove
        G: Graph object for shortest path calculations
        start_locs: List of start locations
        goal_locs: List of goal locations
        method: Cost calculation method ("manhattan" or "shortest_path")
        
    Returns:
        Tuple containing:
        - Updated AgentLoader with removed allocations
        - Updated allocations list
        - List of removed allocations
    """
    if len(allocations) == 0:
        return Rs, allocations, []
    
    cost_lookup_values = list(cost_lookup.values())
    cost_lookup_values.sort(reverse=True)
    # Create list of worst allocations (agent_idx, task_id, start_idx, goal_idx)
    worst_allocations = []
    for i in range(min(num_to_remove, len(cost_lookup_values))):
        worst_allocations.append(list(cost_lookup.keys())[i])

    # Remove selected allocations from allocations list
    for allocation in worst_allocations:
        if allocation in allocations:
            allocations.remove(allocation)

    # Remove selected allocations from solution
    for agent_idx, task_id, start_idx, goal_idx in worst_allocations:
        # Remove task from agent's sequence
        for task in Rs.agents[agent_idx].task_sequence:
            if task[0] == task_id:
                Rs.agents[agent_idx].task_sequence.remove(task)
                cost_lookup.pop((agent_idx, task_id, start_idx, goal_idx))
                break
    
    return Rs, allocations
