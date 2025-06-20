import numpy as np
from typing import List, Tuple
from ...agent import AgentLoader
from ...graph import Graph
from ...utils import manhattan_distance

def worst_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int, 
                 G: Graph, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], 
                 method: str = "manhattan") -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
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
    
    # Calculate cost for each allocation
    allocation_costs = []
    for agent_idx, task_idx, start_idx, goal_idx in allocations:
        agent = Rs.agents[agent_idx]
        
        # Calculate cost from agent's current state to start location
        if method == "manhattan":
            agent_to_start_cost = manhattan_distance(agent.state, start_locs[start_idx])
            start_to_goal_cost = manhattan_distance(start_locs[start_idx], goal_locs[goal_idx])
        else:  # shortest_path
            print(f"Start idx: {start_idx}")
            print(f"Start locs: {start_locs}")
            print(f"Start loc: {start_locs[start_idx]}")
            agent_to_start_cost = G.get_distance(agent.state, start_locs[start_idx])
            start_to_goal_cost = G.get_distance(start_locs[start_idx], goal_locs[goal_idx])
        
        total_cost = agent_to_start_cost + start_to_goal_cost
        allocation_costs.append((total_cost, agent_idx, task_idx, start_idx, goal_idx))
    
    # Sort by cost in descending order (highest cost first)
    allocation_costs.sort(reverse=True, key=lambda x: x[0])
    
    # Select the worst (highest cost) allocations
    worst_allocations = []
    for i in range(min(num_to_remove, len(allocation_costs))):
        _, agent_idx, task_idx, start_idx, goal_idx = allocation_costs[i]
        worst_allocations.append((agent_idx, task_idx, start_idx, goal_idx))
    
    # Remove selected allocations from allocations list
    for allocation in worst_allocations:
        if allocation in allocations:
            allocations.remove(allocation)

    # Remove selected allocations from solution
    for agent_idx, task_id, _, _ in worst_allocations:
        # Remove task from agent's sequence
        for task in Rs.agents[agent_idx].task_sequence:
            if task[0] == task_id:
                Rs.agents[agent_idx].task_sequence.remove(task)
                break
    
    return Rs, allocations, worst_allocations
