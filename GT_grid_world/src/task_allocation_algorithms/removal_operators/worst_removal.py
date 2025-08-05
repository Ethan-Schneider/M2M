import numpy as np
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...graph import Graph
from ...utils import manhattan_distance

def worst_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int, 
                 G: Graph, start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], 
                 method: str = "manhattan", cost_lookup: Dict[Tuple[int, int, int, int], int] = None,
                 task_id_to_idx: Dict[int, int] = None, task_start_mask: np.ndarray = None, 
                 task_goal_mask: np.ndarray = None, original_task_start_mask: np.ndarray = None,
                 original_task_goal_mask: np.ndarray = None, agent_start_cost_tensor: np.ndarray = None,
                 agent_task_sequence_time: np.ndarray = None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int, int], int]]:
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
        cost_lookup: Dictionary to store allocation costs
        task_id_to_idx: Mapping from task IDs to tensor indices
        task_start_mask: (N, P) array indicating valid start locations for each task
        task_goal_mask: (N, Q) array indicating valid goal locations for each task
        original_task_start_mask: Original task_start_mask before any modifications
        original_task_goal_mask: Original task_goal_mask before any modifications
        agent_start_cost_tensor: (M, P) array of agent-start costs
        agent_task_sequence_time: (M) array of agent task sequence times
        
    Returns:
        Tuple containing:
        - Updated AgentLoader with removed allocations
        - Updated allocations list
        - Updated cost_lookup dictionary
    """
    if len(allocations) == 0:
        return Rs, allocations, cost_lookup
    
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

    # Track removed locations and tasks
    removed_start_locs = []
    removed_goal_locs = []
    tasks_to_update = []
    changed_agents = []

    # Remove selected allocations from solution
    for agent_idx, task_id, start_idx, goal_idx in worst_allocations:
        changed_agents.append(agent_idx)
        tasks_to_update.append(task_id)
        
        # Remove task from agent's sequence
        for task in Rs.agents[agent_idx].task_sequence:
            if task[0] == task_id:
                Rs.agents[agent_idx].task_sequence.remove(task)
                cost_lookup.pop((agent_idx, task_id, start_idx, goal_idx))
                removed_start_locs.append(start_idx)
                removed_goal_locs.append(goal_idx)
                break
    
    # Update cost elements so that locations are usable for re-allocation
    if task_start_mask is not None and original_task_start_mask is not None:
        for start_idx in removed_start_locs:
            task_start_mask[:, start_idx] = original_task_start_mask[:, start_idx]
    if task_goal_mask is not None and original_task_goal_mask is not None:
        for goal_idx in removed_goal_locs:
            task_goal_mask[:, goal_idx] = original_task_goal_mask[:, goal_idx]

    # Update cost elements of task_start_mask and task_goal_mask for removed task idx
    if task_id_to_idx is not None and task_start_mask is not None and original_task_start_mask is not None:
        for task_id in tasks_to_update:
            if task_id in task_id_to_idx:
                task_start_mask[task_id_to_idx[task_id], :] = original_task_start_mask[task_id_to_idx[task_id], :]
                task_goal_mask[task_id_to_idx[task_id], :] = original_task_goal_mask[task_id_to_idx[task_id], :]

    # Update agent_start_cost_tensor for changed agents
    if agent_start_cost_tensor is not None:
        for agent_idx in changed_agents:
            if len(Rs.agents[agent_idx].task_sequence) == 0:
                agent_pos = Rs.agents[agent_idx].state
            else:
                agent_pos = Rs.agents[agent_idx].task_sequence[-1][2]  # goal location of most recent task
            
            for i, s in enumerate(start_locs):
                if method == "manhattan":
                    cost = -1.0 * manhattan_distance(agent_pos, s)
                elif method == "shortest_path":
                    cost = -1.0 * G.get_distance(agent_pos, s)
                else:
                    raise ValueError(f"Invalid cost calculation method: {method}")
                agent_start_cost_tensor[agent_idx, i] = cost

    # Update agent_task_sequence_time for changed agents
    if agent_task_sequence_time is not None:
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
    
    return Rs, allocations, cost_lookup
