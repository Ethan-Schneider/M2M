import numpy as np
from typing import List, Tuple, Dict
from ...agent import AgentLoader
from ...graph import Graph
from ...utils import manhattan_distance

def random_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int, 
                   idx_to_task_id: Dict[int, int], start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], 
                   cost_lookup: Dict[Tuple[int, int, int, int], int], G: Graph = None, method: str = "manhattan",
                   task_id_to_idx: Dict[int, int] = None, task_start_mask: np.ndarray = None, 
                   task_goal_mask: np.ndarray = None, original_task_start_mask: np.ndarray = None,
                   original_task_goal_mask: np.ndarray = None, agent_start_cost_tensor: np.ndarray = None,
                   agent_task_sequence_time: np.ndarray = None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int, int], int]]:
    """
    Randomly remove a specified number of allocations from the solution.
    Removes all allocations up to and including the chosen task from each agent's sequence.
    
    Args:
        Rs: Current solution as AgentLoader
        allocations: List of (agent_idx, task_idx, start_idx, goal_idx) tuples for all allocations
        num_to_remove: Number of allocations to remove
        idx_to_task_id: Mapping from tensor indices to task IDs
        start_locs: List of start locations
        goal_locs: List of goal locations
        cost_lookup: Dictionary to store allocation costs
        G: Graph object for shortest path calculations
        method: Cost calculation method ("manhattan" or "shortest_path")
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
    # Randomly select allocations to remove
    num_to_remove = min(num_to_remove, len(allocations))
    selected_indices = np.random.choice(len(allocations), num_to_remove, replace=False)
    selected_allocations = [allocations[i] for i in selected_indices]

    # print(f"Selected allocations: {selected_allocations}")

    # Track all allocations that need to be removed
    all_removed_allocations = []
    removed_start_locs = []
    removed_goal_locs = []
    tasks_to_update = []
    changed_agents = []
    
    # For each selected allocation, remove all allocations up to and including that task
    for agent_idx, task_id, start_idx, goal_idx in selected_allocations:
        agent = Rs.agents[agent_idx]
        tasks_to_remove = []
        
        # Find the index of the task to remove
        task_index = -1
        for i, task in enumerate(agent.task_sequence):
            if task[0] == task_id:
                task_index = i
                break
        
        if task_index != -1:
            # Remove the selected task and all tasks that come after it in the sequence
            tasks_to_remove = agent.task_sequence[task_index:]
            agent.task_sequence = agent.task_sequence[:task_index]
            
            # Add these tasks to the list of removed allocations
            for task in tasks_to_remove:
                # Find the corresponding allocation in the allocations list
                for allocation in allocations[:]:  # Create a copy to iterate
                    if allocation[0] == agent_idx and allocation[1] == task[0]:
                        all_removed_allocations.append(allocation)
                        allocations.remove(allocation)
                        cost_lookup.pop(allocation, None)  # Use allocation directly, with None as default
                        removed_start_locs.append(allocation[2])
                        removed_goal_locs.append(allocation[3])
                        tasks_to_update.append(task[0])
                        break
            
            if agent_idx not in changed_agents:
                changed_agents.append(agent_idx)
    
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
    if agent_start_cost_tensor is not None and G is not None:
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
    if agent_task_sequence_time is not None and G is not None:
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
