import numpy as np
from typing import List, Tuple, Dict
from ...agent import AgentLoader

def random_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int, 
                   idx_to_task_id: Dict[int, int], start_locs: List[Tuple[int, int]], goal_locs: List[Tuple[int, int]], 
                   cost_lookup: Dict[Tuple[int, int, int, int], int]) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
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
        
    Returns:
        Tuple containing:
        - Updated AgentLoader with removed allocations
        - Updated allocations list
        - List of removed allocations
    """
    # Randomly select allocations to remove
    num_to_remove = min(num_to_remove, len(allocations))
    selected_indices = np.random.choice(len(allocations), num_to_remove, replace=False)
    selected_allocations = [allocations[i] for i in selected_indices]

    # print(f"Selected allocations: {selected_allocations}")

    # Track all allocations that need to be removed
    all_removed_allocations = []
    
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
                        break
    
    return Rs, allocations
