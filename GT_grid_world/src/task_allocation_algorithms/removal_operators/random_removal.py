import numpy as np
from typing import List, Tuple
from ...agent import AgentLoader

def random_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Randomly remove a specified number of allocations from the solution.
    
    Args:
        solution: Current solution as AgentLoader
        allocations: List of (agent_idx, task_idx, start_idx, goal_idx) tuples for all allocations
        num_to_remove: Number of allocations to remove
        
    Returns:
        List of (agent_idx, task_idx, start_idx, goal_idx) tuples for removed allocations
    """
    # Randomly select allocations to remove
    num_to_remove = min(num_to_remove, len(allocations))
    selected_indices = np.random.choice(len(allocations), num_to_remove, replace=False)
    removed_allocations = [allocations[i] for i in selected_indices]

    #Update allocations by removing the selected allocations
    for allocation in removed_allocations:
        allocations.remove(allocation)

    # Remove selected allocations from solution
    for agent_idx, task_id, _, _ in removed_allocations:
        # Remove task from agent's sequence
        for task in Rs.agents[agent_idx].task_sequence:
            if task[0] == task_id:
                Rs.agents[agent_idx].task_sequence.remove(task)
                break
    
    return Rs, allocations, removed_allocations
