import numpy as np
from typing import List, Tuple
from ...agent import AgentLoader

def random_removal(Rs: AgentLoader, allocations: List[Tuple[int, int, int, int]], num_to_remove: int) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]]]:
    """
    Randomly remove a specified number of allocations from the solution.
    
    Args:
        solution: Current solution as AgentLoader
        allocations: List of (agent_idx, task_idx, start_idx, goal_idx) tuples for all allocations
        num_to_remove: Number of allocations to remove
        
    Returns:
        List of (agent_idx, task_idx, start_idx, goal_idx) tuples for removed allocations
    """
    # Collect all allocations that can be removed (skip current tasks)
    removable_allocations = []
    for agent_idx, agent in enumerate(Rs.agents):
        if len(agent.task_sequence) > 1:  # Skip if only current task
            for task_idx in range(1, len(agent.task_sequence)):  # Skip first task (current)
                task = agent.task_sequence[task_idx]
                # Convert task locations to indices in cost tensor
                start_idx = task[1]  # start location tuple
                goal_idx = task[2]   # goal location tuple
                removable_allocations.append((agent_idx, task_idx, start_idx, goal_idx))
    
    if not removable_allocations:
        return Rs, allocations
    
    # Randomly select allocations to remove
    num_to_remove = min(num_to_remove, len(removable_allocations))
    selected_indices = np.random.choice(len(removable_allocations), num_to_remove, replace=False)
    removed_allocations = [removable_allocations[i] for i in selected_indices]
    
    # Sort removals in reverse order by task_idx to avoid index shifting issues
    removed_allocations.sort(key=lambda x: (x[0], -x[1]))
    
    #Update allocations by removing the selected allocations
    for agent_idx, task_idx, start_idx, goal_idx in removed_allocations:
        allocations = [alloc for alloc in allocations if alloc != (agent_idx, task_idx, start_idx, goal_idx)]

    # Remove selected allocations from solution
    for agent_idx, task_idx, _, _ in removed_allocations:
        # Remove task from agent's sequence
        Rs.agents[agent_idx].task_sequence.pop(task_idx)
    
    return Rs, allocations
